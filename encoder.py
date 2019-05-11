import torch

from utils import xywh2xyxy, nms


class YEncoder:
    '''
    对gtbb进行变换使其能够参与网络的反向传播
    '''
    def __init__(self, S=14, B=2, C=20, conf_thre=0.1, nms_thre=0.5):
        '''
        args:
            S，每张图片使用的网格数量是S^2；
            B，每个cell使用B个bbox；
            C，类别数；
        '''
        self.S = S
        self.B = B
        self.C = C
        self.conf_thre = conf_thre
        self.nms_thre = nms_thre
        self.pred_c = 5 * B + C
        self.cell_size = 1. / S  # 这意味着我们必须把locs归一化到0-1的范围内

    def encode(self, labels, locs, img_size):
        '''
        将图片的gtbb转换为YOLO接受的形式；
        args：
            labels，[N,]，每个obj的标签；
            locs，[N, 4]，每个obj的坐标，xyxy；
            img_size，每张图片的大小，(w, h)；
        returns：
            target，SxSxself.pred_c。
        '''
        # 将gtbb的坐标归一化到0-1的范围内
        img_size = torch.tensor([list(img_size) * 2], dtype=torch.float)
        locs = locs / img_size

        target = torch.zeros(self.S, self.S, self.pred_c)
        wh = locs[:, 2:] - locs[:, :2]
        xy = (locs[:, 2:] + locs[:, :2]) / 2
        for i, (wh_i, xy_i) in enumerate(zip(wh, xy)):
            # 如果center在网格边上，属于上一个网格
            ij = (xy_i / self.cell_size).ceil() - 1
            # 此处放的是两个B的confidence，是有物体的概率*预测和真实框的IoU，这
            #   时暂时只写1，因为之后这个分数可能需要实时计算。
            # 其他位置的cell因为并没有包含object的中心点所有其没有预测的能力，
            #   所以这些位置的cells都是0。
            target[int(ij[1]), int(ij[0]), [4, 9]] = 1
            # 这里每个cell一个类别，注意！！！
            target[int(ij[1]), int(ij[0]), int(labels[i])+10] = 1
            # cell_lt = ij * self.cell_size
            # (xy_i - cell_lt) / self.cell_size
            # 上面两项等价于下面的一个计算
            # 网络输出的是预测框中心位置信息是相对于匹配网格左上角的相对位置
            delta_xy = xy_i / self.cell_size - ij
            target[int(ij[1]), int(ij[0]), [0, 1, 5, 6]] = delta_xy.repeat(2)
            target[int(ij[1]), int(ij[0]), [2, 3, 7, 8]] = wh_i.repeat(2)
        return target

    def decode(self, preds, img_size=None):
        '''
        对得到的self.S x self.S x self.pred_c大小的preds进行decode，得到
            object的类别和坐标
        args：
            preds，self.S x self.S x self.pred_c，是网络的输出；
            img_size，图片的大小，如果不是None，则得到的坐标是真实的坐标，如果
                为None，则得到的坐标是归一化到0-1区间的；
        returns：
            res_c，预测的类别；
            res_s，用于nms使用的score，其代表的含义是在确定为object的条件下属于
                此类的概率，和与真实对象IoU的乘积；
            res_l，预测框的标签；
            （以上得到的都是tensor）
        '''
        confidence = preds[..., [4, 9]]
        mask1 = confidence > self.conf_thre
        mask2 = confidence == confidence.max()
        mask = (mask1 + mask2).gt(0)
        if mask.sum() == 0:
            return None
        indx = mask.nonzero()[:, :2][:, [1, 0]].float()
        # lt = indx * self.cell_size
        # indx[:, 2] = indx[:, 2] *

        p_shape = list(preds.shape)
        preds_locs_conf = preds[:, :, :(self.B*5)].view(
            *p_shape[:-1], self.B, 5)
        preds_locs = preds_locs_conf[..., :4]
        preds_conf = preds_locs_conf[..., 4]
        # 计算输出的class应该有的维度，这里将其重复两次分别对应两个bboxes
        pcs = list(preds.shape[:2]) + [self.B, self.C]
        preds_class = preds[:, :, (self.B*5):].unsqueeze(2).expand(*pcs)

        remain_locs = xywh2xyxy(preds_locs[mask], self.cell_size, indx)
        remain_conf = preds_conf[mask]
        remain_class = preds_class[mask]

        # 进行nms，使用的是预测最大的类别的概率*confidence来作为score
        #   其=pr(class_i)*IoU
        probs, cls_index = remain_class.max(1)
        scores = probs * remain_conf
        keep = nms(remain_locs, scores, self.nms_thre)

        res_c, res_s, res_l = cls_index[keep], scores[keep], remain_locs[keep]
        if img_size is not None:
            res_l = res_l * torch.tensor(
                [list(img_size) * 2], dtype=torch.float)

        return res_c, res_s, res_l


def test():
    import numpy as np
    import argparse
    import matplotlib.pyplot as plt

    from visual import draw_rect
    from dataset import VOCDataset

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-p', '--phase', default='train',
        help='载入哪一部分的数据，默认是train，还可以是valid、test'
    )
    parser.add_argument(
        '-c', '--channle', default=4, type=int,
        help='可视化preds的哪个维度，默认是4，即第一个B的confidence'
    )
    parser.add_argument(
        '-f', '--func', default='encode', choices=['encode', 'decode'],
        help='测试的方法，默认是encode，也可以是decode'
    )
    args = parser.parse_args()

    dat = VOCDataset(
        'G:/dataset/VOC2012/VOCdevkit/VOC2012/', phase=args.phase,
        drop_diff=False, return_tensor=True, out='all'
    )
    for img, labels, locs, preds in dat:
        if args.func == 'encode':
            img = draw_rect(img, locs, labels=labels)
            fig, axes = plt.subplots(ncols=2, figsize=(10, 5))
            axes[0].imshow(np.asarray(img))
            axes[1].imshow(preds[..., args.channle])
            plt.show()
        else:
            res_c, res_s, res_l = dat.y_encoder.decode(preds, img.size)
            print(res_c)
            print(res_s)
            print(res_l)
            print(labels)
            print(locs)
            img = draw_rect(img, res_l)
            fig, axes = plt.subplots(ncols=2, figsize=(10, 5))
            axes[0].imshow(np.asarray(img))
            axes[1].imshow(preds[..., args.channle])
            plt.show()
            break


if __name__ == "__main__":
    test()
