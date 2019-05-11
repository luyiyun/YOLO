import numpy as np
import torch

from utils import boxes_iou


''' 和retina中的相同 '''


def compute_ap(rec, prec):
    mrec = np.concatenate([[0.], rec, [1.]])
    mprec = np.concatenate([[0.], prec, [0.]])
    for i in range(mprec.shape[0] - 1, 0, -1):
        mprec[i-1] = np.maximum(mprec[i-1], mprec[i])
    i = np.where(mrec[1:] != mrec[:-1])[0]  # np.where返回的是2-tuple
    ap = np.sum((mrec[i + 1] - mrec[i]) * mprec[i + 1])
    return ap


def mAP(
    true_cls, true_loc, pred_cls, pred_score, pred_loc, iou_thre=0.5,
    num_class=20, ap_func=compute_ap
):
    device = pred_cls[0].device
    true_cls = [tc.to(device) for tc in true_cls]
    true_loc = [tl.to(device) for tl in true_loc]
    all_classes = torch.cat(pred_cls, dim=0)
    all_scores = torch.cat(pred_score, dim=0)

    aps = []
    num_imgs = len(true_cls)
    for c in range(num_class):
        tp = np.zeros((0,))
        num_true_objs = 0.0
        for i in range(num_imgs):
            # 得到一张图片中这一类的所有预测框
            true_c_mask = true_cls[i] == c
            num_true_objs += true_c_mask.sum()
            pred_c_mask = pred_cls[i] == c
            true_loc_i_c = true_loc[i][true_c_mask]
            pred_loc_i_c = pred_loc[i][pred_c_mask]
            # 用于记录已经匹配到的gtbb的序号，每张图片重新记录
            detected_true_boxes = []
            for d in pred_loc_i_c:
                if true_loc_i_c.size(0) == 0:
                    tp = np.append(tp, 0)
                    continue
                # 计算预测框和所有gtbb的IoU，并去最大的一个作为此预测框的预测对象
                ious = boxes_iou(d.unsqueeze(0), true_loc_i_c).squeeze(0)
                max_iou, max_idx = ious.max(dim=0)
                if max_iou >= iou_thre and max_idx not in detected_true_boxes:
                    tp = np.append(tp, 1)
                    detected_true_boxes.append(max_idx)
                else:
                    tp = np.append(tp, 0)

        # 如果对于某一类，所有图片的gtbb中都没有这一类，则认为此类的ap是0，？
        if num_true_objs == 0.0:
            aps.append(0.)
            continue
        # 依据score进行排序
        _, order = all_scores[all_classes == c].sort(dim=0, descending=True)
        order = order.cpu().numpy()
        tp = tp[order]
        fp = 1 - tp
        # 逐个计算array的前n个元素中fp和tp的个数，这个可以看做在每个元素的间隔间
        #   变化阈值来使的低于此阈值的所有都被预测是0，则计算postive（不管是fp
        #   还是tp）只需要考虑前面就可以了。
        fp = fp.cumsum()
        tp = tp.cumsum()
        # 计算recall和precision
        recall = tp / num_true_objs.item()
        # --这里可能出现预测的里没有postive（比如我们把阈值卡的特别高的时候），
        #   当然这是tp也是0，但分母=0会使得无法计算，所以需要加一个eps来避免
        precision = tp / np.maximum((tp + fp), np.finfo(np.float64).eps)
        aps.append(ap_func(recall, precision))
    return aps, np.mean(aps)


if __name__ == "__main__":
    rec = np.random.rand(10)
    prec = [0.2, 0.3, 0.35, 0.6, 0.8, 0.5, 0.3, 0.2, 0.1]
    print(compute_ap(rec, prec))
