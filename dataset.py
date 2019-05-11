import os
from xml.etree import ElementTree as ET

import torch
import numpy as np
from PIL import Image
import torch.utils.data as data
from torch.utils.data.dataloader import default_collate

from encoder import YEncoder


LABEL_MAPPING = {
    'aeroplane': 0, 'bicycle': 1, 'bird': 2, 'boat': 3, 'bottle': 4,
    'bus': 5, 'car': 6, 'chair': 7, 'cow': 8, 'diningtable': 9, 'dog': 10,
    'horse': 11, 'motorbike': 12, 'person': 13, 'pottedplant': 14, 'sheep': 15,
    'sofa': 16, 'train': 17, 'tvmonitor': 18, 'cat': 19
}


def parse_xml(xml_file):
    '''
    提取xml文件中的信息
    args:
        xml_file，xml文件的路径；
    returns:
        objs_name，每个object的类别，list of string；
        objs_loc，每个object的坐标，是#obj x 4的嵌套list；
        objs_difficult，每个object是否是difficult的，list of 0-1；
        objs_truncated，每个object是否是truncated，list of 0-1；
    '''
    loc_name = ['xmin', 'ymin', 'xmax', 'ymax']
    tree = ET.parse(xml_file)
    root = tree.getroot()
    objs_name = []
    objs_loc = []
    objs_truncated = []
    objs_difficult = []
    for obj in root.iter('object'):
        objs_name.append(obj.findtext('name'))
        objs_truncated.append(int(obj.findtext('truncated')))
        objs_difficult.append(int(obj.findtext('difficult')))
        loc_element = obj.find('bndbox')
        loc = [int(loc_element.findtext(ln)) for ln in loc_name]
        objs_loc.append(loc)
    # check
    # size_elem = root.find('size')
    # w, h = int(size_elem.findtext('width')), int(size_elem.findtext('height'))
    # locs = np.array(objs_loc)
    # index = np.concatenate([
    #     locs[:, :2] < 0, np.expand_dims(locs[:, 2] > w, 1),
    #     np.expand_dims(locs[:, 3] > h, 1)
    # ], axis=1).sum()
    # if index > 0:
    #     import ipdb; ipdb.set_trace()
    return objs_name, objs_loc, objs_difficult, objs_truncated


class VOCDataset(data.Dataset):
    '''
    读取PASCAL VOC数据集构建pytorch dataset对象
    '''
    def __init__(
        self, root, phase='train', classes=None, year=None,
        label_mapping=LABEL_MAPPING, drop_diff=True, return_tensor=True,
        transfers=None, out='obj', **kwargs
    ):
        '''
        args:
            root，VOC数据集的root路径，即此文件夹下是ImageSets等子文件夹；
            phase，读取的是哪一部分的数据，默认是train，还可以是val、test或
                trainval；
            classes，not implement；
            year，not implement；
            label_mapping，直接读取的标签是字符串，需要使用此处提供的dict来转换
                成int型；
            drop_diff，是否将diff的object去掉，默认是True；
            return_tensor，是否将label和loc转换成tensor，默认是True；
            transfers，这里提供对数据集进行处理的transforms，如果是None，则不进
                行变换；
            out，控制输出的类型，其值可以是'all'，'obj'，'encode'
        '''
        assert out in ['all', 'obj', 'encode']
        # 参数整理
        self.label_mapping = label_mapping
        self.drop_diff = drop_diff
        self.return_tensor = return_tensor
        self.transfers = transfers
        self.out = out
        cast_dir = os.path.join(root, 'ImageSets/Main')
        jpg_dir = os.path.join(root, 'JPEGImages')
        xml_dir = os.path.join(root, 'Annotations')
        # 读取文件名
        if classes is None:
            f = os.path.join(cast_dir, phase + '.txt')
            self.imgs_name = np.loadtxt(f, dtype=str)
        else:
            raise NotImplementedError
        if year is not None:
            raise NotImplementedError
        # 补全完整的xml以及jpg路径名
        self.img_files = [
            os.path.join(jpg_dir, n+'.jpg') for n in self.imgs_name]
        self.xml_files = [
            os.path.join(xml_dir, n+'.xml') for n in self.imgs_name]
        # 如果我们要返回encode后的preds，则需要实例化YEncoder对象
        if self.out in ['all', 'encode']:
            self.y_encoder = YEncoder(**kwargs)

    def __len__(self):
        return len(self.imgs_name)

    def __getitem__(self, idx):
        # 读取相应的文件
        img = self.img_files[idx]
        xml = self.xml_files[idx]
        img = Image.open(img)
        # 解析xml文件
        labels = []
        locs = []
        xml_res = parse_xml(xml)
        # 控制一下difficult的object是否读取
        for label, loc, diff, trun in zip(*xml_res):
            if not (diff and self.drop_diff):
                labels.append(label)
                locs.append(loc)
        # 如果有label mapping，则将所有的label进行映射
        if self.label_mapping is not None:
            new_labels = []
            for label, loc in zip(labels, locs):
                new_labels.append(self.label_mapping[label])
            labels = new_labels
        # 一般都要变成tensor（进行test的时候可能不需要）
        if self.return_tensor:
            labels = torch.tensor(labels, dtype=torch.float)
            locs = torch.tensor(locs, dtype=torch.float)
        # 图像处理
        if self.transfers is not None:
            img, labels, locs = self.transfers([img, labels, locs])
        # 根据不同的out来编码
        if self.out == 'obj':
            return img, labels, locs
        elif self.out in ['all', 'encode']:
            # pascal数据集上不同图片的大小可能不一样，所以需要这里读取其size输入
            #   到y_encoder中
            if isinstance(img, Image.Image):
                img_size = img.size
            elif isinstance(img, torch.Tensor):
                img_size = list(img.size()[-1:-3:-1])
            try:
                preds = self.y_encoder.encode(labels, locs, img_size)
            except IndexError:
                import ipdb; ipdb.set_trace()
            if self.out == 'encode':
                return img, preds
            return img, labels, locs, preds

    def collate_fn(self, batch):
        if self.out == 'encode':
            return tuple(default_collate(batch))
        elif self.out == 'obj':
            img_batch = [b[0] for b in batch]
            label_batch = [b[1] for b in batch]
            marker_batch = [b[2] for b in batch]
            return default_collate(img_batch), label_batch, marker_batch
        else:
            img_batch = [b[0] for b in batch]
            label_batch = [b[1] for b in batch]
            marker_batch = [b[2] for b in batch]
            preds_batch = [b[3] for b in batch]
            return (
                default_collate(img_batch),
                label_batch,
                marker_batch,
                default_collate(preds_batch)
            )


def test():
    import argparse
    import matplotlib.pyplot as plt

    from visual import draw_rect

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-f', '--func', default='parse_xml',
        help='进行哪个函数的测试，默认是parse_xml函数'
    )
    parser.add_argument(
        '-p', '--phase', default='train',
        help='载入哪一部分的数据，默认是train，还可以是valid、test'
    )
    args = parser.parse_args()

    if args.func == 'parse_xml':
        xml_file = (
            'G:/dataset/VOC2012/VOCdevkit/VOC2012/'
            'Annotations/2007_000032.xml'
        )
        objs_name, objs_loc, objs_difficult, objs_truncated = parse_xml(
            xml_file)
        print(objs_name)
        print(objs_loc)
        print(objs_difficult)
        print(objs_truncated)
    elif args.func == 'VOCDataset':
        dat = VOCDataset(
            'G:/dataset/VOC2012/VOCdevkit/VOC2012/', phase=args.phase,
            drop_diff=False, return_tensor=False, label_mapping=None)
        print("The VOC Dataset of %s has %i images." % (args.phase, len(dat)))
        for img, labels, locs in dat:
            img = draw_rect(img, locs, labels=labels)
            plt.imshow(np.asarray(img))
            plt.show()


if __name__ == "__main__":
    test()
