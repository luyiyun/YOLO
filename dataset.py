import os
from xml.etree import ElementTree as ET

import torch
import numpy as np
from PIL import Image
import torch.utils.data as data

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
        '''
        self.label_mapping = label_mapping
        self.drop_diff = drop_diff
        self.return_tensor = return_tensor
        self.transfers = transfers
        self.out = out
        cast_dir = os.path.join(root, 'ImageSets/Main')
        jpg_dir = os.path.join(root, 'JPEGImages')
        xml_dir = os.path.join(root, 'Annotations')
        if classes is None:
            f = os.path.join(cast_dir, phase + '.txt')
            self.imgs_name = np.loadtxt(f, dtype=str)
        else:
            raise NotImplementedError

        if year is not None:
            raise NotImplementedError

        self.img_files = [
            os.path.join(jpg_dir, n+'.jpg') for n in self.imgs_name]
        self.xml_files = [
            os.path.join(xml_dir, n+'.xml') for n in self.imgs_name]

        if self.out in ['all', 'encode']:
            self.y_encoder = YEncoder(**kwargs)

    def __len__(self):
        return len(self.imgs_name)

    def __getitem__(self, idx):
        img = self.img_files[idx]
        xml = self.xml_files[idx]

        img = Image.open(img)

        labels = []
        locs = []
        xml_res = parse_xml(xml)
        for label, loc, diff, trun in zip(*xml_res):
            if not (diff and self.drop_diff):
                if self.label_mapping is not None:
                    label = self.label_mapping[label]
                labels.append(label)
                locs.append(loc)
        if self.return_tensor:
            labels = torch.tensor(labels, dtype=torch.float)
            locs = torch.tensor(locs, dtype=torch.float)
        if self.transfers is not None:
            img, labels, locs = self.transfers([img, labels, locs])
        if self.out == 'obj':
            return img, labels, locs
        elif self.out in ['all', 'encode']:
            if isinstance(img, Image.Image):
                img_size = img.size
            elif isinstance(img, torch.Tensor):
                img_size = img.size()[:2][::-1].tolist()
            preds = self.y_encoder.encode(labels, locs, img_size)
            if self.out == 'encode':
                return img, preds
            return img, labels, locs, preds


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
