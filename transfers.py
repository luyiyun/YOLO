import random

import torch
from PIL.ImageFilter import BLUR
from torchvision.transforms import Compose
from torchvision.transforms import functional as F


class OnlyImage:
    '''
    将只应用在image上的transforms进行特殊的调整，这样这个transfoms输入和输出
    都是image和labels了，便于和其他label也会发成变化的transforms对接
    '''
    def __init__(self, transfers, return_tuple=False):
        '''
        args:
            transfers: 多个transforms对象或一个transforms对象；
            return_tuple: 如果是true，则返回的是tuple；
        '''
        if isinstance(transfers, (tuple, list)):
            self.transfers = Compose(transfers)
        else:
            self.transfers = transfers
        self.return_tuple = return_tuple

    def __call__(self, inpts):
        img, others = inpts[0], inpts[1:]
        img = self.transfers(img)
        oupts = [img] + list(others)
        if self.return_tuple:
            return tuple(oupts)
        return oupts


class RandomBlur:
    '''
    随机进行模糊操作（github原代码中给blur设定了一个大小(5,5)，这里使用的PIL没
        有这个参数）
    '''
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            img = img.filter(BLUR)
        return img


class RandomHorizontalFlip:
    '''
    随机水平翻转
    '''
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, inpts):
        img, labels, locs = inpts
        if random.random() < self.p:
            img = F.hflip(img)
            w = img.width
            xmin = w - locs[:, 2]
            xmax = w - locs[:, 0]
            locs[:, 0] = xmin
            locs[:, 2] = xmax
        return (img, labels, locs)


class RandomResize:
    '''
    高度固定，随机从width_factor中得到一个factor，并随机将其乘到width上
    '''
    def __init__(self, width_factor=(0.8, 1.2), p=0.5):
        self.width_factor = width_factor
        self.p = p

    def __call__(self, inpts):
        img, labels, locs = inpts
        if random.random() < self.p:
            factor = random.uniform(*self.width_factor)
            w, h = img.size
            img = img.resize((int(w * factor), h))
            scale_tensor = torch.FloatTensor([[factor, 1, factor, 1]])
            locs = locs * scale_tensor
        return (img, labels, locs)


class RandomShift:
    '''
    对图像进行随机平移操作
    '''
    def __init__(self, p=0.5, shift_ratio=((-0.2, 0.2), (-0.2, 0.2))):
        '''
        args：
            p，多大概率会发生平移；
            shift_ratio，((w_left, w_right), (h_bottom, h_top))，其中left和
                bottom是用负值来表示的，这里使用的相对于原图的比例，所以必须
                绝对值小于1；
        '''
        self.p = p
        self.w_ratio, self.h_ratio = shift_ratio

    def __call__(self, inpts):
        img, labels, locs = inpts
        if random.random() < self.p:
            # 图像的平移
            w, h = img.size
            shift_x = random.uniform(w*self.w_ratio[0], w*self.w_ratio[1])
            shift_y = random.uniform(h*self.h_ratio[0], h*self.h_ratio[1])
            shift_img = F.affine(
                img, angle=0., translate=(shift_x, shift_y), scale=1, shear=0)
            # 标记框的平移
            center = (locs[:, 2:] + locs[:, :2]) / 2
            shift_xy = torch.FloatTensor([[shift_x, shift_y]])
            center = center + shift_xy
            mask1 = (center[:, 0] > 0.) & (center[:, 0] < w)
            mask2 = (center[:, 1] > 0.) & (center[:, 1] < h)
            mask = mask1 & mask2  # 只保留那些中心还在图像中的gtbb
            locs_in = locs[mask]
            if len(locs_in) < 0:
                # 如果本次平移导致没有一个gtbb还在图像中，则不进行平移操作，
                # 返回原图
                return img, labels, locs
            locs_shift = torch.FloatTensor([[shift_x, shift_y] * 2])
            locs_in += locs_shift
            locs_in[:, [0, 2]] = locs_in[:, [0, 2]].clamp(0, w)
            locs_in[:, [1, 3]] = locs_in[:, [1, 3]].clamp(0, h)
            labels_in = labels[mask]
            return shift_img, labels_in, locs_in
        return img, labels, locs


class RandomCrop:
    def __init__(self, p=0.5, wh_ratio=(0.6, 0.6)):
        self.p = p
        self.w_ratio, self.h_ratio = wh_ratio

    def __call__(self, inpts):
        img, labels, locs = inpts
        if random.random() < self.p:
            # 图像的crop
            w, h = img.size
            crop_w = random.uniform(self.w_ratio*w, w)
            crop_h = random.uniform(self.h_ratio*h, h)
            crop_x = random.uniform(0, w - crop_w)
            crop_y = random.uniform(0, h - crop_h)
            # gtbb的crop
            center = (locs[:, :2] + locs[:, 2:]) / 2
            center = center - torch.FloatTensor([[crop_x, crop_y]])
            mask1 = (center[:, 0] > 0) & (center[:, 0] < crop_w)
            mask2 = (center[:, 1] > 0) & (center[:, 1] < crop_h)
            mask = mask1 & mask2  # 只保留那些中心还在图像中的gtbb
            locs_in = locs[mask]
            if len(locs_in) == 0:
                # 如果本次平移导致没有一个gtbb还在图像中，则不进行平移操作，
                # 返回原图
                return img, labels, locs
            # 计算距离新的left和bottom的距离从而作为新的locs
            locs_shift = torch.FloatTensor([[crop_x, crop_y, crop_x, crop_y]])
            locs_in -= locs_shift
            locs_in[:, [0, 2]] = locs_in[:, [0, 2]].clamp(0, crop_w)
            locs_in[:, [1, 3]] = locs_in[:, [1, 3]].clamp(0, crop_h)
            labels_in = labels[mask]
            # 把crop img放在此是为了防止当不进行crop的时候不会进行这一步，从而
            #   能够提供一定程度的计算节省
            crop_img = img.crop(
                (
                    int(crop_x),
                    int(crop_y),
                    int(crop_x+crop_w),
                    int(crop_y+crop_h)
                )
            )
            return crop_img, labels_in, locs_in
        return img, labels, locs


def test():
    import numpy as np
    import argparse
    import matplotlib.pyplot as plt
    from torchvision import transforms

    from visual import draw_rect
    from dataset import VOCDataset

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-p', '--phase', default='train',
        help='载入哪一部分的数据，默认是train，还可以是valid、test'
    )
    args = parser.parse_args()

    # 这里和github略有不同，github上是每个变换都各自有0.5的概率是不发生的，而
    #   这里是只要发生变换则三种变换是同时发生的。
    color_transfers = transforms.RandomApply(
        [
            transforms.ColorJitter(
                brightness=(0.5, 1.5), saturation=(0.5, 1.5), hue=(-0.2, 0.2)
            )
        ],
        0.5
    )
    img_transfers = OnlyImage([RandomBlur(), color_transfers])
    all_transfers = Compose([
        RandomHorizontalFlip(), RandomResize(), img_transfers,
        RandomShift(), RandomCrop()
    ])
    dat = VOCDataset(
        'G:/dataset/VOC2012/VOCdevkit/VOC2012/', phase=args.phase,
        drop_diff=False, return_tensor=True, transfers=all_transfers
    )
    for img, labels, locs in dat:
        img = draw_rect(img, locs, labels=labels)
        plt.imshow(np.asarray(img))
        plt.show()


if __name__ == "__main__":
    test()
