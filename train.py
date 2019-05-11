import os
import copy
import json
import platform

import numpy as np
import pandas as pd
import torch
import argparse
from torch.utils.data import DataLoader
import torch.optim as optim
from torchvision import transforms
from torchvision import models
from torchvision.transforms import Compose, Normalize, ToTensor
import progressbar as pb

from dataset import VOCDataset
import transfers as trans
from net import YOLONet
from losses import YOLOLoss
from metrics import mAP


def json_save(obj, fname):
    with open(fname, 'w') as f:
        json.dump(obj, f)


def default_root():
    if platform.system() == 'Windows':
        return 'G:/dataset/VOC2012/VOCdevkit/VOC2012/'
    return '/home/dl/code/caolei/faster_rcnn/VOCdevkit/VOC2007/'


class History:
    '''
    用于储存训练时期结果的对象，其中其有两个最重要的属性：history和best
    history，是一个dict或df，用于储存在训练时期loss和mAP的信息；
    best，用于储存当前训练阶段最好的t个模型及其metrics；
    '''
    def __init__(self, best_num=3, best_metric='mAP', min_better=False):
        self.best_metric = best_metric
        self.verse_f = np.less if min_better else np.greater
        self.history = {'loss': [], 'val_loss': [], 'mAP': []}
        self.hist_keys = self.history.keys()
        self.best = [
            {
                'mAP': 0., 'loss': float('inf'),
                'epoch': -1, 'model_wts': None
            } for _ in range(best_num)
        ]
        self.best_keys = set(self.best[0].keys())

    def update_best(self, **kwargs):
        '''
        将本次得到的结果更新到best属性中
        args是self.best的keys，包括mAP、loss、epoch和
            model_wts，其中model_wts是model的state_dict的deepcopy值
        这里需要注意的是，需要将所有要更新的参数都写上，不然会报错。
        '''
        self.compare_keys(kwargs.keys(), 'best')  # 检查输入是否符合格式
        best_scores = [b[self.best_metric] for b in self.best]
        new_score = kwargs[self.best_metric]
        for i, bs in enumerate(best_scores):
            if self.verse_f(new_score, bs):
                self.best.insert(i, kwargs)
                self.best.pop()
                break

    def update_hist(self, **kwargs):
        '''
        更新history属性
        '''
        # self.compare_keys(kwargs.keys(), 'hist')
        # 因为hist的内容需要分别在train和valid阶段进行更新，所以一次更新是无法
        #   完成的，就没有设置检查机制
        for k, v in kwargs.items():
            self.history[k].append(v)

    def compare_keys(self, new_keys, func='best'):
        '''
        检查输入是否符合格式，即是否是我们要求的那些keys；
        args:
            new_keys，是输入的参数名；
            func，我们要比较的是哪个函数的参数s，可以选择best or hist；
        '''
        new_keys = set(new_keys)
        if func == 'best':
            old_keys = self.best_keys
        elif func == 'hist':
            old_keys = self.hist_keys
        diff = len(new_keys.symmetric_difference(old_keys))
        if diff > 0:
            raise ValueError(
                '输入的参数和要求的不一致，要求的参数是%s' % str(old_keys)
            )


def train(
    net, criterion, optimizer, dataloaders, epoch,
    device=torch.device('cuda:0'), hist_args={}
):
    history = History(**hist_args)

    for e in range(epoch):
        epoch_loss = 0.
        for phase in ['train', 'valid']:
            y_encoder = dataloaders[phase].dataset.y_encoder
            if phase == 'train':
                net.train()
                prefix = '%d, Training: ' % e
            else:
                net.eval()
                prefix = '%d, Validation: ' % e
                # validation时需要记录所有的真实标签和预测
                epoch_label, epoch_loc = [], []
                epoch_res_c, epoch_res_s, epoch_res_l = [], [], []
            for batch in pb.progressbar(dataloaders[phase], prefix=prefix):
                imgs, targets = batch[0].to(device), batch[-1].to(device)
                with torch.set_grad_enabled(phase == 'train'):
                    preds = net(imgs)
                    loss = criterion(preds, targets)
                    if phase == 'train':
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                    else:
                        # 如果是eval，需要记录所有的预测，用于计算mAP
                        img_size = list(imgs.shape[-1:-3:-1])
                        epoch_label.extend(batch[1])
                        epoch_loc.extend(batch[2])
                        for pred in preds:
                            res_c, res_s, res_l = y_encoder.decode(
                                pred, img_size)
                            epoch_res_c.append(res_c)
                            epoch_res_s.append(res_s)
                            epoch_res_l.append(res_l)
                with torch.no_grad():
                    epoch_loss += loss.item()
            # 整个epoch的所有batches循环结束
            with torch.no_grad():
                num_batches = len(dataloaders[phase].dataset)
                epoch_loss /= num_batches
                if phase == 'train':
                    history.update_hist(loss=epoch_loss)
                    print('%d, %s, loss: %.4f' % (e, phase, epoch_loss))
                else:
                    aps, map_score = mAP(
                        epoch_label, epoch_loc, epoch_res_c,
                        epoch_res_s, epoch_res_l
                    )
                    history.update_hist(val_loss=epoch_loss, mAP=map_score)
                    if map_score is np.nan:
                        map_score = 0
                    history.update_best(
                        mAP=map_score, loss=epoch_loss, epoch=e,
                        model_wts=copy.deepcopy(net.state_dict())
                    )
                    print(
                        '%d, %s, loss: %.4f, mAP: %.4f' %
                        (e, phase, epoch_loss, map_score)
                    )
    if 'valid' in dataloaders.keys():
        net.load_state_dict(history.best[0]['model_wts'])
        print(
            'valid best loss: %.4f, best mAP: %.4f' %
            (history.best[0]['loss'], history.best[0]['mAP'])
        )
    return history


def test(
    net, dataloader, criterion=None, evaluate=True, predict=True,
    device=torch.device('cuda:0')
):
    '''
    使用训练好的net进行预测或评价
    args：
        net，训练好的RetinaNet对象；
        dataloader，需要进行预测或评价的数据集的dataloader；
        criterion，计算损失的函数，如果是None则不计算loss；
        evaluate，如果是True则进行模型的评价（即计算mAP），如果
            是False，则只进行预测或计算loss；
        predict，是否进行预测，如果True则会返回预测框的类别和坐标，如果False，
            则不会返回；
    returns：
        all_labels_pred, all_markers_pred，如果predict=True，则返回预测的每张图
            片的预测框的标签和loc；
        losses, cls_losses, loc_losses，如果criterion不是None，则得到其3个loss；
        APs, mAP_score，如果evaluate=True，则返回每一类的AP值和其mAP；
    '''
    y_encoder = dataloader.dataset.y_encoder
    y_encoder_mode = dataloader.dataset.out
    assert (evaluate and y_encoder_mode in ['obj', 'all']) or not evaluate
    assert (criterion is not None and y_encoder_mode in ['all', 'encode']) or \
        criterion is None

    print('Testing ...')
    results = []
    with torch.no_grad():
        losses = 0.
        all_labels_pred = []  # 预测的labels，
        all_scores_pred = []
        all_markers_pred = []
        all_labels_true = []
        all_markers_true = []
        for batch in pb.progressbar(dataloader):
            imgs = batch[0].to(device)
            if y_encoder_mode in ['all', 'obj']:
                labels_true, locs_true = batch[1:3]
            if y_encoder_mode in ['all', 'encode']:
                targets = batch[-1].to(device)
            preds = net(imgs)
            if criterion is not None:
                loss = criterion(preds, targets)
                losses += loss.item()
            if predict or evaluate:
                img_size = list(imgs.shape[-1:-3:-1])
                for pred in preds:
                    res_c, res_s, res_l = y_encoder.decode(pred, img_size)
                    all_labels_pred.append(res_c)
                    all_scores_pred.append(res_s)
                    all_markers_pred.append(res_l)
            if evaluate:
                all_labels_true.extend(labels_true)
                all_markers_true.extend(locs_true)
        if predict:
            results.append((all_labels_pred, all_markers_pred))
        if criterion is not None:
            losses = losses / len(dataloader.dataset)
            results.append(losses)
        if evaluate:
            # 使用chain将两层的嵌套list变成一层，符合mAP函数的输出要求
            APs, mAP_score = mAP(
                all_labels_true, all_markers_true, all_labels_pred,
                all_scores_pred, all_markers_pred,
            )
            results.append((APs, mAP_score))
    return tuple(results)


def main():
    # ---------------------------- 命令行参数 ----------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'save_name', default='res1',
        help='保存在save_dir上的文件夹名称'
    )
    parser.add_argument(
        '-sd', '--save_dir', default='./YOLOresults',
        help="结果保存的根目录，默认是./YOLOresults"
    )
    parser.add_argument(
        '-dr', '--data_root', default=default_root(),
        help=(
            "数据所在的路径，用于VOCDataset，默认是使用default_root，"
            "根据操作系统来进行选择"
        )
    )
    parser.add_argument(
        '-S', default=7, type=int, help='图像在w和h上被平分为SxS个cells，默认是7'
    )
    parser.add_argument(
        '-B', default=2, type=int, help='每个cell对应B个bbox，默认是2'
    )
    parser.add_argument(
        '-nc', '--num_class', default=20, type=int, help='类别数，默认是20'
    )
    parser.add_argument(
        '-ct', '--conf_thre', default=0.1, type=float,
        help='confidence的界限，默认是0.1'
    )
    parser.add_argument(
        '-nt', '--nms_thre', default=0.5, type=float,
        help='nms的界限，默认是0.5'
    )
    parser.add_argument(
        '-bs', '--batch_size', default=2, type=int, help='batch size，默认是2'
    )
    parser.add_argument(
        '-is', '--input_size', default=(448, 448), nargs=2, type=int,
        help='进入网络的图片大小, 默认是448,448, w x h'
    )
    parser.add_argument(
        '-bb', '--backbone', default='resnet50',
        help='使用的backbone网络，默认是resnet50'
    )
    parser.add_argument(
        '-l', '--lambda2', default=(5., 0.5), nargs=2, type=float,
        help=(
            "用于计算loss时对loc loss和背景类的confidence"
            "回归loss分配的权重，默认是5.0和0.5")
    )
    parser.add_argument(
        '-lr', '--learning_rate', default=0.01, type=float,
        help="学习率，默认是0.01"
    )
    parser.add_argument(
        '-e', '--epoch', default=100, type=int,
        help='epoch的数量，默认是100'
    )
    args = parser.parse_args()

    # ---------------------------- 读取数据 ----------------------------
    # 数据增强
    color_transfers = transforms.RandomApply(
        [
            transforms.ColorJitter(
                brightness=(0.5, 1.5), saturation=(0.5, 1.5), hue=(-0.2, 0.2)
            )
        ],
        0.5
    )
    transfer1 = trans.OnlyImage([trans.RandomBlur(), color_transfers])
    aug_trans = Compose([
        trans.RandomHorizontalFlip(), trans.RandomResize(), transfer1,
        trans.RandomShift(), trans.RandomCrop()
    ])
    # 数据预处理
    means = [123, 117, 104]
    resize = trans.Resize(args.input_size)
    pre_trans = trans.OnlyImage([
        ToTensor(),
        Normalize(means, [1, 1, 1]),
    ])
    pre_trans = Compose([resize, pre_trans])
    # 组合transforms
    train_trans = Compose([aug_trans, pre_trans])
    test_trans = pre_trans
    # 设定YEncoder的参数
    y_encoder_args = {
        'S': args.S, 'B': args.B, 'C': args.num_class,
        'conf_thre': args.conf_thre, 'nms_thre': args.nms_thre
    }
    # 数据集
    datasets = {
        'train': VOCDataset(
            args.data_root, phase='train',
            drop_diff=True, return_tensor=True, transfers=train_trans,
            out='encode', **y_encoder_args
        ),
        'valid': VOCDataset(
            args.data_root, phase='val',
            drop_diff=True, return_tensor=True, transfers=test_trans,
            out='all', **y_encoder_args
        ),
        # 因为暂时还没有test数据，暂时使用valid来替代
        'test': VOCDataset(
            args.data_root, phase='val',
            drop_diff=True, return_tensor=True, transfers=test_trans,
            out='all', **y_encoder_args
        )
    }
    # DataLoaders
    dataloaders = {
        k: DataLoader(
            v, batch_size=args.batch_size, shuffle=True,
            collate_fn=v.collate_fn
        ) if k == 'train' else
        DataLoader(
            v, batch_size=args.batch_size, shuffle=False,
            collate_fn=v.collate_fn
        )
        for k, v in datasets.items()
    }

    # ---------------------------- 模型构建 ----------------------------
    if args.backbone == 'resnet50':
        backbone = models.resnet50
    elif args.backbone == 'resnet101':
        backbone = models.resnet101
    net = YOLONet(args.S, args.B, args.num_class, backbone)
    net.cuda()
    criterion = YOLOLoss(args.S, args.B, args.num_class, *args.lambda2)
    optimizer = optim.SGD(
        net.parameters(), lr=args.learning_rate, momentum=0.9)

    # ---------------------------- 模型训练 ----------------------------
    hist = train(
        net, criterion, optimizer, dataloaders, args.epoch,
        device=torch.device('cuda'),
    )

    # ---------------------------- 模型测试 ----------------------------
    test_loss, (aps_test, map_test) = test(
        net, dataloaders['test'], criterion=criterion, evaluate=True,
        predict=False
    )
    print('Testing Results: loss: %.4f, mAP: %.4f' % (test_loss, map_test))

    # ---------------------------- 结果保存 ----------------------------
    save_folder = os.path.join(args.save_dir, args.save_name)
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    # 训练结果保存
    train_res = pd.DataFrame(hist.history)
    train_res.to_csv(os.path.join(save_folder, 'train.csv'))
    # 测试结果保存
    test_res = dict(loss=test_loss, APs=aps_test, mAP=map_test)
    json_save(test_res, os.path.join(save_folder, 'test.json'))
    # 模型保存
    torch.save(hist.best, os.path.join(save_folder, 'model.pth'))
    # 参数保存
    json_save(args.__dict__, os.path.join(save_folder, 'args.json'))


if __name__ == "__main__":
    main()
