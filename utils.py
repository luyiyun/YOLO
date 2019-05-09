import torch


def boxes_iou(boxes1, boxes2):
    '''
    计算两组boxes的IoU，使用的boxes的格式是xxyy；
    args：
        boxes1，size，[N, 4]
        boxes2，size，[M, 4]
    return：
        IoU matrix，size，[N, M]
    '''
    lt = torch.max(
        boxes1[:, None, :2], boxes2[:, :2]
    )
    rb = torch.min(
        boxes1[:, None, 2:], boxes2[:, 2:]
    )
    wh = rb - lt
    wh[wh < 0] = 0
    inter = wh[..., 0] * wh[..., 1]
    area1 = (boxes1[:, 2]-boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2]-boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    area1 = area1.unsqueeze(1).expand_as(inter)
    area2 = area2.unsqueeze(0).expand_as(inter)
    iou = inter / (area1 + area2 - inter)
    return iou


def xywh2xyxy(boxes, cell_size, cell_lt=None):
    '''
    将YOLO风格的xywh转换为YOLO风格的xyxy，这里和retina的区别在于其xy center是
    相对于cell左上角的坐标，所以需要乘以cell的边长。
    args：
        boxes，size=[N, 4]；
        cell_size，cell的边长，一般来说，整张图片的长度一般设定为1；
        cell_lt，理论上，如果我们不使用此参数，得到的xyxy坐标是不准确的，差了
            一个对应cell左上角坐标，但在计算IoU的时候，这个坐标会被约去，所以
            为了方便起见可以设为None；
    returns：
        boxes，size=[N, 4]，xyxy格式的，但注意的是其坐标一般来说是将整个图片的w
            h看成1得到的；
    '''
    xy_min = boxes[:, :2] * cell_size - boxes[:, 2:] / 2 + cell_lt
    xy_max = boxes[:, :2] * cell_size + boxes[:, 2:] / 2 + cell_lt
    if cell_lt is not None:
        xy_min += cell_lt
        xy_max += cell_lt
    return torch.cat([xy_min, xy_max], dim=1)


def nms(boxes, confs, thre=0.5):
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)

    _, order = confs.sort(0, descending=True)
    keep = []
    while order.numel() > 0:
        i = order[0]
        keep.append(i)

        if order.numel() == 1:
            break

        xx1 = x1[order[1:]].clamp(min=x1[i])
        yy1 = y1[order[1:]].clamp(min=y1[i])
        xx2 = x2[order[1:]].clamp(max=x2[i])
        yy2 = y2[order[1:]].clamp(max=y2[i])

        w = (xx2 - xx1).clamp(min=0)
        h = (yy2 - yy1).clamp(min=0)
        inter = w * h

        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        ids = (ovr <= thre).nonzero().squeeze(1)
        if ids.numel() == 0:
            break
        order = order[ids+1]
    return torch.tensor(keep, dtype=torch.long)


if __name__ == "__main__":
    import sys
    import random

    from PIL import Image

    from visual import draw_rect

    if sys.argv[1] == 'iou':
        boxes1 = []
        for i in range(10):
            xmin = random.uniform(0, 1920)
            ymin = random.uniform(0, 1200)
            xmax = random.uniform(xmin, 1920)
            ymax = random.uniform(ymin, 1200)
            boxes1.append(torch.tensor([xmin, ymin, xmax, ymax]))
        boxes2 = []
        for i in range(20):
            xmin = random.uniform(0, 1920)
            ymin = random.uniform(0, 1200)
            xmax = random.uniform(xmin, 1920)
            ymax = random.uniform(ymin, 1200)
            boxes2.append(torch.tensor([xmin, ymin, xmax, ymax]))
        boxes1 = torch.stack(boxes1, dim=0)
        boxes2 = torch.stack(boxes2, dim=0)
        iou = boxes_iou(boxes1, boxes2)
        print(iou)
    elif sys.argv[1] == 'nms':
        boxes = []
        confs = []
        for i in range(100):
            xmin = random.uniform(0, 1920)
            ymin = random.uniform(0, 1200)
            xmax = random.uniform(xmin, 1920)
            ymax = random.uniform(ymin, 1200)
            boxes.append([xmin, ymin, xmax, ymax])
            confs.append(random.uniform(0, 1))
        img = Image.new(mode='RGB', color='black', size=(1920, 1200))
        img = draw_rect(img, boxes, confs)
        img.show()
        # nms
        boxes_t = torch.tensor(boxes, dtype=torch.float)
        confs_t = torch.tensor(confs)
        keep = nms(boxes_t, confs_t, 0.2)
        boxes_t = boxes_t[keep]
        confs_t = confs_t[keep]
        img = Image.new(mode='RGB', color='black', size=(1920, 1200))
        img = draw_rect(img, boxes_t, confs_t)
        img.show()


