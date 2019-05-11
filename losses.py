import torch
import torch.nn as nn

from utils import boxes_iou, xywh2xyxy


class YOLOLoss(nn.Module):
    '''
    这里计算loss，需要注意的是：
        1. 我们计算位置的时候，只使用匹配了gtbb的cell（即包括了gtbb中心的cell）
            来计算；
        2. 计算概率的时候也是只使用上面说的cell进行计算（这里对概率的loss也是
            使用mse）；
        3. 但计算confidence的时候是使用所有的cell（这也是可以理解的，因为这个
            confidence是用来判断哪些cell是预测正确的，所以需要正样本--匹配到gtbb
            的cells和负样本--没有匹配到gtbb的cells都要有），但需要注意的，因为
            负样本特别多所以需要给与负样本部分更小的权重；
        虽然组成正样本需要计算3个部分，负样本需要计算1个部分，但正样本使用相同的
            权重、负样本使用相同的权重，所以实际上需要这样正、负样本分开计算。
    '''
    def __init__(self, S=14, B=2, C=20, l_coord=5., l_noobj=0.5):
        '''
        args:
            S，网格的数量是S^2；
            B，每个cell预测几个bbox；
            C，预测的类别数；
            l_coord，对于loc_loss给与的权重；
            l_noobj，对于没有匹配到gtbb的cell的confidence部分的loss给与的权重；
        '''
        super(YOLOLoss, self).__init__()
        self.S = S
        self.B = B
        self.C = C
        self.no_class = B * 5
        self.cell_channel = B * 5 + C  # 又称为pred_c
        self.l_coord = l_coord
        self.l_noobj = l_noobj
        self.cell_size = 1 / S
        self.mse = nn.MSELoss(reduction='sum')

    def forward(self, pred, target):
        '''
        args：
            pred，预测得到的tensor，batch x self.S x self.S x self.pred_c
            target，将真实的标签进行了encode，batch x self.S x self.S x self.pred_c
        '''
        # 得到batch数和pred所在的device
        N = target.size(0)
        device = pred.device
        # 得到匹配了gtbb的cell的坐标（这里是mask）和没有匹配gtbb的cell的坐标
        coo_mask = target[..., 4] > 0  # batch x S x S
        noo_mask = target[..., 4] == 0  # batch x S x S

        # ----- 正样本部分 -----
        # 把预测分成不同的部分
        # batch*matched_cell_num x pred_c
        coo_pred = pred[coo_mask].view(-1, self.cell_channel)
        # batch*matched_cell_num*B x 5
        box_pred = coo_pred[:, :self.no_class].reshape(-1, 5)
        # batch*matched_cell_num x C
        class_pred = coo_pred[:, self.no_class:]

        # 把标签分成不同的部分
        # batch*matched_cell_num x pred_c
        coo_target = target[coo_mask].view(-1, self.cell_channel)
        # batch*matched_cell_num*B x 5
        box_target = coo_target[:, :self.no_class].reshape(-1, 5)
        # batch*matched_cell_num x C
        class_target = coo_target[:, self.no_class:]

        # ----- 负样本部分 -----
        noo_pred_c = pred[noo_mask].view(-1, self.cell_channel)[:, [4, 9]]
        noo_target_c = target[noo_mask].view(-1, self.cell_channel)[:, [4, 9]]

        # ----- 计算负样本部分 -----
        noo_loss = self.mse(noo_pred_c, noo_target_c)

        # ----- 计算正样本部分 -----
        coo_response_index = []
        coo_not_response_index = []
        boxes_target_iou = []
        for i in range(0, box_target.size(0), self.B):
            # 因为我们是resize成[N, 5]的维度，所以每个cell的bbox有2个
            box1 = box_pred[i:i+self.B]
            # target那里这两个bbox是一样的，所以考虑一个就好了
            box2 = box_target[i].view(-1, 5)
            iou = boxes_iou(
                xywh2xyxy(box1[:, :4], self.cell_size),
                xywh2xyxy(box2[:, :4], self.cell_size)
            )  # [2, 1]
            max_iou, max_index = iou.max(0)
            coo_response_index.append(i + max_index.item())
            for bb in range(self.B):
                if bb != max_index:
                    coo_not_response_index.append(i + bb)
            boxes_target_iou.append(max_iou)
        # 1. 正样本中有关confidencee的loss，包括两部分
        #   因为一个cell会有多个bboxes，只用其中最高的那个bboxes对应confidence，
        #   其confidence=IoU*1，另外的bboxes一律对应0。
        box_pred_response = box_pred[coo_response_index]
        box_pred_not_response = box_pred[coo_not_response_index]
        contain_loss = self.mse(
            box_pred_response[:, 4], torch.tensor(boxes_target_iou).to(device),
        )
        not_contain_loss = self.mse(
            box_pred_not_response[:, 4],
            torch.zeros_like(box_pred_not_response[:, 4], device=device),
        )
        # 2. loc loss
        #   loc loss只使用对应上的bbox来计算（即2个bboxes中和gtbb最大的那个bbox）
        box_target_response = box_target[coo_response_index]
        loc_loss = self.mse(
            box_pred_response[:, :2], box_target_response[:, :2],
        ) + self.mse(
            box_pred_response[:, 2:4].sqrt(),
            box_target_response[:, 2:4].sqrt(),
        )
        # 3. class loss
        #   显然class loss就使用有obj的cell的分类来计算的
        class_loss = self.mse(class_pred, class_target)

        # ----- 所有的loss相加，并除以batch数 -----
        # 为什么contain loss会乘以2？？？？？
        all_loss = (
            self.l_coord * loc_loss + 2 * contain_loss + not_contain_loss +
            self.l_noobj * noo_loss + class_loss
        ) / N

        return all_loss


if __name__ == "__main__":
    a = torch.rand(2, 14, 14, 30)
    b = torch.rand(2, 14, 14, 30)
    yolo_loss = YOLOLoss()
    print(yolo_loss(a, b))
