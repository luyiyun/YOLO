import torch
import torch.nn as nn
from torchvision.models import resnet50


class YOLONet(nn.Module):

    def __init__(self, S=14, B=2, C=20, backbone=resnet50):
        super(YOLONet, self).__init__()
        self.target_channle = 5 * B + C
        self.bb = backbone(True)
        self.bb = nn.Sequential(*list(self.bb.children())[:-2])
        self.adaptive_pool = nn.AdaptiveAvgPool2d((S, S))
        self.conv_end = nn.Conv2d(
            2048, self.target_channle, kernel_size=3, stride=1,
            padding=1, bias=False)
        self.bn_end = nn.BatchNorm2d(self.target_channle)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.bb(x)
        x = self.adaptive_pool(x)
        x = self.conv_end(x)
        x = self.bn_end(x)
        x = self.sigmoid(x)
        x = x.permute(0, 2, 3, 1)  # (-1, 30, 14, 14) -- > (-1, 14, 14, 30)
        return x


if __name__ == "__main__":
    img = torch.rand(2, 3, 448, 448)
    net = YOLONet()
    target = net(img)
    print(target.size())
