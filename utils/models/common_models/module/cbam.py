import torch
from torch import nn
from torch.nn import functional as F
# from ...builder import MODELS


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)     # 反应整体特征
        self.max_pool = nn.AdaptiveMaxPool2d(1)     # 反应最重要的特征
        self.shared_MLP = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )       # 学习通道重要性

        self.sigmoid = nn.Sigmoid()                     # 输出限制在 0 到 1 之间，使得注意力权重可以表示为每个通道的重要性比例。

    def forward(self, x):
        avg_out =self.shared_MLP(self.avg_pool(x))      # 空间压缩为1*1
        max_out =self.shared_MLP(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)        # 计算通道维度的平均值
        max_out, _ = torch.max(x, dim=1, keepdim=True)      # 计算通道维度的最大值
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)                                   # 生成空间特征权重
        return self.sigmoid(x)

# @MODELS.register()
class CBAM(nn.Module):
    def __init__(self, planes):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(planes)
        self.sa = SpatialAttention()

    def forward(self, x):
        x = self.ca(x) * x
        x = self.sa(x) * x
        return x

if __name__ == '__main__':
    img = torch.randn(2, 256, 18, 50)
    net = CBAM(256)
    print(net)
    out = net(img)
    print(out.size())