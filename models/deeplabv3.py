import torch
import torch.nn as nn
from torch.nn import functional as F


class BasicConv(nn.Module):
    def __init__(self, in_channles, out_channels, kernel_size, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channles, out_channels, kernel_size, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x

class ASPPPooling(nn.Module):
    def __init__(self, in_channles, out_channels):
        super().__init__()
        self.aspp_pooling = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            BasicConv(in_channles, out_channels, 1)
        )

    def forward(self, x):
        """Specifically, we apply global average pooling on the last feature
        map of the model, feed the resulting image-level features to a 1 × 1
        convolution with 256 filters (and batch normalization [38]), and then
        bilinearly upsample the feature to the desired spatial dimension.
        """
        size = x.shape[-2:]
        x = self.aspp_pooling(x)
        x = F.interpolate(x, size=size, mode='bilinear', align_corners=False)

        return x

class ASPP(nn.Module):
    def __init__(self, in_channles, out_channels, rates):
        super().__init__()

        self.aspp_conv1 = BasicConv(in_channles, out_channels, 1)
        self.aspp_conv2 = BasicConv(in_channles, out_channels, 3, padding=rates[0], dilation=rates[0])
        self.aspp_conv3 = BasicConv(in_channles, out_channels, 3, padding=rates[1], dilation=rates[1])
        self.aspp_conv4 = BasicConv(in_channles, out_channels, 3, padding=rates[2], dilation=rates[2])

        self.aspp_pooling = ASPPPooling(in_channles, out_channels)
        self.project = nn.Sequential(
            BasicConv(5 * out_channels, out_channels, 1),
            nn.Dropout(0.5)
        )

    def forward(self, x):
        x1 = self.aspp_conv1(x)
        x2 = self.aspp_conv2(x)
        x3 = self.aspp_conv3(x)
        x4 = self.aspp_conv4(x)
        x5 = self.aspp_pooling(x)

        x = torch.cat([x1, x2, x3, x4, x5], dim=1)
        x = self.project(x)

        return x

class BottleNeck(nn.Module):
    expansion = 4
    def __init__(self, in_channels, out_channels, stride, padding=0, dilation=1):
        super().__init__()
        self.residual = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, stride=stride),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=padding, dilation=dilation),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BottleNeck.expansion, 1),
            nn.BatchNorm2d(out_channels * BottleNeck.expansion),
        )

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * BottleNeck.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BottleNeck.expansion, 1, stride=stride),
                nn.BatchNorm2d(out_channels * BottleNeck.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual(x) + self.shortcut(x))

class ResNet(nn.Module):
    def __init__(self, num_classes, num_blocks, multi_grids, output_stride=8):
        super().__init__()
        self.stem = nn.Sequential(
            BasicConv(3, 64, 7, stride=2, padding=3),
            nn.MaxPool2d(3, 2, 1, ceil_mode=True)
        )

        self.in_channels = 64

       # Stride and dilation
        if output_stride == 8:
            s = [1, 2, 1, 1]
            d = [1, 1, 2, 4]
        elif output_stride == 16:
            s = [1, 2, 2, 1]
            d = [1, 1, 1, 2]

        self.conv1 = self._make_layer(BottleNeck, 64, num_blocks[0], s[0], d[0])
        self.conv2 = self._make_layer(BottleNeck, 128, num_blocks[1], s[1], d[1])
        self.conv3 = self._make_layer(BottleNeck, 256, num_blocks[2], s[2], d[2])
        self.conv4 = self._make_layer(BottleNeck, 512, num_blocks[3], s[3], d[3], multi_grids)

    def forward(self, x):
        x = self.stem(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        return x

    def _make_layer(self, block, out_channels, num_blocks, stride, dilation, multi_grids=None):
        """make resnet layers(by layer i didnt mean this 'layer' was the
        same as a neuron netowork layer, ex. conv layer), one layer may
        contain more than one residual block
        Args:
            block: block type, basic block or bottle neck block
            out_channels: output depth channel number of this layer
            num_blocks: how many blocks per layer
            stride: the stride of the first block of this layer
        Return:
            return a resnet layer
        """
        if multi_grids is None:
            multi_grids = [1 for _ in range(num_blocks)]

        assert len(multi_grids) == num_blocks

        # we have num_block blocks per layer, the first block
        # could be 1 or 2, other blocks would always be 1
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for grids, stride in zip(multi_grids, strides):
            layers.append(block(self.in_channels, out_channels, stride, padding=dilation * grids, dilation=dilation * grids))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

class DeepLabV3(nn.Module):
    def __init__(self, class_num, num_blocks, multi_grids, rates):
        super().__init__()
        self.backbone = ResNet(class_num, num_blocks, multi_grids)
        self.aspp = ASPP(2048, 256, rates)
        self.conv = nn.Conv2d(256, class_num, 1)

    def forward(self, x):
        size = x.shape[-2:]
        x = self.backbone(x)
        x = self.aspp(x)
        x = self.conv(x)
        x = F.interpolate(x, size=size, mode='bilinear', align_corners=False)

        return x



def deeplabv3(num_classes):
    return DeepLabV3(num_classes, num_blocks=[3, 4, 23, 3], multi_grids=[1, 2, 4], rates=[6, 12, 18])




#net = ASPP(30, 255, [6, 12, 18])
#net = ResNet(21, ,  [1, 2, 4])
#net = DeepLabV3(21, [3, 4, 23, 3], [1, 2, 4], )
#net = deeplabv3(21)
#print(net)

#images = torch.Tensor(2, 3, 513, 513)

#res = net(images)
#print(net.modules())
#print(sum([p.numel() for p in net.parameters()]))
#print(res.shape)




