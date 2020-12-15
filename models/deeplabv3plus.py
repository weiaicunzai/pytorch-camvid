import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthWiseSeperable(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super().__init__()
        """extra batch normalization [75] and ReLU activation are added
        after each 3 × 3 depthwise convolution, similar to MobileNet design [29].
        """
        self.depthwise = nn.Sequential(
                            nn.Conv2d(in_channels, in_channels, kernel_size, groups=in_channels, **kwargs),
                            nn.BatchNorm2d(in_channels),
                            nn.ReLU(inplace=True)
                        )

        self.pointwise = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)

        return x


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super().__init__()
        self.conv1 = DepthWiseSeperable(in_channels, out_channels, kernel_size, padding=1)
        self.conv2 = DepthWiseSeperable(out_channels, out_channels, kernel_size, padding=1)
        self.conv3 = DepthWiseSeperable(out_channels, out_channels, kernel_size, stride=stride, padding=1)

        self.shortcut = nn.Sequential()

        if stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        shortcut = self.shortcut(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x + shortcut


class Xception(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.in_channels = 64

        self.entry_flow1 = self._make_layers([2], [128])
        self.entry_flow2 = self._make_layers([2, 2], [256, 728])
        self.middle_flow = self._make_layers([1] * 16, [728] * 16)
        #Note that we do not employ the multi-grid method [77,78,23],
        #which we found does not improve the performance.
        self.exit_flow1 = nn.Sequential(
            DepthWiseSeperable(728, 728, 3, padding=1),
            DepthWiseSeperable(728, 1024, 3, padding=1),
            DepthWiseSeperable(1024, 1024, 3, stride=2, padding=1),
        )
        self.exit_flow_shortcut = nn.Sequential(
            nn.Conv2d(728, 1024, 1, stride=2),
            nn.BatchNorm2d(1024)
        )

        self.exit_flow2 = nn.Sequential(
            DepthWiseSeperable(1024, 1536, 3, padding=1),
            DepthWiseSeperable(1536, 1536, 3, padding=1),
            DepthWiseSeperable(1536, 2048, 3, padding=1)
        )


    def forward(self, x):
        x = self.stem(x)
        low_level = self.entry_flow1(x)
        x = self.entry_flow2(low_level)
        x = self.middle_flow(x)
        shortcut = self.exit_flow_shortcut(x)
        x = self.exit_flow1(x)
        x = x + shortcut
        x = self.exit_flow2(x)

        return low_level, x

    def _make_layers(self, strides, out_channels):
        assert len(strides) == len(out_channels)
        l = []
        for stride, out in zip(strides, out_channels):
            l.append(BasicBlock(self.in_channels, out, 3, stride=stride))
            self.in_channels = out

        return nn.Sequential(*l)

class DeepLabv3Puls(nn.Module):
    def __init__(self, class_num):
        super().__init__()
        self.backbone = Xception()
        self.aspp = ASPP(2048, 256, [6, 12, 18])
        self.reduction = BasicConv(128,48, 1)

        # We find that after concatenating the Conv2 feature
        # map (before striding) with DeepLabv3 feature map,
        # it is more effective to employ two 3×3 convolution with 256
        # filters than using simply one or three convolutions.Changing the
        # number of filters from 256 to 128 or the kernel size from 3 × 3 to
        # 1×1 degrades performance.
        self.conv = nn.Sequential(
            BasicConv(256 + 48, 256, 3, padding=1),
            BasicConv(256, 256, 3, padding=1),
            BasicConv(256, class_num, 1)
        )



    def forward(self, x):
        size = x.size()
        low_level, x = self.backbone(x)
        x = self.aspp(x)

        # As shown in Tab. 1, reducing the channels of the
        # low-level feature map from the encoder module to either
        # 48 or 32 leads to better performance. We thus adopt
        # [1 × 1, 48] for channel reduction.
        low_level = self.reduction(low_level)

        # The encoder features are first bilinearly upsampled by a
        # factor of 4 and then concatenated with the corresponding low-level
        # features [73] from the network backbone that have the same spatial
        # resolution (e.g., Conv2 before striding in ResNet-101 [25]).
        x = F.interpolate(x, size=low_level.size()[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, low_level], dim=1)
        x = self.conv(x)

        # Note that our proposed DeepLabv3+ model has output stride = 4.
        # We do not pursue further denser output feature map (i.e.,
        # output stride < 4) given the limited GPU resources.
        x = F.interpolate(x, size=size[2:], mode='bilinear', align_corners=False)

        return x


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

def deeplabv3plus(class_num):
    return DeepLabv3Puls(class_num)
