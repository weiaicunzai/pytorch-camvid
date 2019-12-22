import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class UpSample2d(nn.Module):

    def __init__(self, in_channels, out_channels, scale_factor=2.0):

        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = BasicConv2d(in_channels, out_channels)

    def forward(self, x):
        x = self.up(x)
        x = self.conv(x)

        return x


class UNet(nn.Module):

    def __init__(self, input_channels, class_num):
        super().__init__()

        self.down1 = nn.Sequential(
            BasicConv2d(input_channels, 64),
            BasicConv2d(64, 64),
        )

        #At each downsampling step we double the number of feature
        #channels.
        self.down2 = nn.Sequential(
            BasicConv2d(64, 128),
            BasicConv2d(128, 128)
        )

        self.down3 = nn.Sequential(
            BasicConv2d(128, 256),
            BasicConv2d(256, 256)
        )

        self.down4 = nn.Sequential(
            BasicConv2d(256, 512),
            BasicConv2d(512, 512)
        )

        self.down5 = nn.Sequential(
            BasicConv2d(512, 1024),
            BasicConv2d(1024, 1024)
        )

        self.upsample1 = UpSample2d(1024, 512)
        self.up1 = nn.Sequential(
            BasicConv2d(1024, 512),
            BasicConv2d(512, 512)
        )

        self.upsample2 = UpSample2d(512, 256)
        self.up2 = nn.Sequential(
            BasicConv2d(512, 256),
            BasicConv2d(256, 256)
        )

        self.upsample3 = UpSample2d(256, 128)
        self.up3 = nn.Sequential(
            BasicConv2d(256, 128),
            BasicConv2d(128, 128)
        )

        self.upsample4 = UpSample2d(128, 64)
        self.up4 = nn.Sequential(
            BasicConv2d(128, 64),
            BasicConv2d(64, 64)
        )

        self.output = BasicConv2d(64, class_num)
        self.maxpool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        """It consists of the repeated application of two 3x3 convolutions
        (unpadded convolutions), each followed by a rectified linear unit
        (ReLU) and a 2x2 max pooling operation with stride 2 for downsampling.
        """
        xd1 = self.down1(x)
        x = self.maxpool(xd1)

        xd2 = self.down2(x)
        x = self.maxpool(xd2)

        xd3 = self.down3(x)
        x = self.maxpool(xd3)

        xd4 = self.down4(x)
        x = self.maxpool(xd4)

        x = self.down5(x)

        """Every step in the expansive path consists of an upsampling of the
        feature map followed by a 2x2 convolution (“up-convolution”) that
        halves the number of feature channels, a concatenation with the
        correspondingly cropped feature map from the contracting path, and
        two 3x3 convolutionsimage_size, each followed by a ReLU."""
        xup1 = self.upsample1(x)

        diff_h = xd4.size(2) - xup1.size(2)
        diff_w = xd4.size(3) - xup1.size(3)
        xup1 = F.pad(xup1, [diff_w // 2, diff_w - diff_w //
                            2, diff_h // 2, diff_h - diff_h // 2])
        xup1 = torch.cat([xup1, xd4], dim=1)

        x = self.up1(xup1)
        xup2 = self.upsample2(x)

        diff_h = xd3.size(2) - xup2.size(2)
        diff_w = xd3.size(3) - xup2.size(3)
        xup2 = F.pad(xup2, [diff_w // 2, diff_w - diff_w //
                            2, diff_h // 2, diff_h - diff_h // 2])
        xup2 = torch.cat([xup2, xd3], dim=1)

        x = self.up2(xup2)
        xup3 = self.upsample3(x)

        diff_h = xd2.size(2) - xup3.size(2)
        diff_w = xd2.size(3) - xup3.size(3)
        xup3 = F.pad(xup3, [diff_w // 2, diff_w - diff_w //
                            2, diff_h // 2, diff_h - diff_h // 2])
        xup3 = torch.cat([xup3, xd2], dim=1)

        x = self.up3(xup3)
        xup4 = self.upsample4(x)

        diff_h = xd1.size(2) - xup4.size(2)
        diff_w = xd1.size(3) - xup4.size(3)
        xup4 = F.pad(xup4, [diff_w // 2, diff_w - diff_w //
                            2, diff_h // 2, diff_h - diff_h // 2])
        xup4 = torch.cat([xup4, xd1], dim=1)

        x = self.up4(xup4)
        x = self.output(x)

        return x
