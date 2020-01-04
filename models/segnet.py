import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicConv(nn.Module):
    def __init__(self, input_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(input_channels, out_channels, 3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x

class SegNet(nn.Module):
    def __init__(self, input_channels, class_num):
        super().__init__()

        self.encoder1 = nn.Sequential(
            BasicConv(input_channels, 64),
            BasicConv(64, 64)
        )

        self.encoder2 = nn.Sequential(
            BasicConv(64, 128),
            BasicConv(128, 128)
        )

        self.encoder3 = nn.Sequential(
            BasicConv(128, 256),
            BasicConv(256, 256),
            BasicConv(256, 256)
        )

        self.encoder4 = nn.Sequential(
            BasicConv(256, 512),
            BasicConv(512, 512),
            BasicConv(512, 512)
        )

        self.encoder5 = nn.Sequential(
            BasicConv(512, 512),
            BasicConv(512, 512),
            BasicConv(512, 512)
        )

        self.decoder5 = nn.Sequential(
            BasicConv(512, 512),
            BasicConv(512, 512),
            BasicConv(512, 512)
        )

        self.decoder4 = nn.Sequential(
            BasicConv(512, 512),
            BasicConv(512, 512),
            BasicConv(512, 256)
        )

        self.decoder3 = nn.Sequential(
            BasicConv(256, 256),
            BasicConv(256, 256),
            BasicConv(256, 128)
        )

        self.decoder2 = nn.Sequential(
            BasicConv(128, 128),
            BasicConv(128, 64)
        )

        self.decoder1 = nn.Sequential(
            BasicConv(64, 64),
            BasicConv(64, class_num)
        )

        self.maxpool = nn.MaxPool2d(2, return_indices=True)
        self.unpool = nn.MaxUnpool2d(2)

    def forward(self, x):

        en_x1 = self.encoder1(x)
        fm1 = en_x1.shape
        en_x1, idx1 = self.maxpool(en_x1)

        en_x2 = self.encoder2(en_x1)
        fm2 = en_x2.shape
        en_x2, idx2 = self.maxpool(en_x2)

        en_x3 = self.encoder3(en_x2)
        fm3 = en_x3.shape
        en_x3, idx3 = self.maxpool(en_x3)

        en_x4 = self.encoder4(en_x3)
        fm4 = en_x4.shape
        en_x4, idx4 = self.maxpool(en_x4)

        en_x5 = self.encoder5(en_x4)
        fm5 = en_x5.shape
        en_x5, idx5 = self.maxpool(en_x5)

        de_x5 = self.unpool(en_x5, idx5, output_size=fm5)
        de_x5 = self.decoder5(de_x5)

        de_x4 = self.unpool(de_x5, idx4, output_size=fm4)
        de_x4 = self.decoder4(de_x4)

        de_x3 = self.unpool(de_x4, idx3, output_size=fm3)
        de_x3 = self.decoder3(de_x3)

        de_x2 = self.unpool(de_x3, idx2, output_size=fm2)
        de_x2 = self.decoder2(de_x2)

        de_x1 = self.unpool(de_x2, idx1, output_size=fm1)
        de_x1 = self.decoder1(de_x1)

        return de_x1