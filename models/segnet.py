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
            BasicConv(64, 64)
        )

        self.maxpool = F.max_pool2d_with_indices
        self.softmax = nn.Softmax(dim=1)

    @staticmethod
    def _unpool(input_tensor, indices, target_shape, kernel_size, stride):

        x = F.max_unpool2d(input_tensor, indices, kernel_size, stride)

        diff_h = x.shape[2] - target_shape[2]
        diff_w = x.shape[3] - target_shape[3]

        x = F.pad(x, [diff_w // 2, diff_w - diff_w //
                      2, diff_h // 2, diff_h - diff_h // 2])

        return x

    def forward(self, x):

        en_x1 = self.encoder1(x)
        en_x2_mask, idx1 = self.maxpool(en_x1, 2, 2, return_indices=True)

        en_x2 = self.encoder2(en_x2_mask)
        en_x3_mask, idx2 = self.maxpool(en_x2, 2, 2, return_indices=True)

        en_x3 = self.encoder3(en_x3_mask)
        en_x4_mask, idx3 = self.maxpool(en_x3, 2, 2, return_indices=True)

        en_x4 = self.encoder4(en_x4_mask)
        en_x5_mask, idx4 = self.maxpool(en_x4, 2, 2, return_indices=True)

        en_x5 = self.encoder5(en_x5_mask)
        en_x6_mask, idx5 = self.maxpool(en_x5, 2, 2, return_indices=True)

        de_x5 = self._unpool(en_x6_mask, idx5, en_x5.shape, 2, 2)
        de_x5 = self.decoder5(de_x5)

        de_x4 = self._unpool(en_x5_mask, idx4, en_x4.shape, 2, 2)
        de_x4 = self.decoder4(de_x4)

        de_x3 = self._unpool(en_x4_mask, idx3, en_x3.shape, 2, 2)
        de_x3 = self.decoder3(de_x3)

        de_x2 = self._unpool(en_x3_mask, idx2, en_x2.shape, 2, 2)
        de_x2 = self.decoder2(de_x2)

        de_x1 = self._unpool(en_x2_mask, idx1, en_x1.shape, 2, 2)
        de_x1 = self.decoder1(de_x1)

        x = self.softmax(de_x1)

        return x