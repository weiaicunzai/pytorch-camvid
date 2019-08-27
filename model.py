import torch
import torch.nn as nn

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

    def __init__(self, image_size, input_channels, class_num):
        super().__init__()

        self.down1 = nn.Sequential(
            BasicConv2d(3, 64),
            BasicConv2d(64, 64),
        )

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
            BasicConv2d(512, 256),
            BasicConv2d(256, 256)
        )

        self.upsample2 = UpSample2d(512, 256)

        self.up2 = nn.Sequential(
            BasicConv2d(512, 256),
            BasicConv2d(256, 256)
        )

        self.upsample3 = UpSample2d(512, )

        self.maxpool = nn.MaxPool2d(2, 2)
    
    def forward(self, x):
        """It consists of the repeated application of two 3x3 convolutions 
        (unpadded convolutions), each followed by a rectified linear unit 
        (ReLU) and a 2x2 max pooling operation with stride 2 for downsampling.
        """
        x = self.down1(x)

a = BasicConv2d(3, 10)

t = torch.Tensor(1, 3, 30, 30)
c = a(t)
print(c.shape)

up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
print(up)
print(up.__dir__())