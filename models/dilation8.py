import torch
import torch.nn as nn


class BasicConv2d(nn.Module):

    def __init__(self, input_channels, output_channels, kernel_size, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size, **kwargs)
        self.bn = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x

class FrontEnd(nn.Module):
    def __init__(self, class_num):
        super().__init__()

        """We also remove the padding of the intermediate feature maps.
        Intermediate padding was used in the original classification network,
        but is neither necessary nor justified in dense prediction.
        """
        self.conv1 = nn.Sequential(
            BasicConv2d(3, 64, 3, padding=0),
            BasicConv2d(64, 64, 3, padding=0)
        )
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Sequential(
            BasicConv2d(64, 128, 3, padding=0),
            BasicConv2d(128, 128, 3, padding=0)
        )
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Sequential(
            BasicConv2d(128, 256, 3, padding=0),
            BasicConv2d(256, 256, 3, padding=0),
            BasicConv2d(256, 256, 3, padding=0)
        )
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv4 = nn.Sequential(
            BasicConv2d(256, 512, 3, padding=0),
            BasicConv2d(512, 512, 3, padding=0),
            BasicConv2d(512, 512, 3, padding=0)
        )

        """Specifically, each of these pooling and striding layers was
        removed and convolutions in all subsequent layers were dilated
        by a factor of 2 for each pooling layer that was ablated.
        """
        self.conv5 = nn.Sequential(
            BasicConv2d(512, 512, 3, padding=0, dilation=2),
            BasicConv2d(512, 512, 3, padding=0, dilation=2),
            BasicConv2d(512, 512, 3, padding=0, dilation=2)
        )



        """Thus convolutions in the final layers, which follow both
        ablated pooling layers, are dilated by a factor of 4.
        """
        self.fc6 = BasicConv2d(512, 4096, 7, padding=0, dilation=4)
        self.dropout6 = nn.Dropout(inplace=True)

        self.fc7 = BasicConv2d(4096, 4096, 1, padding=0)
        self.dropout7 = nn.Dropout(inplace=True)

        self.final = nn.Conv2d(4096, class_num, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.pool3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.fc6(x)
        x = self.fc7(x)
        x = self.final(x)

        return x


class Context(nn.Module):
    def __init__(self, class_num):
        super().__init__()

        self.pad1 = nn.ZeroPad2d(1)
        self.pad2 = nn.ZeroPad2d(2)
        self.pad4 = nn.ZeroPad2d(4)
        self.pad8 = nn.ZeroPad2d(8)
        self.pad16 = nn.ZeroPad2d(16)
        self.conv1 = BasicConv2d(class_num, class_num, 3, padding=0, dilation=1)
        self.conv2 = BasicConv2d(class_num, class_num, 3, padding=0, dilation=1)
        self.conv3 = BasicConv2d(class_num, class_num, 3, padding=0, dilation=2)
        self.conv4 = BasicConv2d(class_num, class_num, 3, padding=0, dilation=4)
        self.conv5 = BasicConv2d(class_num, class_num, 3, padding=0, dilation=8)
        self.conv6 = BasicConv2d(class_num, class_num, 3, padding=0, dilation=16)
        self.conv7 = BasicConv2d(class_num, class_num, 3, padding=0, dilation=1)
        self.conv8 = nn.Conv2d(class_num, class_num, 3, padding=0, dilation=1)

    def forward(self, x):
        x = self.pad1(x)
        x = self.conv1(x)

        x = self.pad1(x)
        x = self.conv2(x)

        x = self.pad2(x)
        x = self.conv3(x)

        x = self.pad4(x)
        x = self.conv4(x)

        x = self.pad8(x)
        x = self.conv5(x)

        x = self.pad16(x)
        x = self.conv6(x)

        x = self.pad1(x)
        x = self.conv7(x)

        x = self.pad1(x)
        x = self.conv8(x)

        return x

"""We refer to our complete convolutional
network (front-end + context) as Dilation8,
since the context module has 8 layers."""
class Dilation8(nn.Module):
    def __init__(self, class_num, output_size):
        super().__init__()
        self.frontend = FrontEnd(class_num)
        self.context = Context(class_num)
        self.upsample = nn.Upsample(output_size)

    def forward(self, x):
        x = self.frontend(x)
        x = self.context(x)
        x = self.upsample(x)
        return x