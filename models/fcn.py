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

class FCN(nn.Module):
    """The segmentation-equipped VGG net (FCN-VGG16) already
    appears to be state-of-the-art at 56.0 mean IU on val, com-
    pared to 52.6 on test"""
    def __init__(self, class_num, net_type):
        super().__init__()
        """Why pad the input?: The 100 pixel input padding guarantees
        that the network output can be aligned to the input for any
        input size in the given datasets, for instance PASCAL VOC.
        The alignment is handled automatically by net specification
        and the crop layer. It is possible, though less convenient,
        to calculate the exact offsets necessary and do away with this
        amount of padding.
        https://github.com/shelhamer/fcn.berkeleyvision.org
        """
        self.net_type = net_type
        self.class_num = class_num
        self.conv1 = nn.Sequential(
            BasicConv2d(3, 64, 3, padding=100),
            BasicConv2d(64, 64, 3, padding=1)
        )
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)

        self.conv2 = nn.Sequential(
            BasicConv2d(64, 128, 3, padding=1),
            BasicConv2d(128, 128, 3, padding=1)
        )
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)

        self.conv3 = nn.Sequential(
            BasicConv2d(128, 256, 3, padding=1),
            BasicConv2d(256, 256, 3, padding=1),
            BasicConv2d(256, 256, 3, padding=1)
        )
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)

        self.conv4 = nn.Sequential(
            BasicConv2d(256, 512, 3, padding=1),
            BasicConv2d(512, 512, 3, padding=1),
            BasicConv2d(512, 512, 3, padding=1)
        )
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)

        self.conv5 = nn.Sequential(
            BasicConv2d(512, 512, 3, padding=1),
            BasicConv2d(512, 512, 3, padding=1),
            BasicConv2d(512, 512, 3, padding=1)
        )
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)

        # conv6-7
        self.fc6 = BasicConv2d(512, 4096, 7)
        self.fc7 = BasicConv2d(4096, 4096, 1)

        self.score_fc7 = nn.Conv2d(4096, class_num, 1)
        """First row (FCN-32s): Our single-stream net, described
        in Section 4.1, upsamples stride 32 predictions back to
        pixels in a single step."""
        if net_type == '32s':
            self.x32upscore = nn.ConvTranspose2d(class_num, class_num, 64, stride=32)

        #"""Second row (FCN-16s): Combining predictions from both
        #the final layer and the pool4 layer, at stride 16, lets
        #our net predict finer details, while retaining high-level semantic
        #information.
        #"""
        elif net_type == '16s':
            self.x2upscore = nn.ConvTranspose2d(class_num, class_num, 4, stride=2)
            self.x16upscore = nn.ConvTranspose2d(class_num, class_num, 32, stride=16)
            self.score_pool4 = nn.Conv2d(512, class_num, 1)

        #"""Third row (FCN-8s): Additional predictions from pool3,
        #at stride 8, provide further precision.
        #"""
        elif net_type == '8s':
            self.x2upscore1 = nn.ConvTranspose2d(class_num, class_num, 4, stride=2)
            self.x2upscore2 = nn.ConvTranspose2d(class_num, class_num, 4, stride=2)
            self.score_pool3 = nn.Conv2d(256, class_num, 1)
            self.score_pool4 = nn.Conv2d(512, class_num, 1)
            self.x8upscore = nn.ConvTranspose2d(class_num, class_num, 16, stride=8)

        else:
            raise ValueError('wrong net_type value, should be one of "8s", "16s", "32s"')

    def forward(self, x):
        output = self.conv1(x)
        output = self.pool1(output) # 1/2
        output = self.conv2(output)
        output = self.pool2(output) # 1/4
        output = self.conv3(output)
        output = self.pool3(output) # 1/8
        if self.net_type == '8s':
            pool3 = output
            pool3 = self.score_pool3(pool3)

        output = self.conv4(output)
        output = self.pool4(output) # 1/16
        """We add a 1 × 1 convolution
        layer on top of pool4 to produce
        additional class predic-tions.
        """
        if self.net_type != '32s':
            pool4 = output
            pool4 = self.score_pool4(pool4)

        output = self.conv5(output)
        output = self.pool5(output) # 1/32
        output = self.fc6(output)
        output = self.fc7(output)
        output = self.score_fc7(output)

        """We append a 1 × 1 convolution with chan-
        nel dimension 21 to predict scores for each of the PAS-
        CAL classes (including background) at each of the coarse
        output locations, followed by a deconvolution layer to bi-
        linearly upsample the coarse outputs to pixel-dense outputs
        as described in Section 3.3.
        """
        if self.net_type == '32s':
            output = self.x32upscore(output)

        #"""We fuse this output with the predictions computed
        #on top of conv7 (convolutionalized fc7) at stride 32 by
        #adding a 2× upsampling layer and summing 6 both predic-
        #tions (see Figure 3)."""
        elif self.net_type == '16s':
            output = self.x2upscore(output)
            pool4 = pool4[:, :, 5:5 + output.size()[2], 5:5 + output.size()[3]]
            output = output + pool4
            output = self.x16upscore(output)

        #"""We continue in this fashion by fusing predictions from
        #pool3 with a 2× upsampling of predictions fused from
        #pool4 and conv7,"""
        elif self.net_type == '8s':
            output = self.x2upscore1(output)
            pool4 = pool4[:, :, 5:5 + output.size()[2], 5:5 + output.size()[3]]
            output = pool4 + output
            output = self.x2upscore2(output)
            pool3 = pool3[:, :, 9:9 + output.size()[2], 9:9 + output.size()[3]]
            output = pool3 + output
            output = self.x8upscore(output)

        output = output[:, :, 19 : 19 + x.size()[2], 19 : 19 + x.size()[3]]
        return output

net = FCN(11, '8s')
#net = nn.Conv2d(3, 33, 3)
import torch
img = torch.Tensor(2, 3, 100, 100)
#net = BasicConv2d(3, 33, 3, padding=100)
print(img.shape)
print(net(img).shape)

print(sum([p.numel() for p in net.parameters()]))


