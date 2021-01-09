

import math
import queue
import collections
import threading

_SlavePipeBase = collections.namedtuple('_SlavePipeBase', ['identifier', 'queue', 'result'])


class SlavePipe(_SlavePipeBase):
    """Pipe for master-slave communication."""

    def run_slave(self, msg):
        self.queue.put((self.identifier, msg))
        ret = self.result.get()
        self.queue.put(True)
        return ret


class SyncMaster(object):
    """An abstract `SyncMaster` object.
    - During the replication, as the data parallel will trigger an callback of each module, all slave devices should
    call `register(id)` and obtain an `SlavePipe` to communicate with the master.
    - During the forward pass, master device invokes `run_master`, all messages from slave devices will be collected,
    and passed to a registered callback.
    - After receiving the messages, the master device should gather the information and determine to message passed
    back to each slave devices.
    """

    def __init__(self, master_callback):
        """
        Args:
            master_callback: a callback to be invoked after having collected messages from slave devices.
        """
        self._master_callback = master_callback
        self._queue = queue.Queue()
        self._registry = collections.OrderedDict()
        self._activated = False

    def __getstate__(self):
        return {'master_callback': self._master_callback}

    def __setstate__(self, state):
        self.__init__(state['master_callback'])

    def register_slave(self, identifier):
        """
        Register an slave device.
        Args:
            identifier: an identifier, usually is the device id.
        Returns: a `SlavePipe` object which can be used to communicate with the master device.
        """
        if self._activated:
            assert self._queue.empty(), 'Queue is not clean before next initialization.'
            self._activated = False
            self._registry.clear()
        future = FutureResult()
        self._registry[identifier] = _MasterRegistry(future)
        return SlavePipe(identifier, self._queue, future)

    def run_master(self, master_msg):
        """
        Main entry for the master device in each forward pass.
        The messages were first collected from each devices (including the master device), and then
        an callback will be invoked to compute the message to be sent back to each devices
        (including the master device).
        Args:
            master_msg: the message that the master want to send to itself. This will be placed as the first
            message when calling `master_callback`. For detailed usage, see `_SynchronizedBatchNorm` for an example.
        Returns: the message to be sent back to the master device.
        """
        self._activated = True

        intermediates = [(0, master_msg)]
        for i in range(self.nr_slaves):
            intermediates.append(self._queue.get())

        results = self._master_callback(intermediates)
        assert results[0][0] == 0, 'The first result should belongs to the master.'

        for i, res in results:
            if i == 0:
                continue
            self._registry[i].result.put(res)

        for i in range(self.nr_slaves):
            assert self._queue.get() is True

        return results[0][1]

    @property
    def nr_slaves(self):
        return len(self._registry)


from torch.nn.modules.batchnorm import _BatchNorm

class _SynchronizedBatchNorm(_BatchNorm):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True):
        super(_SynchronizedBatchNorm, self).__init__(num_features, eps=eps, momentum=momentum, affine=affine)

        self._sync_master = SyncMaster(self._data_parallel_master)

        self._is_parallel = False
        self._parallel_id = None
        self._slave_pipe = None

    def forward(self, input):
        # If it is not parallel computation or is in evaluation mode, use PyTorch's implementation.
        if not (self._is_parallel and self.training):
            return F.batch_norm(
                input, self.running_mean, self.running_var, self.weight, self.bias,
                self.training, self.momentum, self.eps)

        # Resize the input to (B, C, -1).
        input_shape = input.size()
        input = input.view(input.size(0), self.num_features, -1)

        # Compute the sum and square-sum.
        sum_size = input.size(0) * input.size(2)
        input_sum = _sum_ft(input)
        input_ssum = _sum_ft(input ** 2)

        # Reduce-and-broadcast the statistics.
        if self._parallel_id == 0:
            mean, inv_std = self._sync_master.run_master(_ChildMessage(input_sum, input_ssum, sum_size))
        else:
            mean, inv_std = self._slave_pipe.run_slave(_ChildMessage(input_sum, input_ssum, sum_size))

        # Compute the output.
        if self.affine:
            # MJY:: Fuse the multiplication for speed.
            output = (input - _unsqueeze_ft(mean)) * _unsqueeze_ft(inv_std * self.weight) + _unsqueeze_ft(self.bias)
        else:
            output = (input - _unsqueeze_ft(mean)) * _unsqueeze_ft(inv_std)

        # Reshape it.
        return output.view(input_shape)

    def __data_parallel_replicate__(self, ctx, copy_id):
        self._is_parallel = True
        self._parallel_id = copy_id

        # parallel_id == 0 means master device.
        if self._parallel_id == 0:
            ctx.sync_master = self._sync_master
        else:
            self._slave_pipe = ctx.sync_master.register_slave(copy_id)

    def _data_parallel_master(self, intermediates):
        """Reduce the sum and square-sum, compute the statistics, and broadcast it."""

        # Always using same "device order" makes the ReduceAdd operation faster.
        # Thanks to:: Tete Xiao (http://tetexiao.com/)
        intermediates = sorted(intermediates, key=lambda i: i[1].sum.get_device())

        to_reduce = [i[1][:2] for i in intermediates]
        to_reduce = [j for i in to_reduce for j in i]  # flatten
        target_gpus = [i[1].sum.get_device() for i in intermediates]

        sum_size = sum([i[1].sum_size for i in intermediates])
        sum_, ssum = ReduceAddCoalesced.apply(target_gpus[0], 2, *to_reduce)
        mean, inv_std = self._compute_mean_std(sum_, ssum, sum_size)

        broadcasted = Broadcast.apply(target_gpus, mean, inv_std)

        outputs = []
        for i, rec in enumerate(intermediates):
            outputs.append((rec[0], _MasterMessage(*broadcasted[i*2:i*2+2])))

        return outputs

    def _compute_mean_std(self, sum_, ssum, size):
        """Compute the mean and standard-deviation with sum and square-sum. This method
        also maintains the moving average on the master device."""
        assert size > 1, 'BatchNorm computes unbiased standard-deviation, which requires size > 1.'
        mean = sum_ / size
        sumvar = ssum - sum_ * mean
        unbias_var = sumvar / (size - 1)
        bias_var = sumvar / size

        self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean.data
        self.running_var = (1 - self.momentum) * self.running_var + self.momentum * unbias_var.data

        return mean, bias_var.clamp(self.eps) ** -0.5

class SynchronizedBatchNorm2d(_SynchronizedBatchNorm):
    r"""Applies Batch Normalization over a 4d input that is seen as a mini-batch
    of 3d inputs
    .. math::
        y = \frac{x - mean[x]}{ \sqrt{Var[x] + \epsilon}} * gamma + beta
    This module differs from the built-in PyTorch BatchNorm2d as the mean and
    standard-deviation are reduced across all devices during training.
    For example, when one uses `nn.DataParallel` to wrap the network during
    training, PyTorch's implementation normalize the tensor on each device using
    the statistics only on that device, which accelerated the computation and
    is also easy to implement, but the statistics might be inaccurate.
    Instead, in this synchronized version, the statistics will be computed
    over all training samples distributed on multiple devices.
    Note that, for one-GPU or CPU-only case, this module behaves exactly same
    as the built-in PyTorch implementation.
    The mean and standard-deviation are calculated per-dimension over
    the mini-batches and gamma and beta are learnable parameter vectors
    of size C (where C is the input size).
    During training, this layer keeps a running estimate of its computed mean
    and variance. The running sum is kept with a default momentum of 0.1.
    During evaluation, this running mean/variance is used for normalization.
    Because the BatchNorm is done over the `C` dimension, computing statistics
    on `(N, H, W)` slices, it's common terminology to call this Spatial BatchNorm
    Args:
        num_features: num_features from an expected input of
            size batch_size x num_features x height x width
        eps: a value added to the denominator for numerical stability.
            Default: 1e-5
        momentum: the value used for the running_mean and running_var
            computation. Default: 0.1
        affine: a boolean value that when set to ``True``, gives the layer learnable
            affine parameters. Default: ``True``
    Shape:
        - Input: :math:`(N, C, H, W)`
        - Output: :math:`(N, C, H, W)` (same shape as input)
    Examples:
        >>> # With Learnable Parameters
        >>> m = SynchronizedBatchNorm2d(100)
        >>> # Without Learnable Parameters
        >>> m = SynchronizedBatchNorm2d(100, affine=False)
        >>> input = torch.autograd.Variable(torch.randn(20, 100, 35, 45))
        >>> output = m(input)
    """

    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(input.dim()))
        super(SynchronizedBatchNorm2d, self)._check_input_dim(input)

def generate_net(cfg):
    if cfg.MODEL_NAME == 'deeplabv3plus' or cfg.MODEL_NAME == 'deeplabv3+':
        return deeplabv3plus(cfg)
# ----------------------------------------
# Written by Yude Wang
# ----------------------------------------

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
#from net.sync_batchnorm import SynchronizedBatchNorm2d
from torch.nn import init
#from net.backbone import build_backbone
#from net.ASPP import ASPP


import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
#from net.sync_batchnorm import SynchronizedBatchNorm2d

class ASPP(nn.Module):

    def __init__(self, dim_in, dim_out, rate=1, bn_mom=0.1):
        super(ASPP, self).__init__()
        self.branch1 = nn.Sequential(
                nn.Conv2d(dim_in, dim_out, 1, 1, padding=0, dilation=rate,bias=True),
                SynchronizedBatchNorm2d(dim_out, momentum=bn_mom),
                nn.ReLU(inplace=True),
        )
        self.branch2 = nn.Sequential(
                nn.Conv2d(dim_in, dim_out, 3, 1, padding=6*rate, dilation=6*rate,bias=True),
                SynchronizedBatchNorm2d(dim_out, momentum=bn_mom),
                nn.ReLU(inplace=True),
        )
        self.branch3 = nn.Sequential(
                nn.Conv2d(dim_in, dim_out, 3, 1, padding=12*rate, dilation=12*rate,bias=True),
                SynchronizedBatchNorm2d(dim_out, momentum=bn_mom),
                nn.ReLU(inplace=True),
        )
        self.branch4 = nn.Sequential(
                nn.Conv2d(dim_in, dim_out, 3, 1, padding=18*rate, dilation=18*rate,bias=True),
                SynchronizedBatchNorm2d(dim_out, momentum=bn_mom),
                nn.ReLU(inplace=True),
        )
        self.branch5_conv = nn.Conv2d(dim_in, dim_out, 1, 1, 0,bias=True)
        self.branch5_bn = SynchronizedBatchNorm2d(dim_out, momentum=bn_mom)
        self.branch5_relu = nn.ReLU(inplace=True)
        self.conv_cat = nn.Sequential(
                nn.Conv2d(dim_out*5, dim_out, 1, 1, padding=0,bias=True),
                SynchronizedBatchNorm2d(dim_out, momentum=bn_mom),
                nn.ReLU(inplace=True),
        )
#        self.conv_cat = nn.Sequential(
#                nn.Conv2d(dim_out*4, dim_out, 1, 1, padding=0),
#                SynchronizedBatchNorm2d(dim_out),
#                nn.ReLU(inplace=True),
#        )
    def forward(self, x):
        [b,c,row,col] = x.size()
        conv1x1 = self.branch1(x)
        conv3x3_1 = self.branch2(x)
        conv3x3_2 = self.branch3(x)
        conv3x3_3 = self.branch4(x)
        global_feature = torch.mean(x,2,True)
        global_feature = torch.mean(global_feature,3,True)
        global_feature = self.branch5_conv(global_feature)
        global_feature = self.branch5_bn(global_feature)
        global_feature = self.branch5_relu(global_feature)
        global_feature = F.interpolate(global_feature, (row,col), None, 'bilinear', True)

        feature_cat = torch.cat([conv1x1, conv3x3_1, conv3x3_2, conv3x3_3, global_feature], dim=1)
#        feature_cat = torch.cat([conv1x1, conv3x3_1, conv3x3_2, conv3x3_3], dim=1)
        result = self.conv_cat(feature_cat)
        return result

bn_mom = 0.0003
__all__ = ['xception']

model_urls = {
    'xception': '/home/wangyude/.torch/models/xception_pytorch_imagenet.pth'#'http://data.lip6.fr/cadene/pretrainedmodels/xception-b5690688.pth'
}

class SeparableConv2d(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=1,stride=1,padding=0,dilation=1,bias=False,activate_first=True,inplace=True):
        super(SeparableConv2d,self).__init__()
        self.relu0 = nn.ReLU(inplace=inplace)
        self.depthwise = nn.Conv2d(in_channels,in_channels,kernel_size,stride,padding,dilation,groups=in_channels,bias=bias)
        self.bn1 = SynchronizedBatchNorm2d(in_channels, momentum=bn_mom)
        self.relu1 = nn.ReLU(inplace=True)
        self.pointwise = nn.Conv2d(in_channels,out_channels,1,1,0,1,1,bias=bias)
        self.bn2 = SynchronizedBatchNorm2d(out_channels, momentum=bn_mom)
        self.relu2 = nn.ReLU(inplace=True)
        self.activate_first = activate_first
    def forward(self,x):
        if self.activate_first:
            x = self.relu0(x)
        x = self.depthwise(x)
        x = self.bn1(x)
        if not self.activate_first:
            x = self.relu1(x)
        x = self.pointwise(x)
        x = self.bn2(x)
        if not self.activate_first:
            x = self.relu2(x)
        return x


class Block(nn.Module):
    def __init__(self,in_filters,out_filters,strides=1,atrous=None,grow_first=True,activate_first=True,inplace=True):
        super(Block, self).__init__()
        if atrous == None:
            atrous = [1]*3
        elif isinstance(atrous, int):
            atrous_list = [atrous]*3
            atrous = atrous_list
        idx = 0
        self.head_relu = True
        if out_filters != in_filters or strides!=1:
            self.skip = nn.Conv2d(in_filters,out_filters,1,stride=strides, bias=False)
            self.skipbn = SynchronizedBatchNorm2d(out_filters, momentum=bn_mom)
            self.head_relu = False
        else:
            self.skip=None

        self.hook_layer = None
        if grow_first:
            filters = out_filters
        else:
            filters = in_filters
        self.sepconv1 = SeparableConv2d(in_filters,filters,3,stride=1,padding=1*atrous[0],dilation=atrous[0],bias=False,activate_first=activate_first,inplace=self.head_relu)
        self.sepconv2 = SeparableConv2d(filters,out_filters,3,stride=1,padding=1*atrous[1],dilation=atrous[1],bias=False,activate_first=activate_first)
        self.sepconv3 = SeparableConv2d(out_filters,out_filters,3,stride=strides,padding=1*atrous[2],dilation=atrous[2],bias=False,activate_first=activate_first,inplace=inplace)

    def forward(self,inp):

        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp

        x = self.sepconv1(inp)
        x = self.sepconv2(x)
        self.hook_layer = x
        x = self.sepconv3(x)

        x+=skip
        return x


class Xception(nn.Module):
    """
    Xception optimized for the ImageNet dataset, as specified in
    https://arxiv.org/pdf/1610.02357.pdf
    """
    def __init__(self, os):
        """ Constructor
        Args:
            num_classes: number of classes
        """
        super(Xception, self).__init__()

        stride_list = None
        if os == 8:
            stride_list = [2,1,1]
        elif os == 16:
            stride_list = [2,2,1]
        else:
            raise ValueError('xception.py: output stride=%d is not supported.'%os)
        self.conv1 = nn.Conv2d(3, 32, 3, 2, 1, bias=False)
        self.bn1 = SynchronizedBatchNorm2d(32, momentum=bn_mom)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(32,64,3,1,1,bias=False)
        self.bn2 = SynchronizedBatchNorm2d(64, momentum=bn_mom)
        #do relu here

        self.block1=Block(64,128,2)
        self.block2=Block(128,256,stride_list[0],inplace=False)
        self.block3=Block(256,728,stride_list[1])

        rate = 16//os
        self.block4=Block(728,728,1,atrous=rate)
        self.block5=Block(728,728,1,atrous=rate)
        self.block6=Block(728,728,1,atrous=rate)
        self.block7=Block(728,728,1,atrous=rate)

        self.block8=Block(728,728,1,atrous=rate)
        self.block9=Block(728,728,1,atrous=rate)
        self.block10=Block(728,728,1,atrous=rate)
        self.block11=Block(728,728,1,atrous=rate)

        self.block12=Block(728,728,1,atrous=rate)
        self.block13=Block(728,728,1,atrous=rate)
        self.block14=Block(728,728,1,atrous=rate)
        self.block15=Block(728,728,1,atrous=rate)

        self.block16=Block(728,728,1,atrous=[1*rate,1*rate,1*rate])
        self.block17=Block(728,728,1,atrous=[1*rate,1*rate,1*rate])
        self.block18=Block(728,728,1,atrous=[1*rate,1*rate,1*rate])
        self.block19=Block(728,728,1,atrous=[1*rate,1*rate,1*rate])

        self.block20=Block(728,1024,stride_list[2],atrous=rate,grow_first=False)
        #self.block12=Block(728,1024,2,2,start_with_relu=True,grow_first=False)

        self.conv3 = SeparableConv2d(1024,1536,3,1,1*rate,dilation=rate,activate_first=False)
        # self.bn3 = SynchronizedBatchNorm2d(1536, momentum=bn_mom)

        self.conv4 = SeparableConv2d(1536,1536,3,1,1*rate,dilation=rate,activate_first=False)
        # self.bn4 = SynchronizedBatchNorm2d(1536, momentum=bn_mom)

        #do relu here
        self.conv5 = SeparableConv2d(1536,2048,3,1,1*rate,dilation=rate,activate_first=False)
        # self.bn5 = SynchronizedBatchNorm2d(2048, momentum=bn_mom)
        self.layers = []

        #------- init weights --------
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        #-----------------------------

    def forward(self, input):
        self.layers = []
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)
        #self.layers.append(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.block1(x)
        x = self.block2(x)
        self.layers.append(self.block2.hook_layer)
        x = self.block3(x)
        # self.layers.append(self.block3.hook_layer)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.block11(x)
        x = self.block12(x)
        x = self.block13(x)
        x = self.block14(x)
        x = self.block15(x)
        x = self.block16(x)
        x = self.block17(x)
        x = self.block18(x)
        x = self.block19(x)
        x = self.block20(x)
        # self.layers.append(self.block20.hook_layer)

        x = self.conv3(x)
        # x = self.bn3(x)
        # x = self.relu(x)

        x = self.conv4(x)
        # x = self.bn4(x)
        # x = self.relu(x)

        x = self.conv5(x)
        # x = self.bn5(x)
        # x = self.relu(x)
        self.layers.append(x)

        return x

    def get_layers(self):
        return self.layers

def xception(pretrained=True, os=16):
    model = Xception(os=os)
    if pretrained:
        old_dict = torch.load(model_urls['xception'])
        # old_dict = model_zoo.load_url(model_urls['xception'])
        # for name, weights in old_dict.items():
        #     if 'pointwise' in name:
        #         old_dict[name] = weights.unsqueeze(-1).unsqueeze(-1)
        model_dict = model.state_dict()
        old_dict = {k: v for k,v in old_dict.items() if ('itr' not in k and 'tmp' not in k and 'track' not in k)}
        model_dict.update(old_dict)

        model.load_state_dict(model_dict)

    return model

def build_backbone(backbone_name, pretrained=False, os=16):
    if backbone_name == 'res50_atrous':
        net = atrousnet.resnet50_atrous(pretrained=pretrained, os=os)
        return net
    elif backbone_name == 'res101_atrous':
        net = atrousnet.resnet101_atrous(pretrained=pretrained, os=os)
        return net
    elif backbone_name == 'res152_atrous':
        net = atrousnet.resnet152_atrous(pretrained=pretrained, os=os)
        return net
    elif backbone_name == 'xception' or backbone_name == 'Xception':
        net = xception(pretrained=pretrained, os=os)
        return net
    else:
        raise ValueError('backbone.py: The backbone named %s is not supported yet.'%backbone_name)

class deeplabv3plus(nn.Module):
    def __init__(self, cfg):
        super(deeplabv3plus, self).__init__()
        self.backbone = None
        self.backbone_layers = None
        input_channel = 2048
        self.aspp = ASPP(dim_in=input_channel,
                dim_out=cfg.MODEL_ASPP_OUTDIM,
                rate=16//cfg.MODEL_OUTPUT_STRIDE,
                bn_mom = cfg.TRAIN_BN_MOM)
        self.dropout1 = nn.Dropout(0.5)
        self.upsample4 = nn.UpsamplingBilinear2d(scale_factor=4)
        self.upsample_sub = nn.UpsamplingBilinear2d(scale_factor=cfg.MODEL_OUTPUT_STRIDE//4)

        indim = 256
        self.shortcut_conv = nn.Sequential(
                nn.Conv2d(indim, cfg.MODEL_SHORTCUT_DIM, cfg.MODEL_SHORTCUT_KERNEL, 1, padding=cfg.MODEL_SHORTCUT_KERNEL//2,bias=True),
                SynchronizedBatchNorm2d(cfg.MODEL_SHORTCUT_DIM, momentum=cfg.TRAIN_BN_MOM),
                nn.ReLU(inplace=True),
        )
        self.cat_conv = nn.Sequential(
                nn.Conv2d(cfg.MODEL_ASPP_OUTDIM+cfg.MODEL_SHORTCUT_DIM, cfg.MODEL_ASPP_OUTDIM, 3, 1, padding=1,bias=True),
                SynchronizedBatchNorm2d(cfg.MODEL_ASPP_OUTDIM, momentum=cfg.TRAIN_BN_MOM),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Conv2d(cfg.MODEL_ASPP_OUTDIM, cfg.MODEL_ASPP_OUTDIM, 3, 1, padding=1,bias=True),
                SynchronizedBatchNorm2d(cfg.MODEL_ASPP_OUTDIM, momentum=cfg.TRAIN_BN_MOM),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1),
        )
        self.cls_conv = nn.Conv2d(cfg.MODEL_ASPP_OUTDIM, cfg.MODEL_NUM_CLASSES, 1, 1, padding=0)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, SynchronizedBatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        self.backbone = build_backbone(cfg.MODEL_BACKBONE, os=cfg.MODEL_OUTPUT_STRIDE)
        self.backbone_layers = self.backbone.get_layers()

    def forward(self, x):
        x_shape = x.shape[-2:]
        x_bottom = self.backbone(x)
        layers = self.backbone.get_layers()
        feature_aspp = self.aspp(layers[-1])
        feature_aspp = self.dropout1(feature_aspp)
        feature_aspp = self.upsample_sub(feature_aspp)

        feature_shallow = self.shortcut_conv(layers[0])
        feature_shallow = torch.nn.functional.pad(feature_shallow, (1, 0, 1, 0))
        feature_cat = torch.cat([feature_aspp,feature_shallow],1)
        result = self.cat_conv(feature_cat)
        result = self.cls_conv(result)
        result = torch.nn.functional.interpolate(result, size=x_shape)
        return result

import torch
import argparse
import os
import sys
import cv2
import time

class Configuration():
    def __init__(self):
        self.ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname("__file__"),'..','..'))
        self.EXP_NAME = 'deeplabv3+voc'

        self.DATA_NAME = 'VOC2012'
        self.DATA_AUG = True
        self.DATA_WORKERS = 8
        self.DATA_RESCALE = 512
        self.DATA_RANDOMCROP = 512
        self.DATA_RANDOMROTATION = 0
        self.DATA_RANDOMSCALE = 2
        self.DATA_RANDOM_H = 10
        self.DATA_RANDOM_S = 10
        self.DATA_RANDOM_V = 10
        self.DATA_RANDOMFLIP = 0.5

        self.MODEL_NAME = 'deeplabv3plus'
        #self.MODEL_BACKBONE = 'res101_atrous'
        self.MODEL_BACKBONE = 'xception'
        self.MODEL_OUTPUT_STRIDE = 16
        self.MODEL_ASPP_OUTDIM = 256
        self.MODEL_SHORTCUT_DIM = 48
        self.MODEL_SHORTCUT_KERNEL = 1
        self.MODEL_NUM_CLASSES = 21
        self.MODEL_SAVE_DIR = os.path.join(self.ROOT_DIR,'model',self.EXP_NAME)

        self.TRAIN_LR = 0.007
        self.TRAIN_LR_GAMMA = 0.1
        self.TRAIN_MOMENTUM = 0.9
        self.TRAIN_WEIGHT_DECAY = 0.00004
        self.TRAIN_BN_MOM = 0.0003
        self.TRAIN_POWER = 0.9
        self.TRAIN_GPUS = 4
        self.TRAIN_BATCHES = 16
        self.TRAIN_SHUFFLE = True
        self.TRAIN_MINEPOCH = 0
        self.TRAIN_EPOCHS = 46
        self.TRAIN_LOSS_LAMBDA = 0
        self.TRAIN_TBLOG = True
        self.TRAIN_CKPT = None#os.path.join(self.ROOT_DIR,'model/deeplabv3+voc/deeplabv3plus_xception_VOC2012_itr0.pth')

        self.LOG_DIR = os.path.join(self.ROOT_DIR,'log',self.EXP_NAME)

        self.TEST_MULTISCALE = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
        self.TEST_FLIP = True
        self.TEST_CKPT = os.path.join(self.ROOT_DIR,'model/deeplabv3+voc/deeplabv3plus_res101_atrous_VOC2012_epoch46_all.pth')
        self.TEST_GPUS = 4
        self.TEST_BATCHES = 16

        self.__check()
        self.__add_path(os.path.join(self.ROOT_DIR, 'lib'))

    def __check(self):
        if not torch.cuda.is_available():
            raise ValueError('config.py: cuda is not avalable')
        if self.TRAIN_GPUS == 0:
            raise ValueError('config.py: the number of GPU is 0')
        #if self.TRAIN_GPUS != torch.cuda.device_count():
        #    raise ValueError('config.py: GPU number is not matched')
        if not os.path.isdir(self.LOG_DIR):
            os.makedirs(self.LOG_DIR)
        if not os.path.isdir(self.MODEL_SAVE_DIR):
            os.makedirs(self.MODEL_SAVE_DIR)

    def __add_path(self, path):
        if path not in sys.path:
            sys.path.insert(0, path)



cfg = Configuration()

def generate_net(cfg):
    if cfg.MODEL_NAME == 'deeplabv3plus' or cfg.MODEL_NAME == 'deeplabv3+':
        return deeplabv3plus(cfg)
    # if cfg.MODEL_NAME == 'supernet' or cfg.MODEL_NAME == 'SuperNet':
    #     return SuperNet(cfg)
    # if cfg.MODEL_NAME == 'eanet' or cfg.MODEL_NAME == 'EANet':
    #     return EANet(cfg)
    # if cfg.MODEL_NAME == 'danet' or cfg.MODEL_NAME == 'DANet':
    #     return DANet(cfg)
    # if cfg.MODEL_NAME == 'deeplabv3plushd' or cfg.MODEL_NAME == 'deeplabv3+hd':
    #     return deeplabv3plushd(cfg)
    # if cfg.MODEL_NAME == 'danethd' or cfg.MODEL_NAME == 'DANethd':
    #     return DANethd(cfg)
    else:
        raise ValueError('generateNet.py: network %s is not support yet'%cfg.MODEL_NAME)

def deeplab(num_classes):
    cfg.MODEL_NUM_CLASSES = num_classes
    return generate_net(cfg)