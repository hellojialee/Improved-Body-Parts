"""Still in development."""
from torch import nn
from torch.autograd import Function
import sys
import time
import numpy as np
import cv2
import torch
import torch.nn.functional as F


class Residual(nn.Module):
    """Residual Block modified by us"""

    def __init__(self, ins, outs, bn=True, relu=True):
        super(Residual, self).__init__()
        self.relu_flag = relu
        self.convBlock = nn.Sequential(
            nn.Conv2d(ins, outs//2, 1, bias=False),
            nn.BatchNorm2d(outs//2),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Conv2d(outs // 2, outs // 2, 3, 1, 1, bias=False),
            nn.BatchNorm2d(outs // 2),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Conv2d(outs // 2, outs, 1, bias=False),
            nn.BatchNorm2d(outs),
        )
        if ins != outs:
            self.skipConv = nn.Sequential(
                nn.Conv2d(ins, outs, 1, bias=False),
                nn.BatchNorm2d(outs)
            )
        self.relu = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        self.ins = ins
        self.outs = outs

    def forward(self, x):
        residual = x
        x = self.convBlock(x)
        if self.ins != self.outs:
            residual = self.skipConv(residual)
        x += residual  # Bn layer is in the middle, so we can do in-plcae += here

        if self.relu_flag:
            x = self.relu(x)
            return x
        else:
            return x


class BasicResidual(nn.Module):
    """
    Basic block used in ResNet, CornerNet, CenterNet, etc.
    Used as the basic block to replace 3*3 convolution, increasing the shortcuts in network
    """
    def __init__(self, inp_dim, out_dim, stride=1, bn=True, relu=True):
        super(BasicResidual, self).__init__()

        self.relu_flag = relu
        self.conv1 = nn.Conv2d(inp_dim, out_dim, (3, 3), padding=(1, 1), stride=(stride, stride), bias=False)
        self.bn1 = nn.BatchNorm2d(out_dim)
        self.relu1 = nn.LeakyReLU(negative_slope=0.01, inplace=True)

        self.conv2 = nn.Conv2d(out_dim, out_dim, (3, 3), padding=(1, 1), bias=False)
        self.bn2 = nn.BatchNorm2d(out_dim)

        self.skip = nn.Sequential(
            nn.Conv2d(inp_dim, out_dim, (1, 1), stride=(stride, stride), bias=False),
            nn.BatchNorm2d(out_dim)
        ) if stride != 1 or inp_dim != out_dim else nn.Sequential()
        self.relu = nn.LeakyReLU(negative_slope=0.01, inplace=True)

    def forward(self, x):
        conv1 = self.conv1(x)
        bn1 = self.bn1(conv1)
        relu1 = self.relu1(bn1)

        conv2 = self.conv2(relu1)
        bn2 = self.bn2(conv2)

        skip = self.skip(x)

        if self.relu_flag:
            out = self.relu(bn2 + skip)
        else:
            out = bn2 + skip
        return out


class Conv(nn.Module):
    # conv block used in hourglass
    def __init__(self, inp_dim, out_dim, kernel_size=3, stride=1, bn=True, relu=True, dropout=False, dialated=1):
        super(Conv, self).__init__()
        self.inp_dim = inp_dim
        self.relu = None
        self.bn = None
        self.dropout = dropout
        if relu:
            self.relu = nn.LeakyReLU(negative_slope=0.01, inplace=True)  # 换成 Leak Relu减缓神经元死亡现象
        if bn:
            self.conv = nn.Conv2d(inp_dim, out_dim, kernel_size, stride, padding=(kernel_size - 1) // 2, bias=False, dilation=1)
            # Different form TF, momentum default in Pytorch is 0.1, which means the decay rate of old running value
            self.bn = nn.BatchNorm2d(out_dim)
        else:
            self.conv = nn.Conv2d(inp_dim, out_dim, kernel_size, stride, padding=(kernel_size - 1) // 2, bias=True, dilation=1)

    def forward(self, x):
        # examine the input channel equals the conve kernel channel
        assert x.size()[1] == self.inp_dim, "input channel {} dese not fit kernel channel {}".format(x.size()[1],
                                                                                                     self.inp_dim)
        if self.dropout:  # comment these two lines if we do not want to use Dropout layers
            # p: probability of an element to be zeroed
            x = F.dropout(x, p=0.2, training=self.training, inplace=False)  # 直接注释掉这一行，如果我们不想使用Dropout

        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class DilatedConv(nn.Module):
    """
    Dilated convolutional layer of stride=1 only!
    """
    def __init__(self, inp_dim, out_dim, kernel_size=3, stride=1, bn=True, relu=True, dropout=False, dialation=3):
        super(DilatedConv, self).__init__()
        self.inp_dim = inp_dim
        self.relu = None
        self.bn = None
        self.dropout = dropout
        if relu:
            self.relu = nn.LeakyReLU(negative_slope=0.01, inplace=True)  # 换成 Leak Relu减缓神经元死亡现象
        if bn:
            self.conv = nn.Conv2d(inp_dim, out_dim, kernel_size, stride, padding=dialation, bias=False, dilation=dialation)
            # Different form TF, momentum default in Pytorch is 0.1, which means the decay rate of old running value
            self.bn = nn.BatchNorm2d(out_dim)
        else:
            self.conv = nn.Conv2d(inp_dim, out_dim, kernel_size, stride, padding=dialation, bias=True, dilation=dialation)

    def forward(self, x):
        # examine the input channel equals the conve kernel channel
        assert x.size()[1] == self.inp_dim, "input channel {} dese not fit kernel channel {}".format(x.size()[1],
                                                                                                     self.inp_dim)
        if self.dropout:  # comment these two lines if we do not want to use Dropout layers
            # p: probability of an element to be zeroed
            x = F.dropout(x, p=0.2, training=self.training, inplace=False)  # 直接注释掉这一行，如果我们不想使用Dropout

        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class Backbone(nn.Module):
    """
    Input Tensor: a batch of images with shape (N, C, H, W)
    """
    def __init__(self, nFeat=256, inplanes=3, resBlock=Residual, dilatedBlock=DilatedConv):
        super(Backbone, self).__init__()
        self.nFeat = nFeat
        self.resBlock = resBlock
        self.inplanes = inplanes
        self.conv1 = nn.Conv2d(self.inplanes, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        self.res1 = self.resBlock(64, 128)
        self.pool = nn.MaxPool2d(2, 2)
        self.res2 = self.resBlock(128, 128)
        self.dilation = nn.Sequential(
            dilatedBlock(128, 128, dialation=3),
            dilatedBlock(128, 128, dialation=3),
            dilatedBlock(128, 128, dialation=4),
            dilatedBlock(128, 128, dialation=4),
            dilatedBlock(128, 128, dialation=5),
            dilatedBlock(128, 128, dialation=5),
        )

    def forward(self, x):
        # head
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.res1(x)
        x = self.pool(x)
        x = self.res2(x)
        x1 = self.dilation(x)
        concat_merge = torch.cat([x, x1], dim=1)  # (N, C1+C2, H, W)

        return concat_merge


class Hourglass(nn.Module):
    """Instantiate an n order Hourglass Network block using recursive trick."""
    def __init__(self, depth, nFeat, increase=128, bn=False, resBlock=Residual, convBlock=Conv):
        super(Hourglass, self).__init__()
        self.depth = depth  # oder number
        self.nFeat = nFeat  # input and output channels
        self.increase = increase  # increased channels while the depth grows
        self.bn = bn
        self.resBlock = resBlock
        self.convBlock = convBlock
        # will execute when instantiate the Hourglass object, prepare network's parameters
        self.hg = self._make_hour_glass()
        self.downsample = nn.MaxPool2d(2, 2)  # no learning parameters, can be used any times repeatedly
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')  # no learning parameters  # FIXME: 改成反卷积？

    def _make_single_residual(self, depth_id):
        # the innermost conve layer, return as a layer item
        return self.resBlock(self.nFeat + self.increase * (depth_id + 1), self.nFeat + self.increase * (depth_id + 1),
                             bn=self.bn)                            # ###########  Index: 4

    def _make_lower_residual(self, depth_id):
        # return as a list
        pack_layers = [self.resBlock(self.nFeat + self.increase * depth_id, self.nFeat + self.increase * depth_id,
                                     bn=self.bn),                                     # ######### Index: 0
                       self.resBlock(self.nFeat + self.increase * depth_id, self.nFeat + self.increase * (depth_id + 1),
                                                                                                  # ######### Index: 1
                                     bn=self.bn),
                       self.resBlock(self.nFeat + self.increase * (depth_id + 1), self.nFeat + self.increase * depth_id,
                                                                                                   # ######### Index: 2
                                     bn=self.bn),
                       self.convBlock(self.nFeat + self.increase * depth_id, self.nFeat + self.increase * depth_id,
                                     # ######### Index: 3
                                     bn=self.bn),  # 添加一个Conv精细化上采样的特征图?
                       ]
        return pack_layers

    def _make_hour_glass(self):
        """
        pack conve layers modules of hourglass block
        :return: conve layers packed in n hourglass blocks
        """
        hg = []
        for i in range(self.depth):
            #  skip path; up_residual_block; down_residual_block_path,
            # 0 ~ n-2 (except the outermost n-1 order) need 3 residual blocks
            res = self._make_lower_residual(i)  # type:list
            if i == (self.depth - 1):  # the deepest path (i.e. the longest path) need 4 residual blocks
                res.append(self._make_single_residual(i))  # list append an element
            hg.append(nn.ModuleList(res))  # pack conve layers of  every oder of hourglass block
        return nn.ModuleList(hg)

    def _hour_glass_forward(self, depth_id, x, up_fms):
        """
        built an hourglass block whose order is depth_id
        :param depth_id: oder number of hourglass block
        :param x: input tensor
        :return: output tensor through an hourglass block
        """
        up1 = self.hg[depth_id][0](x)
        low1 = self.downsample(x)
        low1 = self.hg[depth_id][1](low1)
        if depth_id == (self.depth - 1):  # except for the highest-order hourglass block
            low2 = self.hg[depth_id][4](low1)
        else:
            # call the lower-order hourglass block recursively
            low2 = self._hour_glass_forward(depth_id + 1, low1, up_fms)
        low3 = self.hg[depth_id][2](low2)
        up_fms.append(low2)
        # ######################## # if we don't consider 8*8 scale
        # if depth_id < self.depth - 1:
        #     self.up_fms.append(low2)
        up2 = self.upsample(low3)
        deconv1 = self.hg[depth_id][3](up2)
        # deconv2 = self.hg[depth_id][4](deconv1)
        # up1 += deconv2
        # out = self.hg[depth_id][5](up1)  # relu after residual add
        return up1 + deconv1

    def forward(self, x):
        """
        :param: x a input tensor warpped wrapped as a list
        :return: 5 different scales of feature maps, 128*128, 64*64, 32*32, 16*16, 8*8
        """
        up_fms = []  # collect feature maps produced by low2 at every scale
        feature_map = self._hour_glass_forward(0, x, up_fms)
        return [feature_map] + up_fms[::-1]


class SELayer(nn.Module):
    def __init__(self, inp_dim, reduction=16):
        """
        Squeeze and Excitation
        :param inp_dim: the channel of input tensor
        :param reduction: channel compression ratio
        :return output the tensor with the same shape of input
        """
        assert inp_dim > reduction, f"Make sure your input channel bigger than reduction which equals to {reduction}"
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(inp_dim, inp_dim // reduction),
                nn.LeakyReLU(inplace=True),  # Relu
                nn.Linear(inp_dim // reduction, inp_dim),
                nn.Sigmoid())

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

    # def forward(self, x):  # 去掉Selayer
    #     return x


if __name__ == '__main__':

    se = SELayer(256)
    print(se)
    dummy_input = torch.randn(8, 256, 128, 128)
    out = se(dummy_input)
    print(out.shape)
    out.sum().backward()






