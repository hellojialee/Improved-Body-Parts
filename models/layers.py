from torch import nn
from torch.autograd import Function
import sys
import time
import numpy as np
import cv2
import torch
import torch.nn.functional as F


class Residual(nn.Module):
    """Residual Block for original Hourglass Network"""

    def __init__(self, ins, outs):
        super(Residual, self).__init__()
        self.convBlock = nn.Sequential(
            nn.BatchNorm2d(ins),
            nn.ReLU(inplace=True),
            nn.Conv2d(ins, outs//2,1),
            nn.BatchNorm2d(outs//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(outs // 2, outs // 2, 3, 1, 1),
            nn.BatchNorm2d(outs // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(outs // 2, outs, 1)
        )
        if ins != outs:
            self.skipConv = nn.Conv2d(ins,outs,1)
        self.ins = ins
        self.outs = outs

    def forward(self,x):
        residual = x
        x = self.convBlock(x)
        if self.ins != self.outs:
            residual = self.skipConv(residual)
        x += residual
        return x


class Conv(nn.Module):
    # conv block used in hourglass
    def __init__(self, inp_dim, out_dim, kernel_size=3, stride=1, bn=False, relu=True):
        super(Conv, self).__init__()
        self.inp_dim = inp_dim
        self.conv = nn.Conv2d(inp_dim, out_dim, kernel_size, stride, padding=(kernel_size - 1) // 2, bias=True)
        self.relu = None
        self.bn = None
        if relu:
            self.relu = nn.ReLU(inplace=True)
        if bn:
            self.bn = nn.BatchNorm2d(out_dim)

    def forward(self, x):
        # examine the input channel equals the conve kernel channel
        assert x.size()[1] == self.inp_dim, "input channel {} dese not fit kernel channel {}".format(x.size()[1],
                                                                                                     self.inp_dim)
        x = self.conv(x)
        if self.relu is not None:
            x = self.relu(x)
        if self.bn is not None:
            x = self.bn(x)
        return x


class Hourglass(nn.Module):
    """Instantiate an n order Hourglass Network block using recursive trick."""
    def __init__(self, depth, nFeat, increase=128, bn=False, resBlock=Conv):
        super(Hourglass, self).__init__()
        self.depth = depth  # oder number
        self.nFeat = nFeat  # input and output channels
        self.increase = increase  # increased channels while the depth grows
        self.bn = bn
        self.resBlock = resBlock
        # will execute when instantiate the Hourglass object, prepare network's parameters
        self.hg = self._make_hour_glass()
        self.downsample = nn.MaxPool2d(2, 2)  # no learning parameters, can be used any times repeatedly
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')  # no learning parameters

    def _make_single_residual(self, depth_id):
        # the innermost conve layer, return as an element
        return self.resBlock(self.nFeat + self.increase * (depth_id + 1), self.nFeat + self.increase * (depth_id + 1),
                             bn=self.bn)

    def _make_lower_residual(self, depth_id):
        # return as a list
        return [self.resBlock(self.nFeat + self.increase * depth_id, self.nFeat + self.increase * depth_id, bn=self.bn),
                self.resBlock(self.nFeat + self.increase * depth_id, self.nFeat + self.increase * (depth_id + 1),
                              bn=self.bn),
                self.resBlock(self.nFeat + self.increase * (depth_id + 1), self.nFeat + self.increase * depth_id,
                              bn=self.bn)]

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
            low2 = self.hg[depth_id][3](low1)
        else:
            # call the lower-order hourglass block recursively
            low2 = self._hour_glass_forward(depth_id + 1, low1, up_fms)
        low3 = self.hg[depth_id][2](low2)
        up_fms.append(low2)
        # ######################## # if we don't consider 8*8 scale
        # if depth_id < self.depth - 1:
        #     self.up_fms.append(low2)
        up2 = self.upsample(low3)
        return up1 + up2

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
        assert inp_dim > reduction, "Make sure your input channel bigger than reduction which equals to {}".format(reduction)
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(inp_dim, inp_dim // reduction),
                nn.ReLU(inplace=True),
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






