import math
import torch
from torch import nn
from models.layers import Conv, Hourglass, Hourglass_easy


class Merge(nn.Module):
    """Change the channel dimension of the input tensor"""

    def __init__(self, x_dim, y_dim):
        super(Merge, self).__init__()
        self.conv = Conv(x_dim, y_dim, 1, relu=False, bn=False)

    def forward(self, x):
        return self.conv(x)


class Features(nn.Module):
    """Input: feature maps produced by hourglass block
       Return: 5 different scales of feature maps, 128*128, 64*64, 32*32, 16*16, 8*8"""

    def __init__(self, inp_dim, increase=128, bn=False):
        super(Features, self).__init__()
        # regress 4 heat maps per stack
        self.before_regress = nn.ModuleList(
            [nn.Sequential(Conv(inp_dim + i * increase, inp_dim + i * increase, 3, bn=bn),
                           Conv(inp_dim + i * increase, inp_dim + i * increase, 3, bn=bn)) for i in range(5)])

    def forward(self, fms):
        assert len(fms) == 5, "hourglass output {} tensors,but 5 scale heatmaps are supervised".format(len(fms))
        return [self.before_regress[i](fms[i]) for i in range(5)]


class PoseNet(nn.Module):
    def __init__(self, nstack, inp_dim, oup_dim, bn=False, increase=128, init_weights=True, **kwargs):
        """
        Pack or initialize the trainable parameters of the network
        :param nstack: number of stack
        :param inp_dim: input tensor channels fed into the hourglass block
        :param oup_dim: channels of regressed feature maps
        :param bn:
        :param increase: increased channels once down-sampling
        :param kwargs:
        """
        super(PoseNet, self).__init__()
        self.pre = nn.Sequential(
            Conv(3, 64, 7, 2, bn=bn),
            Conv(64, 128, bn=bn),
            nn.MaxPool2d(2, 2),
            Conv(128, 128, bn=bn),
            Conv(128, inp_dim, bn=bn)
        )
        self.hourglass = nn.ModuleList([Hourglass(4, inp_dim, increase, bn=bn) for _ in range(nstack)])
        self.features = nn.ModuleList([Features(inp_dim, increase=increase, bn=bn) for _ in range(nstack)])
        # predict 5 different scales of heatmpas per stack, keep in mind to pack the list using ModuleList.
        # Notice: nn.ModuleList can only identify Module subclass! Thus, we must pack the inner layers in ModuleList.
        self.outs = nn.ModuleList(
            [nn.ModuleList([Conv(inp_dim + j * increase, oup_dim, 1, relu=False, bn=False) for j in range(5)]) for i in
             range(nstack)])
        self.merge_features = nn.ModuleList(
            [nn.ModuleList([Merge(inp_dim + j * increase, inp_dim + j * increase) for j in range(5)]) for i in
             range(nstack - 1)])
        self.merge_preds = nn.ModuleList(
            [nn.ModuleList([Merge(oup_dim, inp_dim + j * increase) for j in range(5)]) for i in range(nstack - 1)])
        self.nstack = nstack
        if init_weights:
            self._initialize_weights()

    def forward(self, imgs):
        # Input Tensor: a batch of images within [0,1], shape=(N, H, W, C). Pre-processing was done in data generator
        x = imgs.permute(0, 3, 1, 2)  # Permute the dimensions of images to (N, C, H, W)
        x = self.pre(x)
        pred = []
        # loop over stack
        for i in range(self.nstack):
            preds_instack = []
            # return 5 scales of feature maps
            hourglass_feature = self.hourglass[i](x)

            if i == 0:  # cache for 4 smaller feature maps produced by hourglass block
                features__cache = [torch.zeros_like(hourglass_feature[k + 1]) for k in range(4)]
            else:  # res connection cross stacks
                for k in range(4):
                    #  python里面的+=, ，*=也是in-place operation,需要注意
                    hourglass_feature[k + 1] = hourglass_feature[k + 1] + features__cache[k]
            # feature maps before heatmap regression
            features_instack = self.features[i](hourglass_feature)

            for j in range(5):  # handle 5 scales of heatmaps
                preds_instack.append(self.outs[i][j](features_instack[j]))
                if i != self.nstack - 1:
                    if j == 0:
                        x = x + self.merge_preds[i][j](preds_instack[j]) + self.merge_features[i][j](
                            features_instack[j])
                    else:
                        # reset the res caches
                        features__cache[j - 1] = self.merge_preds[i][j](preds_instack[j]) + self.merge_features[i][j](
                            features_instack[j])
            pred.append(preds_instack)
        # returned list shape: [nstack * [128*128, 64*64, 32*32, 16*16, 8*8]]
        return pred

    def _initialize_weights(self):
        for m in self.modules():
            # 卷积的初始化方法
            if isinstance(m, nn.Conv2d):
                # TODO: 使用正态分布进行初始化（0, 0.01) 网络权重看看
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # He kaiming 初始化, 方差为2/n. math.sqrt(2. / n) 或者直接使用现成的nn.init中的函数。在这里会梯度爆炸
                m.weight.data.normal_(0, 0.01)    # # math.sqrt(2. / n)
                # torch.nn.init.uniform_(tensorx)
                # bias都初始化为0
                if m.bias is not None:  # 当有BN层时，卷积层Con不加bias！
                    m.bias.data.zero_()
            # batchnorm使用全1初始化 bias全0
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

            elif isinstance(m, nn.Linear):
                torch.nn.init.normal_(m.weight.data, 0, 0.01)  # m.weight.data.normal_(0, 0.01) m.bias.data.zero_()


class PoseNet_easy(nn.Module):
    def __init__(self, nstack, inp_dim, oup_dim, bn=False, increase=128, init_weights=True, **kwargs):
        super(PoseNet_easy, self).__init__()
        self.pre = nn.Sequential(
            Conv(3, 64, 7, 2, bn=bn),
            Conv(64, 128, bn=bn),
            nn.MaxPool2d(2, 2),
            Conv(128, 128, bn=bn),
            Conv(128, inp_dim, bn=bn)
        )
        self.features = nn.ModuleList([
            nn.Sequential(
                Hourglass_easy(4, inp_dim, increase=increase, bn=bn),
                Conv(inp_dim, inp_dim, 3, bn=False),
                Conv(inp_dim, inp_dim, 3, bn=False)
            ) for i in range(nstack)])  # 构造了nstack个这样的Hourglass+2*Conve模块

        self.outs = nn.ModuleList([Conv(inp_dim, oup_dim, 1, relu=False, bn=False) for i in range(nstack)])
        self.merge_features = nn.ModuleList([Merge(inp_dim, inp_dim) for i in range(nstack - 1)])
        self.merge_preds = nn.ModuleList([Merge(oup_dim, inp_dim) for i in range(nstack - 1)])

        self.nstack = nstack

    def forward(self, imgs):
        x = imgs.permute(0, 3, 1, 2)
        x = self.pre(x)
        preds = []
        for i in range(self.nstack):
            feature = self.features[i](x)
            preds.append(self.outs[i](feature))
            if i != self.nstack - 1:
                x = x + self.merge_preds[i](preds[-1]) + self.merge_features[i](feature)
        return torch.stack(preds, 1)

    def _initialize_weights(self):
        for m in self.modules():
            # 卷积的初始化方法
            if isinstance(m, nn.Conv2d):
                # TODO: 使用正态分布进行初始化（0, 0.01) 网络权重看看
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # He kaiming 初始化, 方差为2/n. math.sqrt(2. / n) 或者直接使用现成的nn.init中的函数。在这里会梯度爆炸
                m.weight.data.normal_(0, 0.01)    # # math.sqrt(2. / n)
                # torch.nn.init.uniform_(tensorx)
                # bias都初始化为0
                if m.bias is not None:  # 当有BN层时，卷积层Con不加bias！
                    m.bias.data.zero_()
            # batchnorm使用全1初始化 bias全0
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

            elif isinstance(m, nn.Linear):
                torch.nn.init.normal_(m.weight.data, 0, 0.01)  # m.weight.data.normal_(0, 0.01) m.bias.data.zero_()



if __name__ == '__main__':
    from time import time

    # pose = PoseNet(4, 256, 54)  # .cuda()
    # for param in pose.parameters():
    #     if param.requires_grad:
    #         print('param autograd')
    #         break
    #
    # t0 = time()
    # input = torch.rand(1, 128, 128, 3)  # .cuda()
    # print(pose)
    # output = pose(input)  # type: torch.Tensor
    #
    # output[0][0].sum().backward()
    # t1 = time()
    # print('********** Consuming Time is: {} second  **********'.format(t1 - t0))

    # #
    # import torch.onnx
    #
    pose = PoseNet(2, 256, 34)
    dummy_input = torch.randn(1, 512, 512, 3)
    torch.onnx.export(pose, dummy_input, "posenet.onnx")  # netron --host=localhost
    torch.onnx.export(pose, dummy_input, "posenet2.onnx")  # netron --host=localhost

