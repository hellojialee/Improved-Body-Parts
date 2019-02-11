#!/usr/bin/env python
# coding:utf-8
import numpy as np
from math import sqrt, isnan


class Heatmapper:
    # 输入图片的尺寸处理
    # 训练时需要相同大小的图片才能组成一个batch，在openpose中有两种做法：
    # 一是直接resize到指定大小的尺寸;
    # 二是源码提供了一种稍微有特色的做法:　# todo: how to resize the image to fixed size to fit the batch
    # 先指定长和宽x，y。然后将图片的长 / 宽和x / y比较，看是否大于1
    # 然后，选择长一些的边（长 > x?, 宽 > y?)，固定长宽比缩放到给定尺寸
    # 再然后，为另一条边加padding，两边加相同的padding
    # 最后，resize到指定大小。

    def __init__(self, config):

        self.config = config
        self.sigma = config.transform_params.sigma
        self.paf_sigma = config.transform_params.paf_sigma
        self.double_sigma2 = 2 * self.sigma * self.sigma
        self.thre = config.transform_params.paf_thre

        # cached common parameters which same for all iterations and all pictures

        stride = self.config.stride
        width = self.config.width // stride
        height = self.config.height // stride

        # this is coordinates of centers of bigger grid
        self.grid_x = np.arange(width) * stride + stride / 2 - 0.5  # Fixme:　stride / 2 -0.5是为了在计算响应图时，使用grid的中心
        self.grid_y = np.arange(height) * stride + stride / 2 - 0.5

        self.Y, self.X = np.mgrid[0:self.config.height:stride, 0:self.config.width:stride]
        # slice操作，比如L[：10：2]前10个数，每隔两个取一个

        # # TODO: check it again
        # # basically we should use center of grid, but in this place classic implementation uses left-top point.
        # self.X = self.X + stride / 2 - 0.5
        # self.Y = self.Y + stride / 2 - 0.5
        #  经过考虑，对于PAF生成不需要这个，因为后面会在limb两端分别延长一个pixle，使得可以包括整个limb的范围

    def create_heatmaps(self, joints, mask):  # 看来图像根据每个main person都被处理成了固定的大小尺寸，因此heat map也是固定大小了
        # print(joints.shape)  # 例如(3, 18, 3)，把每个main person作为图片的中心，但是依然可能会包括其他不同的人在这个裁剪后的图像中
        heatmaps = np.zeros(self.config.parts_shape, dtype=np.float32)  # config.parts_shape: 46, 46, 57
        # 此处的heat map一共有57个channel，包含了heat map以及paf以及背景channel。并且对heat map初始化为0很重要，因为这样使得没有标注的区域是没有值的！
        self.put_joints(heatmaps, joints)  # fixme: 和put_limbs学习，也加slice减少整个heatmap的计算量
        sl = slice(self.config.heat_start, self.config.heat_start + self.config.heat_layers)
        # python切片函数　class slice(start, stop[, step])

        # Generate foreground of keypoint heat map todo: add paf background
        heatmaps[:, :, self.config.bkg_start] = 1. - np.amax(heatmaps[:, :, sl], axis=2)
        # # 某个位置的背景heatmap值定义为这个坐标位置处　最大的某个类型节点高斯响应的补 1. - np.amax(heatmaps[:, :, sl], axis=2)
        # 如果加入的是前景而不是背景，则响应是　np.amax(heatmaps[:, :, sl], axis=2)

        self.put_limbs(heatmaps, joints)

        # todo: add paf background and merge with the keypoint background
        # sl = slice(self.config.heat_start, self.config.heat_start + self.config.heat_layers)
        # ##  如果同时添加了keypoint和limb的背景类channel，则换成如下代码
        # # # sl = slice(self.config.paf_start, self.config.heat_start + self.config.heat_layers)
        # heatmaps[:, :, self.config.bkg_start] = 1. - np.amax(heatmaps[:, :, sl], axis=2)
        # show一下看看生成的背景是否正常

        heatmaps *= mask[:, :, np.newaxis]  # 重要！不要忘了将生成的groundtruth 乘以mask，以此掩盖掉没有标注的crowd以及只有很少keypoint的人

        # see: https://github.com/ZheC/Realtime_Multi-Person_Pose_Estimation/issues/124
        # Mask never touch pictures.  Mask不会叠加到image数据上
        # Mask has exactly same dimensions as ground truth and network output. ie 46 x 46 x num_layers.
        # ------------------------------------------------------------- #
        # Mask applied to:
        # * ground truth heatmap and pafs (multiplied by mask)
        # * network output (multiplied by mask)
        # ------------------------------------------------------------- #
        # If in same point of answer mask is zero this means "ignore answers in this point while training network"
        # because loss will be zero in this point.

        return heatmaps

    def put_gaussian_maps(self, heatmaps, layer, joints):  # FIXME：只计算一定区域内而不是全图像的值来加速GT的生成，参考associate embedding
        #  change the gaussian map to laplace map to get a shapper peak of keypoint ?? the result is not good
        # actually exp(a+b) = exp(a)*exp(b), lets use it calculating 2d exponent, it could just be calculated by

        for i in range(joints.shape[0]):  # 外层循环是对每一个joint都在对应类型channel的feature map上产生一个高斯分布

            # --------------------------------------------------------------------------------------------------#
            # 这里是个技巧，grid_x其实取值范围是0~368，起点是3.5，终点值是363.5，间隔为8，这样就是在原始368个位置上计算高斯值，
            # 采样了46个点，从而最大程度保留了原始分辨率尺寸上的响应值。而不是生成原始分辨率大小的ground truth然后缩小8倍　　notice!!

            # 如果使用高斯分布：
            exp_x = np.exp(-(self.grid_x - joints[i, 0]) ** 2 / self.double_sigma2)
            exp_y = np.exp(-(self.grid_y - joints[i, 1]) ** 2 / self.double_sigma2)

            exp = np.outer(exp_y, exp_x)  # np.outer的计算，两个长度为M,N的向量的外积结果是M*N的矩阵
            exp[exp <= 0.01] = 0  # 距离keypoint太远的点令其响应为0
            # --------------------------------------------------------------------------------------------------#

            # # heatmap　如果使用拉普拉斯分布：dis = exp-(math.sqrt((xx - x) * (xx - x) + (yy - y) * (yy - y)) / 2.0 / sigma)
            # dist = np.sqrt((self.X - joints[i, 0])**2 + (self.Y - joints[i, 1])**2) / 2.0 / self.sigma
            # np.where(dist > 4.6052, 1e8, dist) # 距离中心太远的不赋值
            # exp = np.exp(-dist)

            # note this is correct way of combination - min(sum(...),1.0) as was in C++ code is incorrect
            # https://github.com/ZheC/Realtime_Multi-Person_Pose_Estimation/issues/118
            heatmaps[:, :, self.config.heat_start + layer] = np.maximum(heatmaps[:, :, self.config.heat_start + layer],
                                                                        exp)
            # 这一句代码是解决如何处理一个位置可能有不同人的关节点的高斯响应图的生成“覆盖”的问题，不取这两个点的高斯分布的平均，而是取最大值
            # Notice!
            # ------------------------------------------------------------------------------------------------ #
            # 每一条曲线的峰值都表示这个位置存在关键点的可能性最高，如论文公式(7)图所示，可能有两个关键点距离比较近，这两条高斯曲线
            # 如果取平均值的话，很明显就从两个峰值变成一个峰值了，那最后预测出的结果可能就只有一个关键点了。所以这里取的是最大值。
            # ------------------------------------------------------------------------------------------------ #

    def put_joints(self, heatmaps, joints):

        for i in range(self.config.num_parts):  # len(config.num_parts) = 18, 不包括背景keypoint
            visible = joints[:, i, 2] < 2  # only annotated (visible) keypoints are considered
            self.put_gaussian_maps(heatmaps, i, joints[visible, i, 0:2])  # 逐个channel地进行ground truth的生成

    def put_vector_maps(self, heatmaps, layer, joint_from, joint_to):
        """
        生成一个channel上的PAF groundtruth
        """
        # 实际上作者曹哲说的方式是在两个关键点画椭圆，调参数，加单位矢量。

        count = np.zeros(heatmaps.shape[:-1], dtype=np.int)  # count用来记录某一个位置点上有多少非零的paf，以便后面做平均

        for i in range(joint_from.shape[0]):
            (x1, y1) = joint_from[i]
            (x2, y2) = joint_to[i]

            dx = x2 - x1
            dy = y2 - y1
            dnorm = dx * dx + dy * dy

            if dnorm == 0:  # we get nan here sometimes, it's kills NN
                # TODO: handle it better. probably we should add zero paf, centered paf, or skip this completely. add a special paf?
                # 我认为可以不用去处理，在后处理时，把没有形成limb的点分配给距离最近的那个人即可
                print("Parts are too close to each other. Length is zero. Skipping")
                continue

            dx = dx / dnorm
            dy = dy / dnorm

            assert not isnan(dx) and not isnan(dy), "dnorm is zero, wtf"

            min_sx, max_sx = (x1, x2) if x1 < x2 else (x2, x1)
            min_sy, max_sy = (y1, y2) if y1 < y2 else (y2, y1)

            min_sx = int(round((min_sx - self.thre) / self.config.stride))
            min_sy = int(round((min_sy - self.thre) / self.config.stride))
            max_sx = int(round((max_sx + self.thre) / self.config.stride))
            max_sy = int(round((max_sy + self.thre) / self.config.stride))

            # check PAF off screen. do not really need to do it with max>grid size
            if max_sy < 0:
                continue

            if max_sx < 0:
                continue

            if min_sx < 0:
                min_sx = 0

            if min_sy < 0:
                min_sy = 0

            # TODO: check it again
            slice_x = slice(min_sx, max_sx)  # + 1    this mask is not only speed up but crops paf really. This copied from original code
            slice_y = slice(min_sy, max_sy) # + 1 因为array进行slices时不包括最后一个数    int g_y = min_y; g_y < max_y; g_y++ -- note strict <
            # tt = self.X[slice_y,slice_x]
            dist = distances(self.X[slice_y, slice_x], self.Y[slice_y, slice_x], self.paf_sigma, x1, y1, x2, y2)
            # 这里求的距离是在原始尺寸368*368的尺寸，而不是缩小8倍后在46*46上的距离，然后放到切片slice的位置上去
            # print(dist.shape)
            # TODO: averaging by pafs mentioned in the paper but never worked in C++ augmentation code
            heatmaps[slice_y, slice_x, layer][dist > 0] += dist[dist > 0]  # = dist * dx　若不做平均，则不进行累加

            count[slice_y, slice_x][dist > 0] += 1

        # TODO fixme: averaging by pafs mentioned in the paper but never worked in C++ augmentation code 我采用了平均
        heatmaps[:, :, layer][count > 0] /= count[count > 0]  # 这些都是矢量化（矩阵）操作

    def put_limbs(self, heatmaps, joints):
        """
         # 循环调用逐个channel生成ground truth的函数，最外层循环是对应某个limb的某一个channel
        """
        for (i, (fr, to)) in enumerate(self.config.limbs_conn):
            visible_from = joints[:, fr, 2] < 2  # 判断该点是否被标注了
            visible_to = joints[:, to, 2] < 2
            visible = visible_from & visible_to  # &: 按位取and, 只有两个节点都标注了才能生成paf, v=0,1时表示该点被标注了
            # In this project:  0 - marked but invisible, 1 - marked and visible, which is different from coco　dataset

            layer = self.config.paf_start + i
            self.put_vector_maps(heatmaps, layer, joints[visible, fr, 0:2], joints[visible, to, 0:2])


# parallel calculation distance from any number of points of arbitrary shape(X, Y),
# to line defined by segment (x1,y1) -> (x2, y2)

def gaussian(sigma, x, u):
    double_sigma2 = 2 * sigma ** 2
    y = np.exp(- (x - u) ** 2 / double_sigma2)
    return y


def distances(X, Y, sigma, x1, y1, x2, y2):  # TODO: change the paf area to ellipse
    """
    这里的distance函数实际上返回的是gauss分布的PAF
    # 实验发现在46*46尺寸的feature map上生成PAF，每个limb已经很短了，没有必要区分是直线区域还是椭圆区域
    # 点到两个端点所确定的直线的距离　classic formula is:
    # # d = [(x2-x1)*(y1-y)-(x1-x)*(y2-y1)] / sqrt((x2-x1)**2 + (y2-y1)**2)
    """

    xD = (x2 - x1)
    yD = (y2 - y1)
    detaX = x1 - X
    detaY = y1 - Y
    norm2 = sqrt(xD ** 2 + yD ** 2)  # 注意norm2是一个数而不是numpy数组,因为xD, yD都是一个数。单个数字运算math比numpy快
    dist = xD * detaY - detaX * yD  # 常数与numpy数组(X,Y是坐标数组）的运算，broadcast
    dist /= norm2
    dist = np.abs(dist)
    # ratiox = np.abs(detaX / (xD + 1e-8))
    # ratioy = np.abs(detaY / (yD + 1e-8))
    # ratio = np.where(ratiox < ratioy, ratiox, ratioy)
    # ratio = np.where(ratio > 1, 1, ratio)  # 不用　np.ones_like(ratio)也可以正常运行，并且会快一点点
    # ratio = np.where(ratio > 0.5, 1 - ratio, ratio)
    # oncurve_dist = b * np.sqrt(1 - np.square(ratio * 2))  # oncurve_dist计算的是椭圆边界上的点到长轴的垂直距离

    guass_dist = gaussian(sigma, dist, 0)
    guass_dist[guass_dist <= 0.01] = 0  # 同前面的关键点响应，太远的不要
    # b = thre
    # guass_dist[dist >= b] = 0

    return guass_dist


def test():
    hm = Heatmapper()
    d = distances(hm.X, hm.Y, 100, 100, 50, 150)
    print(d < 8.)


if __name__ == "__main__":
    np.set_printoptions(precision=1, linewidth=1000, suppress=True, threshold=100000)
    test()
