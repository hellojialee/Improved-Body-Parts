# coding:utf-8
import numpy as np
from math import cos, sin, pi
import cv2
import random
import matplotlib.pyplot as plt


class AugmentSelection:
    def __init__(self, flip=False, tint=False, degree=0., crop=(0, 0), scale=1.):
        self.flip = flip
        self.tint = tint
        self.degree = degree  # rotate
        self.crop = crop  # shift actually
        self.scale = scale

    @staticmethod  # staticmethod支持类对象或者实例对方法的调用
    def random(transform_params):
        flip = random.uniform(0., 1.) < transform_params.flip_prob
        tint = random.uniform(0., 1.) < transform_params.tint_prob
        degree = random.uniform(-1., 1.) * transform_params.max_rotate_degree

        scale = (transform_params.scale_max - transform_params.scale_min) * random.uniform(0., 1.) + \
                transform_params.scale_min \
            if random.uniform(0., 1.) < transform_params.scale_prob else 1.

        x_offset = int(random.uniform(-1., 1.) * transform_params.center_perterb_max)
        y_offset = int(random.uniform(-1., 1.) * transform_params.center_perterb_max)

        return AugmentSelection(flip, tint, degree, (x_offset, y_offset), scale)

    @staticmethod
    def unrandom():
        flip = False
        tint = False
        degree = 0.
        scale = 1.
        x_offset = 0
        y_offset = 0

        return AugmentSelection(flip, tint, degree, (x_offset, y_offset), scale)

    def affine(self, center, scale_self, config):
        # the main idea: we will do all image transformations with one affine matrix.
        # this saves lot of cpu and make code significantly shorter
        # same affine matrix could be used to transform joint coordinates afterwards

        A = cos(self.degree / 180. * pi)
        B = sin(self.degree / 180. * pi)

        scale_size = config.transform_params.target_dist / scale_self * self.scale
        # target_dist是调整人占整个图像的比例吗？
        # It used in picture augmentation during training. Rough meaning is "height of main person on image should
        # be approximately 0.6 of the original image size". It used in this file in my code:
        # https://github.com/anatolix/keras_Realtime_Multi-Person_Pose_Estimation/blob/master/py_rmpe_server/py_rmpe_transformer.py
        # This mean we will scale picture so height of person always will be 0.6 of picture.
        # After it we apply random scaling (self.scale) from 0.6 to 1.1
        (width, height) = center
        center_x = width + self.crop[0]
        center_y = height + self.crop[1]

        # 为了处理方便，将图像变换到以原点为中心
        center2zero = np.array([[1., 0., -center_x],
                                [0., 1., -center_y],
                                [0., 0., 1.]])

        rotate = np.array([[A, B, 0],
                           [-B, A, 0],
                           [0, 0, 1.]])

        scale = np.array([[scale_size, 0, 0],
                          [0, scale_size, 0],
                          [0, 0, 1.]])

        flip = np.array([[-1 if self.flip else 1., 0., 0.],
                         [0., 1., 0.],
                         [0., 0., 1.]])

        # 最后再从原点中心变换到指定图像大小尺寸的中心上去
        center2center = np.array([[1., 0., config.width // 2],
                                  [0., 1., config.height // 2],
                                  [0., 0., 1.]])

        # order of combination is reversed
        # 这取决于坐标是行向量还是列向量，对应变换矩阵是左乘还是右乘，此处坐标用的是列向量形式
        combined = center2center.dot(flip).dot(scale).dot(rotate).dot(center2zero)

        return combined[0:2], scale_size


class Transformer:
    def __init__(self, config):

        self.config = config

    @staticmethod  # staticmethod支持类对象或者类的实例对方法的调用
    def distort_color(img):
        # uint8 input，opencv outputs Hue、Saturation、Value ranges are: [0,180)，[0,256)，[0,256)
        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.int16)
        hsv_img[:, :, 0] = np.maximum(np.minimum(hsv_img[:, :, 0] - 10 + np.random.randint(20 + 1), 179),
                                      0)  # hue
        hsv_img[:, :, 1] = np.maximum(np.minimum(hsv_img[:, :, 1] - 40 + np.random.randint(80 + 1), 255),
                                      0)  # saturation
        hsv_img[:, :, 2] = np.maximum(np.minimum(hsv_img[:, :, 2] - 30 + np.random.randint(60 + 1), 255),
                                      0)  # value
        hsv_img = hsv_img.astype(np.uint8)

        distorted_img = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)
        return distorted_img

    def transform(self, img, mask_miss, mask_all, meta, aug=None):
        """ If aug is None, then do random augmentation. Input original data and output transformed data """

        if aug is None:
            aug = AugmentSelection.random(self.config.transform_params)

        if aug.tint:
            img = self.distort_color(img)
        # # ------------------------------------------------------------------------------------ #

        # warp picture and mask
        assert meta['scale_provided'][0] != 0, "************ scale_proviede is zero, dividing zero! ***********"

        M, scale_size = aug.affine(meta['objpos'][0], meta['scale_provided'][0], self.config)
        # 根据排名第一的main person进行图像缩放
        # need to understand this,
        # scale_provided[0] is height of main person divided by 512, calculated in generate_hdf5.py
        # print(img.shape)

        # 变换之后还会缩放到config.height大小, (self.config.height, self.config.width)　指定的是返回图像的尺寸
        img = cv2.warpAffine(img, M, (self.config.height, self.config.width), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=(127, 127, 127))
        # for debug, see the transformed data
        # plt.imshow(img[:,:,[2,1,0]])  # opencv imread ---> BGR order
        # plt.show()

        # mask也要做一致的变换
        mask_miss = cv2.warpAffine(mask_miss, M, (self.config.height, self.config.width), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=255)  # cv2.INTER_CUBIC适合放大

        mask_miss = cv2.resize(mask_miss, self.config.mask_shape,     # mask shape　是统一的 46*46
                          interpolation=cv2.INTER_AREA)

        mask_all = cv2.warpAffine(mask_all, M, (self.config.height, self.config.width), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        #
        mask_all = cv2.resize(mask_all, self.config.mask_shape,    # mask shape　是统一的 46*46
                          interpolation=cv2.INTER_AREA)

        # # debug usage: show the image and corresponding mask area
        # # mask areas are in dark when display
        # plt.imshow(img[:, :, [2, 1, 0]])
        # plt.imshow(np.repeat(mask_image_size[:, :, np.newaxis], 3, axis=2), alpha=0.5)  # mask_all
        # plt.show()

        # warp key points
        # Issue: joint could be cropped by augmentation, in this case we should mark it as invisible.
        # update: may be we don't need it actually, original code removed part sliced more than half totally,
        # may be we should keep it
        original_points = meta['joints'].copy()
        original_points[:, :, 2] = 1  # we reuse 3rd column in completely different way here, it is hack
        # -----------------------------------------------------------------------------　#
        # 需要添加超过边界时此时设为2吗？ 上面的update已经回答了这个问题，在heatmaper.py生成时使用了slice
        # -----------------------------------------------------------------------------　#

        # we use 0 - hidden, 1 visible, 2 absent - it is not coco values they processed by generate_hdf5
        converted_points = np.matmul(M, original_points.transpose([0, 2, 1])).transpose([0, 2, 1])  # 关键点的坐标变换
        # 从矩阵相乘的方式来看，坐标表示用的是列向量，所以是左乘变换矩阵
        meta['joints'][:, :, 0:2] = converted_points

        # we just made image flip, i.e. right leg just became left leg, and vice versa
        if aug.flip:
            tmpLeft = meta['joints'][:, self.config.leftParts, :]  # 通过中间的缓存变量进行交换
            tmpRight = meta['joints'][:, self.config.rightParts, :]
            meta['joints'][:, self.config.leftParts, :] = tmpRight
            meta['joints'][:, self.config.rightParts, :] = tmpLeft
        # print('*********************', img.shape, meta['joints'].shape)
        # meta['joints'].shape = (num_of_person, 18, 3)，其中18是18个关键点，3代表（x,y,v)

        # normalize image to 0~1 here to save gpu/cpu time
        # mask - 除以255之后，被mask地方是0.0,没有mask地方是1.0
        # return transformed data as flot32 format
        return img.astype(np.float32)/255., mask_miss.astype(np.float32)/255., mask_all.astype(np.float32)/255., meta

