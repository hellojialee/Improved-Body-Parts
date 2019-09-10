"""
For 3-stage IMNH, 384*384 input training
"""
import numpy as np


class TrainingOpt:
    batch_size = 8  # for single process 整个分布式模型总的 batch size 是 batch_size*world_size
    learning_rate = 2.5e-5  # 2.5e-5  # 1e-4  # 2.5e-4  for single process 整个分布式模型总的是learning_rate*world_size
    config_name = "Canonical"
    hdf5_train_data = "./data/dataset/coco/link2coco2017/coco_train_dataset384.h5"
    hdf5_val_data = "./data/dataset/coco/link2coco2017/coco_val_dataset384.h5"
    nstack = 3  # stacked number of hourglass
    hourglass_inp_dim = 256
    increase = 128  # increased channels once down-sampling in the hourglass networks
    nstack_weight = [1, 1, 1]  # weight the losses between different stacks, stack 1, stack 2, stack 3...
    scale_weight = [0.2, 0.1, 0.4, 1, 4]  # weight the losses between different scales, scale 128, scale 64, scale 32...
    multi_task_weight = 0.1  # person mask loss vs keypoint loss
    keypoint_task_weight = 1  # keypoint heatmap loss vs body part heatmap loss
    ckpt_path = './link2checkpoints_distributed/L2_loss_3_stack/PoseNet_79_epoch.pth'


class TransformationParams:
    """ Hyper-parameters """
    def __init__(self, stride):
        #  TODO: tune # https://github.com/anatolix/keras_Realtime_Multi-Person_Pose_Estimation/issues/16
        #   We will firstly scale picture so that the height of the main person always will be 0.6 of picture.
        self.target_dist = 0.6
        self.scale_prob = 0.8  # scale probability, 0: never scale, 1: always scale
        self.scale_min = 0.75  # 之前训练设置的是0.8，但发现对小目标很不明显
        self.scale_max = 1.25
        self.max_rotate_degree = 40.  # todo: 看看hourglass中512设置的偏移
        self.center_perterb_max = 40.  # shift augmentation
        self.flip_prob = 0.5  # flip the image to force the network distinguish the mirror symmetrical keypoints
        self.tint_prob = 0.4  # ting着色操作比较耗时，如果按照0.5的概率进行，可能会使得每秒数据扩充图片减少10张
        self.sigma = 9  # 7 当是512输入时是9
        self.keypoint_gaussian_thre = 0.01  # 低于此值的keypoint gt高斯响应的区域被置零
        self.limb_gaussian_thre = 0.01  # 低于此值的body part gt高斯响应的区域被置零
        self.paf_sigma = 7  # 5 todo: sigma of PAF 对于PAF的分布，设其标准差为多少最合适呢
        # the value of sigma is important, there should be an equal contribution between foreground
        # and background heatmap pixels. Otherwise, there is a prior towards the background that forces the
        # network to converge to zero.
        self.paf_thre = 1 * stride  # equals to 1.0 * stride in this program, used to include the end-points of limbs 
        #  为了生成在PAF时，计算limb端点边界时使用，在最后一个feature map上
        # 将下界往下偏移1*stride像素质，把上界往上偏移1*stride个像素值


class CanonicalConfig:
    """Config used in ouf project"""
    def __init__(self, dense_conn=False):
        self.width = 384
        self.height = 384
        self.stride = 4  # 用于计算网络输出的feature map的尺寸
        # self.img_mean = [0.485, 0.456, 0.406]  # RGB format mean and standard variance
        # self.img_std = [0.229, 0.224, 0.225]
        self.parts = ["nose", "neck", "Rsho", "Relb", "Rwri", "Lsho", "Lelb", "Lwri", "Rhip", "Rkne", "Rank",
                      "Lhip", "Lkne", "Lank", "Reye", "Leye", "Rear", "Lear"]  # , "navel"

        self.num_parts = len(self.parts)
        self.parts_dict = dict(zip(self.parts, range(self.num_parts)))
        # help the network to detect body parts
        self.parts += ["background"]  # person mask作为背景之一, global config index: 42
        # force the network to learn to distinguish the keypoints from background
        self.parts += ['reverseKeypoint']  # 对所有keypoints取反作为背景二, global config index: 43
        self.num_parts_with_background = len(self.parts)
        self.leftParts, self.rightParts = CanonicalConfig.ltr_parts(self.parts_dict)

        # this numbers probably copied from matlab they are 1.. based not 0.. based
        self.limb_from = ['neck', 'neck', 'neck', 'neck', 'neck', 'nose', 'nose', 'Reye', 'Leye', 'neck', 'Rsho',
                          'Relb', 'neck', 'Lsho', 'Lelb',
                          'neck', 'Rhip', 'Rkne', 'neck', 'Lhip', 'Lkne', 'Rhip', 'Rsho', 'Lsho']

        self.limb_to = ['nose', 'Reye', 'Leye', 'Rear', 'Lear', 'Reye', 'Leye', 'Rear', 'Lear', 'Rsho', 'Relb', 'Rwri',
                        'Lsho', 'Lelb', 'Lwri',
                        'Rhip', 'Rkne', 'Rank', 'Lhip', 'Lkne', 'Lank', 'Lhip', 'Rear', 'Lear']

        self.limb_from = [self.parts_dict[n] for n in self.limb_from]
        self.limb_to = [self.parts_dict[n] for n in self.limb_to]

        assert self.limb_from == [x for x in
                                  [1, 1, 1, 1, 1, 0, 0, 14, 15, 1, 2, 3, 1, 5, 6, 1, 8, 9, 1, 11, 12, 8, 2, 5]]
        assert self.limb_to == [x for x in
                                [0, 14, 15, 16, 17, 14, 15, 16, 17, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 11, 16, 17]]

        self.limbs_conn = list(zip(self.limb_from, self.limb_to))

        self.paf_layers = len(self.limbs_conn)
        self.heat_layers = self.num_parts
        # layers of keypoint and body part heatmaps PLUS ++ 2 background
        self.num_layers = self.paf_layers + self.heat_layers + 2

        self.paf_start = 0
        self.heat_start = self.paf_layers  # Notice: 此处channel安排上，paf_map在前，heat_map在后
        self.bkg_start = self.paf_layers + self.heat_layers  # 用于feature map的计数,2个background的起始点

        self.offset_layers = 2 * self.num_parts
        self.offset_start = self.num_layers

        self.mask_shape = (self.height // self.stride, self.width // self.stride)  # 46, 46
        self.parts_shape = (self.height // self.stride, self.width // self.stride, self.num_layers)  # 46, 46, 59
        self.offset_shape = (self.height // self.stride, self.width // self.stride, self.offset_layers)

        self.transform_params = TransformationParams(self.stride)

        # Some configurations only used in inference process
        # map between original coco keypoint ids and  our keypoint ids
        # 因为CMU的定义和COCO官方对joint编号的定义不相同，所以需要通过mapping把编号改过来　　
        self.dt_gt_mapping = {0: 0, 1: None, 2: 6, 3: 8, 4: 10, 5: 5, 6: 7, 7: 9, 8: 12, 9: 14, 10: 16, 11: 11, 12: 13,
                              13: 15, 14: 2, 15: 1, 16: 4, 17: 3}  # , 18: None 没有使用肚脐

        # For the flip augmentation in the inference process only
        self.flip_heat_ord = np.array([0, 1, 5, 6, 7, 2, 3, 4, 11, 12, 13, 8, 9, 10, 15, 14, 17, 16, 18, 19])
        self.flip_paf_ord = np.array(
            [0, 2, 1, 4, 3, 6, 5, 8, 7, 12, 13, 14, 9, 10, 11, 18, 19, 20, 15, 16, 17, 21, 23, 22])

        self.draw_list = [0] + list(range(5, 22))

    @staticmethod  # staticmethod修饰的方法定义与普通函数是一样的, staticmethod支持类对象或者实例对方法的调用,即可使用A.f()或者a.f()
    def ltr_parts(parts_dict):
        # When we flip image left parts became right parts and vice versa.
        # This is the list of parts to exchange each other.
        leftParts = [parts_dict[p] for p in ["Lsho", "Lelb", "Lwri", "Lhip", "Lkne", "Lank", "Leye", "Lear"]]
        rightParts = [parts_dict[p] for p in ["Rsho", "Relb", "Rwri", "Rhip", "Rkne", "Rank", "Reye", "Rear"]]
        return leftParts, rightParts


class COCOSourceConfig:
    """Original config used in COCO dataset"""
    def __init__(self, hdf5_source):
        """
        Instantiate a COCOSource Config object，
        :param hdf5_source: the path only of hdf5 training materials generated by coco_mask_hdf5.py
        """
        self.hdf5_source = hdf5_source
        self.parts = ['nose', 'Leye', 'Reye', 'Lear', 'Rear', 'Lsho', 'Rsho', 'Lelb',
                      'Relb', 'Lwri', 'Rwri', 'Lhip', 'Rhip', 'Lkne', 'Rkne', 'Lank',
                      'Rank']  # coco数据集中关键点类型定义的顺序

        self.num_parts = len(self.parts)

        # for COCO neck is calculated like mean of 2 shoulders.
        self.parts_dict = dict(zip(self.parts, range(self.num_parts)))

    def convert(self, meta, global_config):
        """Convert COCO configuration (joint annotation) into ours configuration of this project"""
        # ----------------------------------------------
        # ---将coco config中对数据的定义改成CMU项目中的格式---
        # ----------------------------------------------

        joints = np.array(meta['joints'])

        assert joints.shape[1] == len(self.parts)

        result = np.zeros((joints.shape[0], global_config.num_parts, 3))
        # result是一个三维数组，shape[0]和人数有关，每一行即shape[1]和关节点数目有关，最后一维度长度为3,分别是x,y,v,即坐标值和可见标志位
        result[:, :, 2] = 3.
        # OURS - # 3 never marked up in this dataset, 2 - not marked up in this person, 1 - marked and visible,
        # 0 - marked but invisible. 关于visible值的重新定义在coco_mask_hdf5.py中完成了

        for p in self.parts:
            coco_id = self.parts_dict[p]

            if p in global_config.parts_dict:
                global_id = global_config.parts_dict[p]  # global_id是在该项目中使用的关节点编号，因为额外加入了neck(navel?)，与原始coco数据集中定义不同
                assert global_id != 1, "neck shouldn't be known yet"
                # assert global_id != 2, "navel shouldn't be known yet"
                result[:, global_id, :] = joints[:, coco_id, :]

        if 'neck' in global_config.parts_dict:  # neck point works as a root note
            neckG = global_config.parts_dict['neck']
            # parts_dict['neck']　＝　１, parts_dict是前面定义过的字典类型，节点名称：序号
            RshoC = self.parts_dict['Rsho']
            LshoC = self.parts_dict['Lsho']

            # no neck in coco database, we calculate it as average of shoulders
            #  here, we use 0 - hidden, 1 visible, 2 absent to represent the visibility of keypoints
            #  - it is not the same as coco values they processed by generate_hdf5

            # -------------------------------原始coco关于visible标签的定义－－－－－－－－－--------------－－－－－－－－－#
            # 第三个元素是个标志位v，v为0时表示这个关键点没有标注（这种情况下x = y = v = 0），
            # v为1时表示这个关键点标注了但是不可见（被遮挡了），v为2时表示这个关键点标注了同时也可见。
            # ------------------------------------ ----------------------------　－－－－－－－－－－－－－－－－－－－－－#

            both_shoulders_known = (joints[:, LshoC, 2] < 2) & (joints[:, RshoC, 2] < 2)  # 按位运算
            # 用True和False作为索引
            result[~both_shoulders_known, neckG, 2] = 2.  # otherwise they will be 3. aka 'never marked in this dataset'
            # ~both_shoulders_known bool类型按位取反
            result[both_shoulders_known, neckG, 0:2] = (joints[both_shoulders_known, RshoC, 0:2] +
                                                        joints[both_shoulders_known, LshoC, 0:2]) / 2
            result[both_shoulders_known, neckG, 2] = np.minimum(joints[both_shoulders_known, RshoC, 2],
                                                                joints[both_shoulders_known, LshoC, 2])
            # 最后一位是 visible　标志位，如果两个节点中有任何一个节点不可见，则中间节点neck设为不可见

        if 'navel' in global_config.parts_dict:  # add navel keypoint or not?
            navelG = global_config.parts_dict['navel']
            # parts_dict['navel']　＝ 2, parts_dict是前面定义过的字典类型，节点名称：序号
            RhipC = self.parts_dict['Rhip']
            LhipC = self.parts_dict['Lhip']

            # no navel in coco database, we calculate it as average of hipulders
            both_hipulders_known = (joints[:, LhipC, 2] < 2) & (joints[:, RhipC, 2] < 2)  # 按位运算
            # 用True和False作为索引
            result[
                ~both_hipulders_known, navelG, 2] = 2.  # otherwise they will be 3. aka 'never marked in this dataset'
            # ~both_hipulders_known bool类型按位取反
            result[both_hipulders_known, navelG, 0:2] = (joints[both_hipulders_known, RhipC, 0:2] +
                                                         joints[both_hipulders_known, LhipC, 0:2]) / 2
            result[both_hipulders_known, navelG, 2] = np.minimum(joints[both_hipulders_known, RhipC, 2],
                                                                 joints[both_hipulders_known, LhipC, 2])

        meta['joints'] = result

        return meta

    def repeat_mask(self, mask, global_config, joints=None):
        # 复制mask到个数到global_config通道数，但是我们不进行通道的复制，利用broadcast，节省内存
        mask = np.repeat(mask[:, :, np.newaxis], global_config.num_layers, axis=2)  # mask复制成了57个通道
        return mask

    def source(self):
        # return the path
        return self.hdf5_source


# more information on keypoints mapping is here
# https://github.com/ZheC/Realtime_Multi-Person_Pose_Estimation/issues/7

Configs = {}
Configs["Canonical"] = CanonicalConfig


def GetConfig(config_name):
    config = Configs[config_name]()  # () will instantiate an object of Configs[config_name] class

    dct = config.parts[:]
    dct = [None] * (config.num_layers - len(dct)) + dct

    for (i, (fr, to)) in enumerate(config.limbs_conn):
        name = "%s->%s" % (config.parts[fr], config.parts[to])
        print(i, name)
        x = i

        assert dct[x] is None
        dct[x] = name

    from pprint import pprint
    pprint(dict(zip(range(len(dct)), dct)))

    return config


if __name__ == "__main__":
    # test it
    foo = GetConfig("Canonical")
    print('the number of paf_layers is: %d, and the number of heat_layer is: %d' % (foo.paf_layers, foo.heat_layers))
