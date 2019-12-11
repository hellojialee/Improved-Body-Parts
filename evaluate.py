"""
Hint: please ingore the chinease annotations whcih may be wrong and they are just remains from old version.
"""
import sys

sys.path.append("..")  # 包含上级目录
import json
import math
import numpy as np
from itertools import product
from scipy.ndimage.filters import gaussian_filter
import tqdm
import time
import cv2
import torch
import torch.nn.functional as F
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from utils.config_reader import config_reader
from utils import util
from config.config import GetConfig, COCOSourceConfig, TrainingOpt
import matplotlib.pyplot as plt
from models.posenet import NetworkEval
import warnings
import os
import argparse

os.environ['CUDA_VISIBLE_DEVICES'] = "0"  # choose the available GPUs
warnings.filterwarnings("ignore")

# visualize
colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],
          [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],
          [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85], [193, 193, 255], [106, 106, 255], [20, 147, 255],
          [128, 114, 250], [130, 238, 238], [48, 167, 238], [180, 105, 255]]

parser = argparse.ArgumentParser(description='PoseNet Training')
parser.add_argument('--resume', '-r', action='store_true', default=True, help='resume from checkpoint')
parser.add_argument('--checkpoint_path', '-p', default='checkpoints_parallel', help='save path')
parser.add_argument('--max_grad_norm', default=5, type=float,
                    help="If the norm of the gradient vector exceeds this, re-normalize it to have the norm equal to max_grad_norm")
parser.add_argument('--output', type=str, default='result.jpg', help='output image')

parser.add_argument('--opt-level', type=str, default='O1')
parser.add_argument('--keep-batchnorm-fp32', type=str, default=None)
parser.add_argument('--loss-scale', type=str, default=None)

args = parser.parse_args()

# ###################################  Setup for some configurations ###########################################
opt = TrainingOpt()
config = GetConfig(opt.config_name)

limbSeq = config.limbs_conn
dt_gt_mapping = config.dt_gt_mapping
flip_heat_ord = config.flip_heat_ord
flip_paf_ord = config.flip_paf_ord


# ###############################################################################################################


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# ######################################  For evaluating time ######################################
batch_time = AverageMeter()


def predict(image, params, model, model_params, heat_layers, paf_layers, input_image_path):
    # print (image.shape)
    heatmap_avg = np.zeros((image.shape[0], image.shape[1], heat_layers))
    paf_avg = np.zeros((image.shape[0], image.shape[1], paf_layers))
    multiplier = [x * model_params['boxsize'] / image.shape[0] for x in params['scale_search']]  # 把368boxsize去掉,效果稍微下降了
    # multiplier = [1]  # fixme , add this line
    rotate_angle = params['rotation_search']
    for item in product(multiplier, rotate_angle):
        scale, angle = item
        # if scale * image.shape[0] > 2300 or scale * image.shape[1] > 3400:
        #   scale = min(2300 / image.shape[0], 3400 / image.shape[1])
        if scale * image.shape[0] > 2600 or scale * image.shape[1] > 3800:  # ### 我们更改了这里
            scale = min(2600 / image.shape[0], 3800 / image.shape[1])
            print("Input image: '{}' is too big, shrink it!".format(input_image_path))

        imageToTest = cv2.resize(image, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        imageToTest_padded, pad = util.padRightDownCorner(imageToTest, model_params['max_downsample'],
                                                          model_params['padValue'])

        # ################################# Important!  ###########################################
        # #############################  We use OpenCV to read image (BGR) all the time #######################
        # Input Tensor: a batch of images within [0,1], required shape in this project : (1, height, width, channels)
        input_img = np.float32(imageToTest_padded / 255)

        # ############################## Rotate the input image #####################3
        if angle != 0:
            rotate_matrix = cv2.getRotationMatrix2D((input_img.shape[0] / 2, input_img.shape[1] / 2), angle, 1)
            rotate_matrix_reverse = cv2.getRotationMatrix2D((input_img.shape[0] / 2, input_img.shape[1] / 2), -angle, 1)
            input_img = cv2.warpAffine(input_img, rotate_matrix, (0, 0))

        # input_img -= np.array(config.img_mean[::-1])  # Notice: OpenCV uses BGR format, reverse the last axises
        # input_img /= np.array(config.img_std[::-1])
        # ################################## add flip image ################################
        swap_image = input_img[:, ::-1, :].copy()
        # plt.imshow(swap_image[:, :, [2, 1, 0]])  # Opencv image format: BGR
        # plt.show()
        input_img = np.concatenate((input_img[None, ...], swap_image[None, ...]),
                                   axis=0)  # (2, height, width, channels)
        input_img = torch.from_numpy(input_img).cuda()
        # ###################################################################################
        # output tensor dtype: float 16
        output_tuple = posenet(input_img)

        output = output_tuple[-1][0].cpu().numpy()  # different scales can be shown

        output_blob = output[0].transpose((1, 2, 0))
        output_blob0 = output_blob[:, :, :config.paf_layers]
        output_blob1 = output_blob[:, :, config.paf_layers:config.num_layers]

        output_blob_flip = output[1].transpose((1, 2, 0))
        output_blob0_flip = output_blob_flip[:, :, :config.paf_layers]  # paf layers
        output_blob1_flip = output_blob_flip[:, :, config.paf_layers:config.num_layers]  # keypoint layers

        # ###################################################################################
        # ################################## flip ensemble ################################
        # ###################################################################################
        output_blob0_avg = (output_blob0 + output_blob0_flip[:, ::-1, :][:, :, flip_paf_ord]) / 2
        output_blob1_avg = (output_blob1 + output_blob1_flip[:, ::-1, :][:, :, flip_heat_ord]) / 2

        # extract outputs, resize, and remove padding
        heatmap = cv2.resize(output_blob1_avg, (0, 0), fx=model_params['stride'], fy=model_params['stride'],
                             interpolation=cv2.INTER_CUBIC)
        if angle != 0:
            heatmap = cv2.warpAffine(heatmap, rotate_matrix_reverse, (0, 0))

        heatmap = heatmap[pad[0]:imageToTest_padded.shape[0] - pad[2], pad[1]:imageToTest_padded.shape[1] - pad[3], :]
        heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_CUBIC)

        # output_blob0 is PAFs
        paf = cv2.resize(output_blob0_avg, (0, 0), fx=model_params['stride'], fy=model_params['stride'],
                         interpolation=cv2.INTER_CUBIC)
        if angle != 0:
            paf = cv2.warpAffine(paf, rotate_matrix_reverse, (0, 0))

        paf = paf[pad[0]:imageToTest_padded.shape[0] - pad[2], pad[1]:imageToTest_padded.shape[1] - pad[3], :]
        paf = cv2.resize(paf, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_CUBIC)

        heatmap_avg = heatmap_avg + heatmap / (len(multiplier) * len(rotate_angle))
        paf_avg = paf_avg + paf / (len(multiplier) * len(rotate_angle))

        # heatmap_avg = np.maximum(heatmap_avg, heatmap)
        # paf_avg = np.maximum(paf_avg, paf)  # 如果换成取最大，效果会变差，有很多误检

    return heatmap_avg, paf_avg


def find_peaks(heatmap_avg, params):
    all_peaks = []
    peak_counter = 0

    heatmap_avg = heatmap_avg.astype(np.float32)

    filter_map = heatmap_avg[:, :, :18].copy().transpose((2, 0, 1))[None, ...]
    filter_map = torch.from_numpy(filter_map).cuda()

    # # #######################   Add Gaussian smooth will be bad #######################
    # smoothing = util.GaussianSmoothing(18, 5, 3)
    # filter_map = F.pad(filter_map, (2, 2, 2, 2), mode='reflect')
    # filter_map = smoothing(filter_map)
    # # ######################################################################

    filter_map = util.keypoint_heatmap_nms(filter_map, kernel=3, thre=params['thre1'])
    filter_map = filter_map.cpu().numpy().squeeze().transpose((1, 2, 0))

    for part in range(18):
        map_ori = heatmap_avg[:, :, part]
        # heatmap_avg = gaussian_filter(heatmap_avg, sigma=3)  # TODO: fintune the sigma
        # 在某些情况下，需要对一个像素的周围的像素给予更多的重视。因此，可通过分配权重来重新计算这些周围点的值。
        # 这可通过高斯函数（钟形函数，即喇叭形数）的权重方案来解决。
        peaks_binary = filter_map[:, :, part]
        peaks = list(zip(np.nonzero(peaks_binary)[1], np.nonzero(peaks_binary)[0]))  # note reverse
        refined_peaks_with_score = [util.refine_centroid(map_ori, anchor, params['offset_radius']) for anchor in peaks]
        # peaks_with_score = [x + (map_ori[x[1], x[0]],) for x in refined_peaks]

        id = range(peak_counter, peak_counter + len(refined_peaks_with_score))
        peaks_with_score_and_id = [refined_peaks_with_score[i] + (id[i],) for i in range(len(id))]

        all_peaks.append(peaks_with_score_and_id)
        peak_counter += len(peaks)

    return all_peaks


def find_connections(all_peaks, paf_avg, image_width, params):
    connection_all = []
    special_k = []

    # 有多少个limb,就有多少个connection,相对应地就有多少个paf指向
    for k in range(len(limbSeq)):  # 最外层的循环是某一个limbSeq，因为mapIdx个数与之是一致对应的
        score_mid = paf_avg[:, :, k]  # 某一个channel上limb的响应热图, 它的长宽与原始输入图片大小一致，前面经过resize了
        # score_mid = gaussian_filter(score_mid, sigma=3)
        candA = all_peaks[limbSeq[k][0]]  # all_peaks是list,每一行也是一个list,保存了检测到的特定的parts(joints)
        # 注意具体处理时标号从0还是1开始。从收集的peaks中取出某类关键点（part)集合
        candB = all_peaks[limbSeq[k][1]]
        nA = len(candA)
        nB = len(candB)
        indexA, indexB = limbSeq[k]
        if (nA != 0 and nB != 0):
            connection_candidate = []
            for i in range(nA):
                for j in range(nB):
                    vec = np.subtract(candB[j][:2], candA[i][:2])
                    norm = math.sqrt(vec[0] * vec[0] + vec[1] * vec[1])
                    mid_num = min(int(round(norm + 1)), params['mid_num'])
                    # failure case when 2 body parts overlaps
                    if norm == 0:  # 为了跳过出现不同节点相互覆盖出现在同一个位置，也有说norm加一个接近0的项避免分母为0,详见：
                        # https://github.com/ZheC/Realtime_Multi-Person_Pose_Estimation/issues/54
                        continue

                    startend = list(zip(np.linspace(candA[i][0], candB[j][0], num=mid_num),
                                        np.linspace(candA[i][1], candB[j][1], num=mid_num)))

                    limb_response = np.array([score_mid[int(round(startend[I][1])), int(round(startend[I][0]))] \
                                              for I in range(len(startend))])
                    # limb_response 是代表某一个limb通道下的heat map响应

                    score_midpts = limb_response

                    score_with_dist_prior = sum(score_midpts) / len(score_midpts) + min(0.5 * image_width / norm - 1, 0)
                    # 这一项是为了惩罚过长的connection, 只有当长度大于图像高度的一半时才会惩罚 todo
                    # The term of sum(score_midpts)/len(score_midpts), see the link below.
                    # https://github.com/michalfaber/keras_Realtime_Multi-Person_Pose_Estimation/issues/48

                    criterion1 = len(np.nonzero(score_midpts > params['thre2'])[0]) >= params['connect_ration'] * len(
                        score_midpts)  # todo　改成了0.8
                    # fixme: tune 手动调整, 0.7 or 0.8
                    # 我认为这个判别标准是保证paf朝向的一致性  threshold = param['thre2'] =0.12
                    # CMU原始项目中parm['thre2'] = 0.05
                    criterion2 = score_with_dist_prior > 0
                    if criterion1 and criterion2:
                        connection_candidate.append([i, j, score_with_dist_prior, norm,
                                                     0.5 * score_with_dist_prior + 0.25 * candA[i][2] + 0.25 * candB[j][
                                                         2]])
                        # connection_candidate排序的依据是dist prior概率和两个端点heat map预测的概率值
                        # How to undersatand the criterion?

            connection_candidate = sorted(connection_candidate, key=lambda x: x[4], reverse=True)  # todo: sort by what
            # sorted 函数对可迭代对象，按照key参数指定的对象进行排序，revers=True是按照逆序排序, order: big->small

            connection = np.zeros((0, 6))
            for c in range(len(connection_candidate)):  # 根据confidence的顺序选择connections
                i, j, s, limb_len = connection_candidate[c][0:4]
                if (i not in connection[:, 3] and j not in connection[:, 4]):
                    # 进行判断确保不会出现两个端点集合A,B中，出现一个集合中的点与另外一个集合中两个点同时相连
                    connection = np.vstack([connection, [candA[i][3], candB[j][3], s, i, j, limb_len]])  # 后面会被使用
                    # candA[i][3], candB[j][3]是part的id编号
                    if (len(connection) >= min(nA, nB)):  # 会出现关节点不够连的情况
                        break
            connection_all.append(connection)
        else:
            special_k.append(k)
            connection_all.append([])

    return connection_all, special_k


def find_people(connection_all, special_k, all_peaks, params):
    # last number in each row is the total parts number of that person
    # the second last number in each row is the score of the overall configuration
    subset = -1 * np.ones((0, 20, 2))
    candidate = np.array([item for sublist in all_peaks for item in sublist])
    # candidate.shape = (94, 4). 列表解析式，两层循环，先从all peaks取，再从sublist中取。 all peaks是两层list

    for k in range(len(limbSeq)):
        # ---------------------------------------------------------
        # 外层循环limb  对应论文中，每一个limb就是一个子集，分limb处理,贪心策略?
        # special_K ,表示没有找到关节点对匹配的肢体
        if k not in special_k:  # 即　有与之相连的，这个paf(limb)是存在的
            partAs = connection_all[k][:, 0]  # limb端点part的序号，也就是保存在candidate中的  id号
            partBs = connection_all[k][:, 1]
            # connection_all 每一行是一个类型的limb,格式: N * [idA, idB, score, i, j]
            indexA, indexB = np.array(limbSeq[k])  # 此时处理limb k,limbSeq的端点parts，是parts的类别号.
            #  根据limbSeq列表的顺序依次考察某种类型的limb，从一个关节点到下一个关节点

            for i in range(len(connection_all[k])):  # 该层循环是分配connection　(partAs[i],partBs[i])到某个人　subset[]
                # connection_all[k]保存的是第k个类型的所有limb连接，可能有多个，也可能一个没有
                # ------------------------------------------------
                # 每一行的list保存的是一类limb(connection),遍历所有此类limb,一般的有多少个特定的limb就有多少个人

                found = 0
                subset_idx = [-1, -1]  # 每次循环只解决两个part，所以标记只需要两个flag
                for j in range(len(subset)):
                    # ----------------------------------------------
                    # 这一层循环是遍历所有的人

                    # 1:size(subset,1), 若subset.shape=(5,20), 则len(subset)=5，表示有5个人
                    # subset每一行对应的是一个人的18个关键点和number以及score的结果
                    if subset[j][indexA][0].astype(int) == (partAs[i]).astype(int) or subset[j][indexB][0].astype(
                            int) == partBs[i].astype(int):
                        # 看看这次考察的limb两个端点之一是否有一个已经在上一轮中出现过了,即是否已经分配给某人了
                        # 每一个最外层循环都只考虑一个limb，因此处理的时候就只会有两种part,即表示为partAs,partBs
                        if found >= 2:
                            print('************ error occurs! 3 joints sharing have been found  *******************')
                            continue
                        subset_idx[found] = j  # 标记一下，这个端点应该是第j个人的
                        found += 1

                if found == 1:
                    j = subset_idx[0]

                    if subset[j][indexB][0].astype(int) == -1 and \
                            params['len_rate'] * subset[j][-1][1] > connection_all[k][i][-1]:
                        # 如果新加入的limb比之前已经组装的limb长很多，也舍弃
                        # 如果这个人的当前点还没有被找到时，把这个点分配给这个人
                        # 这一个判断非常重要，因为第18和19个limb分别是 2->16, 5->17,这几个点已经在之前的limb中检测到了，
                        # 所以如果两次结果一致，不更改此时的part分配，否则又分配了一次，编号是覆盖了，但是继续运行下面代码，part数目
                        # 会加１，结果造成一个人的part之和>18。不过如果两侧预测limb端点结果不同，还是会出现number of part>18，造成多检
                        # FIXME: 没有利用好冗余的connection信息，最后两个limb的端点与之前循环过程中重复了，但没有利用聚合，
                        #  只是直接覆盖，其实直接覆盖是为了弥补漏检

                        subset[j][indexB][0] = partBs[i]  # partBs[i]是limb其中一个端点的id号码
                        subset[j][indexB][1] = connection_all[k][i][2]  # 保存这个点被留下来的置信度
                        subset[j][-1][0] += 1
                        # last number in each row is the total parts number of that person

                        # # subset[j][-2][1]用来记录不包括当前新加入的类型节点时的总体初始置信度，引入它是为了避免下次迭代出现同类型关键点，覆盖时重复相加了置信度
                        # subset[j][-2][1] = subset[j][-2][0]  # 因为是不包括此类节点的初始值，所以只会赋值一次 !!

                        subset[j][-2][0] += candidate[partBs[i].astype(int), 2] + connection_all[k][i][2]
                        subset[j][-1][1] = max(connection_all[k][i][-1], subset[j][-1][1])

                        # the second last number in each row is the score of the overall configuration

                    elif subset[j][indexB][0].astype(int) != partBs[i].astype(int):
                        if subset[j][indexB][1] >= connection_all[k][i][2]:
                            # 如果考察的这个limb连接没有已经存在的可信，则跳过
                            pass

                        else:
                            # 否则用当前的limb端点覆盖已经存在的点，并且在这之前，减去已存在关节点的置信度和连接它的limb置信度
                            if params['len_rate'] * subset[j][-1][1] <= connection_all[k][i][-1]:
                                continue
                            # 减去之前的节点置信度和limb置信度
                            subset[j][-2][0] -= candidate[subset[j][indexB][0].astype(int), 2] + subset[j][indexB][1]

                            # 添加当前节点
                            subset[j][indexB][0] = partBs[i]
                            subset[j][indexB][1] = connection_all[k][i][2]  # 保存这个点被留下来的置信度
                            subset[j][-2][0] += candidate[partBs[i].astype(int), 2] + connection_all[k][i][2]

                            subset[j][-1][1] = max(connection_all[k][i][-1], subset[j][-1][1])

                    # overlap the reassigned keypoint with higher score
                    #  如果是添加冗余连接的重复的点，用新的更加高的冗余连接概率取代原来连接的相同的关节点的概率
                    # -- 对上面问题的回答： 使用前500进行测试，发现加上这个能提高0.1%，没有什么区别
                    elif subset[j][indexB][0].astype(int) == partBs[i].astype(int) and subset[j][indexB][1] <= \
                            connection_all[k][i][2]:
                        # 否则用当前的limb端点覆盖已经存在的点，并且在这之前，减去已存在关节点的置信度和连接它的limb置信度

                        # 减去之前的节点置信度和limb置信度
                        subset[j][-2][0] -= candidate[subset[j][indexB][0].astype(int), 2] + subset[j][indexB][1]

                        # 添加当前节点
                        subset[j][indexB][0] = partBs[i]
                        subset[j][indexB][1] = connection_all[k][i][2]  # 保存这个点被留下来的置信度
                        subset[j][-2][0] += candidate[partBs[i].astype(int), 2] + connection_all[k][i][2]

                        subset[j][-1][1] = max(connection_all[k][i][-1], subset[j][-1][1])

                    else:
                        pass

                elif found == 2:  # if found 2 and disjoint, merge them (disjoint：不相交)
                    # -----------------------------------------------------
                    # 如果肢体组成的关节点A,B分别连到了两个人体，则表明这两个人体应该组成一个人体，
                    # 则合并两个人体（当肢体是按顺序拼接情况下不存在这样的状况）
                    # --------------------------------------------------

                    # 说明组装的过程中，有断掉的情况（有limb或者说connection缺失），在之前重复开辟了一个sub person,其实他们是同一个人上的
                    # If humans H1 and H2 share a part index with the same coordinates, they are sharing the same part!
                    #  H1 and H2 are, therefore, the same humans. So we merge both sets into H1 and remove H2.
                    # https://arvrjourney.com/human-pose-estimation-using-openpose-with-tensorflow-part-2-e78ab9104fc8
                    # 该代码与链接中的做法有差异，个人认为链接中的更加合理而且更容易理解
                    j1, j2 = subset_idx

                    membership1 = ((subset[j1][..., 0] >= 0).astype(int))[:-2]  # 用[:,0]也可
                    membership2 = ((subset[j2][..., 0] >= 0).astype(int))[:-2]
                    membership = membership1 + membership2
                    # [:-2]不包括最后个数项与scores项
                    # 这些点应该属于同一个人,将这个人所有类型关键点（端点part)个数逐个相加
                    if len(np.nonzero(membership == 2)[0]) == 0:  # if found 2 and disjoint, merge them

                        min_limb1 = np.min(subset[j1, :-2, 1][membership1 == 1])
                        min_limb2 = np.min(subset[j2, :-2, 1][membership2 == 1])
                        min_tolerance = min(min_limb1, min_limb2)  # 计算允许进行拼接的最低置信度

                        if connection_all[k][i][2] < params['connection_tole'] * min_tolerance or params['len_rate'] * \
                                subset[j1][-1][1] <= connection_all[k][i][-1]:
                            # 如果merge这两个身体部分的置信度不够大，或者当前这个limb明显大于已存在的limb的长度，则不进行连接
                            # todo: finetune the tolerance of connection
                            continue  #

                        subset[j1][:-2][...] += (subset[j2][:-2][...] + 1)
                        # 对于没有节点标记的地方，因为两行subset相应位置处都是-1,所以合并之后没有节点的部分依旧是-１
                        # 把不相交的两个subset[j1],[j2]中的id号进行相加，从而完成合并，这里+1是因为默认没有找到关键点初始值是-1

                        subset[j1][-2:][:, 0] += subset[j2][-2:][:, 0]  # 两行subset的点的个数和总置信度相加

                        subset[j1][-2][0] += connection_all[k][i][2]
                        subset[j1][-1][1] = max(connection_all[k][i][-1], subset[j1][-1][1])
                        # 注意：　因为是disjoint的两行subset点的merge，因此先前存在的节点的置信度之前已经被加过了 !! 这里只需要再加当前考察的limb的置信度
                        subset = np.delete(subset, j2, 0)

                    else:
                        # 出现了两个人同时竞争一个limb的情况，并且这两个人不是同一个人，通过比较两个人包含此limb的置信度来决定，
                        # 当前limb的节点应该分配给谁，同时把之前的那个与当前节点相连的节点(即partsA[i])从另一个人(subset)的节点集合中删除
                        if connection_all[k][i][0] in subset[j1, :-2, 0]:
                            c1 = np.where(subset[j1, :-2, 0] == connection_all[k][i][0])
                            c2 = np.where(subset[j2, :-2, 0] == connection_all[k][i][1])
                        else:
                            c1 = np.where(subset[j1, :-2, 0] == connection_all[k][i][1])
                            c2 = np.where(subset[j2, :-2, 0] == connection_all[k][i][0])

                        # c1, c2分别是当前limb连接到j1人的第c1个关节点，j2人的第c2个关节点
                        c1 = int(c1[0])
                        c2 = int(c2[0])
                        assert c1 != c2, "an candidate keypoint is used twice, shared by two people"

                        # 如果当前考察的limb置信度比已经存在的两个人连接的置信度小，则跳过，否则删除已存在的不可信的连接节点。
                        if connection_all[k][i][2] < subset[j1][c1][1] and connection_all[k][i][2] < subset[j2][c2][1]:
                            continue  # the trick here is useful

                        small_j = j1
                        big_j = j2
                        remove_c = c1

                        if subset[j1][c1][1] > subset[j2][c2][1]:
                            small_j = j2
                            big_j = j1
                            remove_c = c2

                        # 删除和当前limb有连接,并且置信度低的那个人的节点   # FIXME:  获取不删除？为了检测更多？
                        if params['remove_recon'] > 0:
                            subset[small_j][-2][0] -= candidate[subset[small_j][remove_c][0].astype(int), 2] + \
                                                      subset[small_j][remove_c][1]
                            subset[small_j][remove_c][0] = -1
                            subset[small_j][remove_c][1] = -1
                            subset[small_j][-1][0] -= 1

                # if find no partA in the subset, create a new subset
                # 如果肢体组成的关节点A,B没有被连接到某个人体则组成新的人体
                # ------------------------------------------------------------------
                #    1.Sort each possible connection by its score.
                #    2.The connection with the highest score is indeed a final connection.
                #    3.Move to next possible connection. If no parts of this connection have
                #    been assigned to a final connection before, this is a final connection.
                #    第三点是说，如果下一个可能的连接没有与之前的连接有共享端点的话，会被视为最终的连接，加入row
                #    4.Repeat the step 3 until we are done.
                # 说明见：　https://arvrjourney.com/human-pose-estimation-using-openpose-with-tensorflow-part-2-e78ab9104fc8

                elif not found and k < len(limbSeq):
                    # Fixme: 检查是否正确
                    #  原始的时候是18,因为我加了limb，所以是24,因为真正的limb是0~16，最后两个17,18是额外的不是limb
                    # FIXME: 但是后面画limb的时候没有把鼻子和眼睛耳朵的连线画上，要改进
                    row = -1 * np.ones((20, 2))
                    row[indexA][0] = partAs[i]
                    row[indexA][1] = connection_all[k][i][2]
                    row[indexB][0] = partBs[i]
                    row[indexB][1] = connection_all[k][i][2]
                    row[-1][0] = 2
                    row[-1][1] = connection_all[k][i][-1]  # 这一位用来记录上轮连接limb时的长度，用来作为下一轮连接的先验知识
                    row[-2][0] = sum(candidate[connection_all[k][i, :2].astype(int), 2]) + connection_all[k][i][2]
                    # 两个端点的置信度+limb连接的置信度
                    # print('create a new subset:  ', row, '\t')
                    row = row[np.newaxis, :, :]  # 为了进行concatenate，需要插入一个轴
                    subset = np.concatenate((subset, row), axis=0)
        # todo: solve the unmathced keypoint? 将没有被分配到一些人身上的点分配给距离它们近，并且缺少此类节点的人身上？或许这样做坏处更多
    # delete some rows of subset which has few parts occur
    deleteIdx = []
    for i in range(len(subset)):
        if subset[i][-1][0] < 2 or subset[i][-2][0] / subset[i][-1][
            0] < 0.45:  # subset[i][-1][0] < 4 or  FIXME: 一些预知需要调整，并且coco更侧重检测到而不是虚警
            deleteIdx.append(i)
    subset = np.delete(subset, deleteIdx, axis=0)

    return subset, candidate


def process(input_image_path, params, model, model_params, heat_layers, paf_layers):
    oriImg = cv2.imread(input_image_path)  # B,G,R order !!
    # print(input_image)
    torch.cuda.empty_cache()
    heatmap, paf = predict(oriImg, params, model, model_params, heat_layers, paf_layers, input_image_path)

    end = time.time()  # ############# Evaluating the keypoint assignment algorithm ######

    all_peaks = find_peaks(heatmap, params)
    connection_all, special_k = find_connections(all_peaks, paf, oriImg.shape[0], params)
    subset, candidate = find_people(connection_all, special_k, all_peaks, params)

    batch_time.update((time.time() - end))
    if show_eval_speed:
        print('==================>Test: [{0}/{1}]\t'
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'Speed {2:.3f} ({3:.3f})\t'.format(
            1, 1,
            1 / batch_time.val,
            1 / batch_time.avg,
            batch_time=batch_time))

    keypoints = []
    for s in subset[..., 0]:
        keypoint_indexes = s[:18]
        person_keypoint_coordinates = []
        for index in keypoint_indexes:
            if index == -1:
                # "No candidate for keypoint" # 标志为-1的part是没有检测到的
                X, Y = 0, 0
            else:
                X, Y = candidate[index.astype(int)][:2]
            person_keypoint_coordinates.append((X, Y))
        person_keypoint_coordinates_coco = [None] * 17

        for dt_index, gt_index in dt_gt_mapping.items():
            if gt_index is None:
                continue
            person_keypoint_coordinates_coco[gt_index] = person_keypoint_coordinates[dt_index]

        keypoints.append((person_keypoint_coordinates_coco, 1 - 1.0 / s[18]))
        # s[18] is the score, s[19] is the number of keypoint
    return keypoints


def get_image_name(coco, image_id):
    return coco.imgs[image_id]['file_name']


def predict_many(coco, images_directory, validation_ids, params, model, model_params, heat_layers, paf_layers):
    assert (not set(validation_ids).difference(set(coco.getImgIds())))

    keypoints = {}

    for image_id in tqdm.tqdm(validation_ids):
        image_name = get_image_name(coco, image_id)
        image_name = os.path.join(images_directory, image_name)
        keypoints[image_id] = process(image_name, dict(params), model, dict(model_params), heat_layers + 2, paf_layers)
        # fixme: heat_layers + 1 if you use background keypoint  !!!
    return keypoints


def format_results(keypoints, resFile):
    format_keypoints = []
    # Question: do we need to sort the detections by scores before evaluation ?
    # -- I think we do not have. COCO will select the top 20 automatically
    for image_id, people in keypoints.items():
        for keypoint_list, score in people:
            format_keypoint_list = []
            for x, y in keypoint_list:
                for v in [x, y, 1 if x > 0 or y > 0 else 0]:  # int(x), int(y)
                    # 坐标取了整数,为了减少文件的大小，如果x,y有一个有值，那么标记这个点为可见。　如果x or y =0,令v=0,coco只评测v>0的点
                    format_keypoint_list.append(v)

            format_keypoints.append({
                "image_id": image_id,
                "category_id": 1,
                "keypoints": format_keypoint_list,
                "score": score,
            })

    json.dump(format_keypoints, open(resFile, 'w'))


def validation(model, dump_name, validation_ids=None, dataset='val2017'):
    annType = 'keypoints'
    prefix = 'person_keypoints'

    dataDir = 'data/dataset/coco/link2coco2017'

    # # # #############################################################################
    # For evaluation on validation set
    annFile = '%s/annotations/%s_%s.json' % (dataDir, prefix, dataset)
    print(annFile)
    cocoGt = COCO(annFile)

    if validation_ids == None:  # todo: we can set the validataion image ids here  !!!!!!
        validation_ids = cocoGt.getImgIds()[:500]  # [:1000] we can change the range of COCO validation images here
    # # #############################################################################

    # #############################################################################
    # For evaluation on test-dev set
    # annFile = 'data/dataset/coco/link2coco2017/annotations_trainval_info/image_info_test-dev2017.json' # image_info_test2017.json
    # cocoGt = COCO(annFile)
    # validation_ids = cocoGt.getImgIds()
    # #############################################################################

    resFile = '%s/results/%s_%s_%s100_results.json'
    resFile = resFile % (dataDir, prefix, dataset, dump_name)
    print('the path of detected keypoint file is: ', resFile)
    os.makedirs(os.path.dirname(resFile), exist_ok=True)

    keypoints = predict_many(cocoGt, os.path.join(dataDir, dataset), validation_ids, params, model, model_params,
                             config.heat_layers, config.paf_layers)
    format_results(keypoints, resFile)
    cocoDt = cocoGt.loadRes(resFile)
    cocoEval = COCOeval(cocoGt, cocoDt, annType)
    cocoEval.params.imgIds = validation_ids
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()
    return cocoEval


if __name__ == "__main__":
    posenet = NetworkEval(opt, config, bn=True)

    print('Resuming from checkpoint ...... ')
    checkpoint = torch.load(opt.ckpt_path, map_location=torch.device('cpu'))  # map to cpu to save the gpu memory
    posenet.load_state_dict(checkpoint['weights'])  # 加入他人训练的模型，可能需要忽略部分层，则strict=False
    print('Network weights have been resumed from checkpoint...')

    if torch.cuda.is_available():
        posenet.cuda()

    from apex import amp

    posenet = amp.initialize(posenet, opt_level=args.opt_level,
                             keep_batchnorm_fp32=args.keep_batchnorm_fp32,
                             loss_scale=args.loss_scale)
    posenet.eval()  # set eval mode is important

    params, model_params = config_reader()

    # show keypoint assignment algorithm speed
    show_eval_speed = False

    with torch.no_grad():
        eval_result_original = validation(posenet, dump_name='residual_4_hourglass_focal_epoch_52_512_input_1scale_max',
                                          dataset='val2017')  # 'val2017'

    print('over!')

    #  若是在test数据集上进行预测并写结果，则
    # annFile='/home/jia/Desktop/keras_Realtime_Multi-Person_Pose_Estimation-new-generation/dataset/coco/link2coco2017/annotations_trainval_info/image_info_test2017.json'
    # cocoGt = COCO(annFile)
    # validation_ids = cocoGt.getImgIds() 将获得带有image id的一个list
