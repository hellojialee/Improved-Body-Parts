import sys
import json
import math
import numpy as np
from scipy.ndimage.filters import gaussian_filter
import tqdm
import time
import cv2
import torch
import torch.optim as optim
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


os.environ['CUDA_VISIBLE_DEVICES'] = "2"  # choose the available GPUs
warnings.filterwarnings("ignore")

limbSeq = [[1, 0], [1, 14], [1, 15], [1, 16], [1, 17], [0, 14], [0, 15], [14, 16], [15, 17],
           [1, 2], [2, 3], [3, 4], [1, 5], [5, 6], [6, 7], [1, 8], [8, 9],
           [9, 10], [1, 11], [11, 12], [12, 13], [8, 11], [2, 16], [5, 17]]


mapIdx = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11], [12, 13], [14, 15], [16, 17], [18, 19], [20, 21], [22, 23],
          [24, 25], [26, 27], [28, 29], [30, 31], [32, 33], [34, 35], [36, 37], [38, 39], [40, 41], [42, 43], [44, 45],
          [46, 47]]

# visualize
colors = [[	128, 114, 250], [130, 238, 238], [48, 167, 238], [180, 105, 255], [255, 0, 0], [255, 85, 0], [255, 170, 0],
          [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], [0, 255, 85], [0, 255, 170], [0, 255, 255],
          [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], [170, 0, 255], [255, 0, 255], [255, 0, 170],
          [255, 0, 85], [193, 193, 255], [106, 106, 255], [20, 147, 255]]

dt_gt_mapping = {0: 0, 1: None, 2: 6, 3: 8, 4: 10, 5: 5, 6: 7, 7: 9, 8: 12, 9: 14, 10: 16, 11: 11, 12: 13, 13: 15,
                 14: 2, 15: 1, 16: 4, 17: 3, 18: None}  # , 18: None 没有使用肚脐

torch.cuda.empty_cache()
parser = argparse.ArgumentParser(description='PoseNet Training')
parser.add_argument('--resume', '-r', action='store_true', default=True, help='resume from checkpoint')
parser.add_argument('--checkpoint_path', '-p',  default='checkpoints_parallel', help='save path')
parser.add_argument('--max_grad_norm', default=5, type=float,
    help="If the norm of the gradient vector exceeds this, re-normalize it to have the norm equal to max_grad_norm")
parser.add_argument('--image', type=str, default='try_image/cocotry2.jpg', help='input image')  # required=True
parser.add_argument('--output', type=str, default='result.jpg', help='output image')

parser.add_argument('--opt-level', type=str, default='O1')
parser.add_argument('--keep-batchnorm-fp32', type=str, default=None)
parser.add_argument('--loss-scale', type=str, default=None)

args = parser.parse_args()

checkpoint_path = args.checkpoint_path
opt = TrainingOpt()
config = GetConfig(opt.config_name)


def show_color_vector(oriImg, paf_avg, heatmap_avg):
    hsv = np.zeros_like(oriImg)
    hsv[..., 1] = 255

    mag, ang = cv2.cartToPolar(paf_avg[:, :, 17], 1.5 * paf_avg[:, :, 17])  # 设置不同的系数，可以使得显示颜色不同

    # 将弧度转换为角度，同时OpenCV中的H范围是180(0 - 179)，所以再除以2
    # 完成后将结果赋给HSV的H通道，不同的角度(方向)以不同颜色表示
    # 对于不同方向，产生不同色调
    # hsv[...,0]等价于hsv[:,:,0]
    hsv[..., 0] = ang * 180 / np.pi / 2

    # 将矢量大小标准化到0-255范围。因为OpenCV中V分量对应的取值范围是256
    # 对于同一H、S而言，向量的大小越大，对应颜色越亮
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    # 最后，将生成好的HSV图像转换为BGR颜色空间
    limb_flow = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    plt.imshow(oriImg[:, :, [2, 1, 0]])
    plt.imshow(limb_flow, alpha=.5)
    plt.show()

    plt.imshow(oriImg[:, :, [2, 1, 0]])  # show a keypoint
    plt.imshow(heatmap_avg[:, :, 12], alpha=.5)
    plt.show()


def process(input_image, params, model_params, heat_layers, paf_layers):
    oriImg = cv2.imread(input_image)  # B,G,R order.    训练数据的读入也是用opencv，因此也是B, G, R顺序
    # oriImg = cv2.resize(oriImg, (768, 768))
    # oriImg = cv2.flip(oriImg, 1) 因为训练时作了flip，所以用这种方式提升并没有作用
    multiplier = [x * model_params['boxsize'] / oriImg.shape[0] for x in params['scale_search']]  # 按照图片高度进行缩放
    # multipier = [0.21749408983451538, 0.43498817966903075, 0.6524822695035462, 0.8699763593380615],首先把输入图像高度变成368,然后再做缩放

    heatmap_avg = np.zeros((oriImg.shape[0], oriImg.shape[1], heat_layers))  # fixme if you change the number of keypoints
    paf_avg = np.zeros((oriImg.shape[0], oriImg.shape[1], paf_layers))

    multiplier = multiplier
    for m in range(len(multiplier)):
        scale = multiplier[m]

        imageToTest = cv2.resize(oriImg, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)  # cv2.INTER_CUBIC
        imageToTest_padded, pad = util.padRightDownCorner(imageToTest, model_params['max_downsample'],
                                                          model_params['padValue'])

        # ################################# Important!  ###########################################
        # Input Tensor: a batch of images within [0,1], required shape (1, height, width, channels)
        input_img = np.float32(imageToTest_padded[None, ...] / 255)
        input_img = torch.from_numpy(input_img).cuda()
        # output tensor dtype: float 16
        output_tuple = posenet(input_img)

        output = output_tuple[-1][0].cpu().numpy()  # different scales can be shown
        output_blob = output[0].transpose((1, 2, 0))
        output_blob0 = output_blob[:, :, :config.paf_layers]
        output_blob1 = output_blob[:, :, config.paf_layers:config.num_layers]

        # extract outputs, resize, and remove padding
        heatmap = cv2.resize(output_blob1, (0, 0), fx=model_params['stride'], fy=model_params['stride'],
                             interpolation=cv2.INTER_CUBIC)
        heatmap = heatmap[:imageToTest_padded.shape[0] - pad[2], :imageToTest_padded.shape[1] - pad[3], :]
        heatmap = cv2.resize(heatmap, (oriImg.shape[1], oriImg.shape[0]), interpolation=cv2.INTER_CUBIC)

        # output_blob0 is PAFs
        paf = cv2.resize(output_blob0, (0, 0), fx=model_params['stride'], fy=model_params['stride'],
                         interpolation=cv2.INTER_CUBIC)
        paf = paf[:imageToTest_padded.shape[0] - pad[2], :imageToTest_padded.shape[1] - pad[3], :]
        paf = cv2.resize(paf, (oriImg.shape[1], oriImg.shape[0]), interpolation=cv2.INTER_CUBIC)
        heatmap_avg = heatmap_avg + heatmap / len(multiplier)
        paf_avg = paf_avg + paf / len(multiplier)

        # heatmap_avg = np.maximum(heatmap_avg, heatmap)
        # paf_avg = np.maximum(paf_avg, paf)  # 如果换成取最大，效果会变差，有很多误检


    all_peaks = []
    peak_counter = 0
    # --------------------------------------------------------------------------------------- #
    # ------------------------  show the limb and foreground channel  -----------------------#
    # --------------------------------------------------------------------------------------- #

    show_color_vector(oriImg, paf_avg, heatmap_avg)

    # --------------------------------------------------------------------------------------- #
    # ####################################################################################### #
    # ------------------------- find keypoints  ---------------------------------------------#
    # ####################################################################################### #
    # --------------------------------------------------------------------------------------- #

    for part in range(18):  # fixme: 没有对背景（序号19）取非极大值抑制NMS
        map_ori = heatmap_avg[:, :, part]
        map = gaussian_filter(map_ori, sigma=3)  # fixme: use gaussian blure?
        # map = map_ori
        # map up 是值
        map_up = np.zeros(map.shape)  # 为了找到比相邻像素值都大的位置
        map_up[1:, :] = map[:-1, :]
        map_down = np.zeros(map.shape)  # todo： NMS with a sliding window of 3*3
        map_down[:-1, :] = map[1:, :]
        map_left = np.zeros(map.shape)
        map_left[:, 1:] = map[:, :-1]
        map_right = np.zeros(map.shape)
        map_right[:, :-1] = map[:, 1:]

        # 对于左上角右下角之类相邻的像素也做抑制，变成完全的3*3窗口
        map_left_up = np.zeros(map.shape)
        map_left_up[1:, :] = map_left[:-1, :]
        map_right_up = np.zeros(map.shape)
        map_right_up[1:, :] = map_right[:-1, :]
        map_left_down = np.zeros(map.shape)
        map_left_down[:-1, :] = map_left[1:, :]
        map_right_down = np.zeros(map.shape)
        map_right_down[:-1, :] = map_right[1:, :]

        peaks_binary = np.logical_and.reduce((map >= map_left, map >= map_right,
                                              map >= map_up, map >= map_down, map >= map_right_up,
                                              map >= map_right_down,
                                              map >= map_left_up, map >= map_left_down,
                                              map > params['thre1']))  # fixme: finetue it
        # reduce 方法和Python的reduce函数类似，它沿着axis轴对array进行操作，
        # 相当于将<op>运算符插入到沿axis轴的所有子数组或者元素当中。
        # param['thre1'] = 0.1

        peaks = list(zip(np.nonzero(peaks_binary)[1], np.nonzero(peaks_binary)[0]))
        # note reverse. xy坐标系和图像坐标系
        # np.nonzero: Return the indices of the elements that are non-zero
        peaks_with_score = [x + (map_ori[x[1], x[0]],) for x in peaks]  # 列表解析式，生产的是list
        id = range(peak_counter, peak_counter + len(peaks))
        peaks_with_score_and_id = [peaks_with_score[i] + (id[i],) for i in range(len(id))]
        # 为每一个相应peak (parts)都依次编了一个号

        all_peaks.append(peaks_with_score_and_id)
        # all_peaks.append 如果此种关节类型没有元素，append一个空的list []，例如all_peaks[19]:
        # [(205, 484, 0.9319216758012772, 25),
        # (595, 484, 0.777797631919384, 26),
        # (343, 490, 0.8145177364349365, 27), ....
        peak_counter += len(peaks)

    # --------------------------------------------------------------------------------------- #
    # ####################################################################################### #
    # ----------------------------- find connections -----------------------------------------#
    # ####################################################################################### #
    # --------------------------------------------------------------------------------------- #

    connection_all = []
    special_k = []

    # 有多少个limb,就有多少个connection,相对应地就有多少个paf指向
    for k in range(len(mapIdx)):  # 最外层的循环是某一个limbSeq，因为mapIdx个数与之是一致对应的
        score_mid = paf_avg[:, :, mapIdx[k][0] // 2]  # 某一个channel上limb的响应热图, 它的长宽与原始输入图片大小一致，前面经过resize了
        # score_mid = gaussian_filter(orginal_score_mid, sigma=3)  fixme: use gaussisan blure?
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
                    mid_num = max(int(norm), 10)
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

                    score_with_dist_prior = sum(score_midpts) / len(score_midpts) + min(0.5 * oriImg.shape[0] / norm - 1, 0)
                    # 这一项是为了惩罚过长的connection, 只有当长度大于图像高度的一半时才会惩罚 todo
                    # The term of sum(score_midpts)/len(score_midpts), see the link below.
                    # https://github.com/michalfaber/keras_Realtime_Multi-Person_Pose_Estimation/issues/48

                    criterion1 = len(np.nonzero(score_midpts > params['thre2'])[0]) > params['connect_ration'] * len(score_midpts)  # fixme: tune 手动调整, 本来是 > 0.8*len
                    # 我认为这个判别标准是保证paf朝向的一致性  param['thre2']
                    # parm['thre2'] = 0.05
                    criterion2 = score_with_dist_prior > 0

                    if criterion1 and criterion2:
                        connection_candidate.append([i, j, score_with_dist_prior, norm,
                                                     0.5 * score_with_dist_prior + 0.25 * candA[i][2] + 0.25 * candB[j][2]])
                        # todo:直接把两种类型概率相加不合理
                        # connection_candidate排序的依据是dist prior概率和两个端点heat map预测的概率值
                        # How to undersatand the criterion?

            connection_candidate = sorted(connection_candidate, key=lambda x: x[4], reverse=True)
            # sorted 函数对可迭代对象，按照key参数指定的对象进行排序，revers=True是按照逆序排序，sort之后可以把最可能是limb的留下，而把和最可能是limb的端点竞争的端点删除

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
            # 一个空的[]也能加入到list中，这一句是必须的！因为connection_all的数据结构是每一行代表一类limb connection

    # --------------------------------------------------------------------------------------- #
    # ####################################################################################### #
    # --------------------------------- find people ------------------------------------------#
    # ####################################################################################### #
    # --------------------------------------------------------------------------------------- #

    # last number in each row is the total parts number of that person
    # the second last number in each row is the score of the overall configuration
    subset = -1 * np.ones((0, 20, 2))
    candidate = np.array([item for sublist in all_peaks for item in sublist])
    # candidate[:, 2] *= 0.5  # FIXME: change it? part confidence * 0.5
    # candidate.shape = (94, 4). 列表解析式，两层循环，先从all peaks取，再从sublist中取。 all peaks是两层list

    for k in range(len(mapIdx)):
        # ---------------------------------------------------------
        # 外层循环limb  对应论文中，每一个limb就是一个子集，分limb处理,贪心策略?
        # special_K ,表示没有找到关节点对匹配的肢体
        if k not in special_k:  # 即　有与之相连的，这个paf(limb)是存在的
            partAs = connection_all[k][:, 0]  # limb端点part的序号，也就是保存在candidate中的  id号
            partBs = connection_all[k][:, 1]  # limb端点part的序号，也就是保存在candidate中的  id号
            # connection_all 每一行是一个类型的limb,每一行格式: N * [idA, idB, score, i, j]
            indexA, indexB = np.array(limbSeq[k])  # 此时处理limb k,limbSeq的两个端点parts，是parts的类别号.
            #  根据limbSeq列表的顺序依次考察某种类型的limb，从一个关节点到下一个关节点

            for i in range(len(connection_all[k])):  # 该层循环是分配k类型的limb connection　(partAs[i],partBs[i])到某个人　subset[]
                # ------------------------------------------------
                # 每一行的list保存的是一类limb(connection),遍历所有此类limb,一般的有多少个特定的limb就有多少个人

                found = 0
                subset_idx = [-1, -1]  # 每次循环只解决两个part，所以标记只需要两个flag
                for j in range(len(subset)):
                    # ----------------------------------------------
                    # 这一层循环是遍历所有的人

                    # 1:size(subset,1), 若subset.shape=(5,20), 则len(subset)=5，表示有5个人
                    # subset每一行对应的是一个人的18个关键点和number以及score的结果
                    if subset[j][indexA][0].astype(int) == (partAs[i]).astype(int) or subset[j][indexB][0].astype(int) == partBs[i].astype(int):
                        # 看看这次考察的limb两个端点之一是否有一个已经在上一轮中出现过了,即是否已经分配给某人了
                        # 每一个最外层循环都只考虑一个limb，因此处理的时候就只会有两种part,即表示为partAs,partBs
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
                        # FIXME: 没有利用好冗余的connection信息，最后两个limb的端点与之前循环过程中重复了，但没有利用聚合，只是直接覆盖，其实直接覆盖是为了弥补漏检

                        subset[j][indexB][0] = partBs[i]  # partBs[i]是limb其中一个端点的id号码
                        subset[j][indexB][1] = connection_all[k][i][2]  # 保存这个点被留下来的置信度
                        subset[j][-1][0] += 1
                        # last number in each row is the total parts number of that person

                        # # subset[j][-2][1]用来记录不包括当前新加入的类型节点时的总体初始置信度，引入它是为了避免下次迭代出现同类型关键点，覆盖时重复相加了置信度
                        # subset[j][-2][1] = subset[j][-2][0]  # 因为是不包括此类节点的初始值，所以只会赋值一次 !!

                        subset[j][-2][0] += candidate[partBs[i].astype(int), 2] + connection_all[k][i][2]
                        # candidate的格式为：  (343, 490, 0.8145177364349365, 27), ....
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
                        min_tolerance = min(min_limb1, min_limb2)  # 计算允许进行拼接的置信度

                        if connection_all[k][i][2] < params['connection_tole'] * min_tolerance or params['len_rate'] * subset[j1][-1][1] <= connection_all[k][i][-1]:
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

                        # 删除和当前limb有连接,并且置信度低的那个人的节点
                        subset[small_j][-2][0] -= candidate[subset[small_j][remove_c][0].astype(int), 2] + subset[small_j][remove_c][1]
                        subset[small_j][remove_c][0] = -1  # todo
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

                elif not found and k < 24:
                    # Fixme: 原始的时候是18,因为我加了limb，所以是24,因为真正的limb是0~16，最后两个17,18是额外的不是limb
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

    # delete some rows of subset which has few parts occur
    deleteIdx = []
    for i in range(len(subset)):
        if subset[i][-1][0] < 4 or subset[i][-2][0] / subset[i][-1][0] < 0.45:  # (params['thre1'] + params['thre2']) / 2:  # todo: tune, it matters much!
            deleteIdx.append(i)
    subset = np.delete(subset, deleteIdx, axis=0)

    canvas = cv2.imread(input_image)  # B,G,R order
    # canvas = oriImg
    keypoints = []

    for s in subset[..., 0]:
        keypoint_indexes = s[:18]  # 定义的keypoint一共有18个
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

        keypoints.append((person_keypoint_coordinates_coco, 1 - 1.0 / s[-2]))  # s[19] is the score

    for i in range(len(keypoints)):
        print('the {}th keypoint detection result is : '.format(i), keypoints[i])

    # 画所有的峰值
    # for i in range(18):
    #     #     rgba = np.array(cmap(1 - i/18. - 1./36))
    #     #     rgba[0:3] *= 255
    #     for j in range(len(all_peaks[i])):  # all_peaks保存了坐标，score以及id
    #         # 注意x,y坐标谁在前谁在后，在这个project中有点混乱
    #         cv2.circle(canvas, all_peaks[i][j][0:2], 3, colors[i], thickness=-1)

    stickwidth = 3
    # 画所有的骨架
    draw_list = [0] + list(range(5, 22))
    for i in draw_list:  # 画出18个limb　Fixme：我设计了25个limb,画的limb顺序需要调整，相应color数也要增加
        for n in range(len(subset)):
            index = subset[n][np.array(limbSeq[i])][..., 0]
            if -1 in index:  # 有-1说明没有对应的关节点与之相连,即有一个类型的part没有缺失，无法连接成limb
                continue
            # 在上一个cell中有　canvas = cv2.imread(test_image) # B,G,R order
            cur_canvas = canvas.copy()
            Y = candidate[index.astype(int), 0]
            X = candidate[index.astype(int), 1]
            mX = np.mean(X)
            mY = np.mean(Y)
            length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
            angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
            polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), 3), int(angle), 0,
                                       360, 1)

            cv2.circle(cur_canvas, (int(Y[0]), int(X[0])), 4, color=[0, 0, 0], thickness=2)
            cv2.circle(cur_canvas, (int(Y[1]), int(X[1])), 4, color=[0, 0, 0], thickness=2)

            cv2.fillConvexPoly(cur_canvas, polygon, colors[i])
            canvas = cv2.addWeighted(canvas, 0.4, cur_canvas, 0.6, 0)
    return canvas


if __name__ == '__main__':
    input_image = args.image
    output = args.output

    posenet = NetworkEval(opt, config)

    print('Resuming from checkpoint ...... ')
    checkpoint = torch.load(opt.ckpt_path, map_location=torch.device('cpu'))  # map to cpu to save the gpu memory

    # #################################################
    # from collections import OrderedDict
    #
    # new_state_dict = OrderedDict()
    # for k, v in checkpoint['weights'].items():
    #     # if 'out' in k or 'merge' in k:
    #     #     continue
    #     name = 'module.' + k  # add prefix 'module.'
    #     new_state_dict[name] = v
    # posenet.load_state_dict(new_state_dict)  # , strict=False
    # # #################################################

    checkpoint = torch.load(opt.ckpt_path, map_location=torch.device('cpu'))  # map to cpu to save the gpu memory
    posenet.load_state_dict(checkpoint['weights'])  # 加入他人训练的模型，可能需要忽略部分层，则strict=False
    print('Network weights have been resumed from checkpoint...')

    if torch.cuda.is_available():
        posenet.cuda()
    posenet.eval()   # set eval mode is important

    from apex import amp

    optimizer = optim.Adam(posenet.parameters())  # Redundant.

    posenet, optimizer = amp.initialize(posenet, optimizer,
                                        opt_level=args.opt_level,
                                        keep_batchnorm_fp32=args.keep_batchnorm_fp32,
                                        loss_scale=args.loss_scale)

    tic = time.time()
    print('start processing...')
    # load config
    params, model_params = config_reader()
    tic = time.time()
    # generate image with body parts
    with torch.no_grad():
        canvas = process(input_image, params, model_params, config.heat_layers+2, config.paf_layers)  # background + 2

    toc = time.time()
    print('processing time is %.5f' % (toc - tic))

    # TODO: the prediction is slow, how to fix it? Not solved yet. see:
    #  https://github.com/anatolix/keras_Realtime_Multi-Person_Pose_Estimation/issues/5

    cv2.namedWindow("result", cv2.WINDOW_AUTOSIZE)  # cv2.WINDOW_NORMAL 自动适合的窗口大小
    cv2.imshow('result', canvas)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite(output, canvas)

