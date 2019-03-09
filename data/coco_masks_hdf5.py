#!/usr/bin/env python
# coding:utf-8
"""
Python script for generating the training and validation hdf5 data from MSCOCO dataset
"""
from pycocotools.coco import COCO
from scipy.spatial.distance import cdist
import numpy as np
import cv2
import os
import os.path
import h5py
import json
import time
import matplotlib.pyplot as plt

dataset_dir = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'data/dataset/coco/link2coco2017'))

tr_anno_path = os.path.join(dataset_dir, "annotations/person_keypoints_train2017.json")  # 只取keypoint的标注信息
tr_img_dir = os.path.join(dataset_dir, "train2017")

val_anno_path = os.path.join(dataset_dir, "annotations/person_keypoints_val2017.json")
val_img_dir = os.path.join(dataset_dir, "val2017")

datasets = [
    (val_anno_path, val_img_dir, "COCO_val"),  # it is important to have 'val' in validation dataset name,
    # look for 'val' below
    (tr_anno_path, tr_img_dir, "COCO")
]


tr_hdf5_path = os.path.join(dataset_dir, "coco_train_dataset512.h5")
val_hdf5_path = os.path.join(dataset_dir, "coco_val_dataset512.h5")

val_size = 100  # size of validation set  设置的validation subset的大小.　剩余的val数据将选入train数据中
image_size = 512  # 用于训练网络时，设定的训练集图片的统一尺寸　


def make_mask(img_dir, img_id, img_anns, coco):
    """Mask all unannotated people (including the crowd which has no keypoint annotation)"""
    # 对于某一张图像和图像中所有人或者人群的标注做处理
    # mask miss 和　mask all的解释：
    # mask_all记录了一张图像上所有人的mask(包括单个人和一群人)，　而mask miss是为了掩盖掉那些是人，有segmentation但是没有标注keypoint
    # 需要注意，mak miss是把没有keypoint的区域变成0，而mask all是把所有人区域变成１。最后mask_miss又从0,1 bool型变到0~255的uint8
    # apply mask miss if p["num_keypoints"] <= 0 i.e. person is segmented but have no keypoints(joints)
    # "people who has little annotation(<5), who has little scale(<32*32) and who is so close to 'main_person'" are
    # not masked, they just can't be selected as main person of image, but they are still passed to the netwrok.
    # ----------------------------------------------------------------------------------- #
    #  (我认为这样做可以使得网络对于小于32大小的或者节点少于5的不敏感，使得训练时抑制网络可以屏蔽它们）
    # ----------------------------------------------------------------------------------- #
    # see:　https://github.com/michalfaber/keras_Realtime_Multi-Person_Pose_Estimation/issues/43
    # About just "mask". It contains pixel segment borders. For my perception it never used in the algorithm,
    # may be it was created for some visualisation purposes.

    img_path = os.path.join(img_dir, "%012d.jpg" % img_id)

    if not os.path.exists(img_path):
        raise IOError("image path dose not exist: %s" % img_path)

    img = cv2.imread(img_path)
    h, w, c = img.shape
    # mask:　https://github.com/michalfaber/keras_Realtime_Multi-Person_Pose_Estimation/issues/8#issuecomment-342977756
    mask_all = np.zeros((h, w), dtype=np.uint8)
    mask_miss = np.zeros((h, w), dtype=np.uint8)

    flag = 0
    for p in img_anns:
        seg = p["segmentation"]   # seg is just a boarder of an object, see annotation file

        if p["iscrowd"] == 1:   # the handel of crowd
            # segmentation格式取决于这个实例是一个单个的对象（即iscrowd=0，将使用polygons格式）还是一组对象（即iscrowd=1，将使用RLE格式）
            mask_crowd = coco.annToMask(p)

            temp = np.bitwise_and(mask_all, mask_crowd)  # 我感觉temp是之前mask_all与当前crowded instances的mask的交集IOU
            mask_crowd = mask_crowd - temp

            flag += 1
            continue
        else:
            mask = coco.annToMask(p)

        mask_all = np.bitwise_or(mask, mask_all)  # mask_all记录了一张图像上所有人的mask
        # mask_all never used for anything except visualization !!!!
        if p["num_keypoints"] <= 0:
            mask_miss = np.bitwise_or(mask, mask_miss)

    if flag < 1:
        mask_miss = np.logical_not(mask_miss)
    elif flag == 1:
        # mask the few keypoint and crowded persons at the same time ! mask areas are 0 !
        mask_miss = np.logical_not(np.bitwise_or(mask_miss, mask_crowd))
        # mask all the persons including crowd, mask area are 1 !
        mask_all = np.bitwise_or(mask_all, mask_crowd)
    else:
        raise Exception("crowd segments > 1")  # 对一个区域，只能存在一个segment,不存在一个区域同时属于某两个instances的部分

    mask_miss = mask_miss.astype(np.uint8)
    mask_miss *= 255  # 保存的　mask_miss　的数值非0即255

    mask_all = mask_all.astype(np.uint8)
    mask_all *= 255  # 保存的　mask_miss　的数值非0即255
    # Mask miss is multiplied by the loss,
    # so masked areas are 0. (被mask的区域是0) I.e. second mask is real mask miss. First mask (mask_all) is just for visuals.
    mask_concat = np.concatenate((mask_miss[:, :, np.newaxis], mask_all[:, :, np.newaxis]), axis=2)

    # # # # ------------ 注释部分代码用来显示mask crowded instance  --------------
    # # # print('***************', mask_miss.min(), mask_miss.max())
    # plt.imshow(img[:,:,[2,1,0]])
    # plt.show()
    # plt.imshow(np.repeat(mask_concat[:, :, 1][:,:,np.newaxis], 3, axis=2))  # mask_all
    # plt.show()
    # plt.imshow(np.repeat(mask_concat[:, :, 0][:,:,np.newaxis], 3, axis=2))  # mask_miss
    # plt.show()
    # print('show')
    # # # -------------------------------------------------------------------

    return img,  mask_concat


def process_image(image_rec, img_id, image_index, img_anns, dataset_type):
    # 针对处理的对象是　某一张id对应的image　及这张图上所有人的标注

    numPeople = len(img_anns)
    h, w = image_rec['height'], image_rec['width']
    print("Image ID: ", img_id, '  ,', 'number of people: ', numPeople)

    all_persons = []

    for p in range(numPeople):

        pers = dict()  # 用字典类型保存数据

        person_center = [img_anns[p]["bbox"][0] + img_anns[p]["bbox"][2] / 2,  # 标注格式为(x, y, w, h)
                         img_anns[p]["bbox"][1] + img_anns[p]["bbox"][3] / 2]

        pers["objpos"] = person_center  # objpos 代表的是人体的中心位置
        pers["bbox"] = img_anns[p]["bbox"]
        pers["segment_area"] = img_anns[p]["area"]
        pers["num_keypoints"] = img_anns[p]["num_keypoints"]

        anno = img_anns[p]["keypoints"]

        pers["joint"] = np.zeros((17, 3))
        for part in range(17):
            pers["joint"][part, 0] = anno[part * 3]  # x坐标， 因为每一个part的信息有(x, y, v) 3个值
            pers["joint"][part, 1] = anno[part * 3 + 1]  # y坐标，注意x，y坐标的先后顺序

            # visible/invisible
            # COCO - Each keypoint has a 0-indexed location x,y and a visibility flag v defined as v=0: not labeled (in which case x=y=0), v=1: labeled but not visible, and v=2: labeled and visible.
            # OURS - # 3 never marked up in this dataset, 2 - not marked up in this person, 1 - marked and visible, 0 - marked but invisible
            if anno[part * 3 + 2] == 2:   # +2　对应visibility的值
                pers["joint"][part, 2] = 1
            elif anno[part * 3 + 2] == 1:
                pers["joint"][part, 2] = 0
            else:
                pers["joint"][part, 2] = 2

        pers["scale_provided"] = img_anns[p]["bbox"][3] / image_size  # 每一个person占比
        # img_anns[p]["bbox"][3] 对应的是人体框的高度h　!　

        all_persons.append(pers)

    main_persons = []
    prev_center = []

    """ 
    The idea of main person: each picture is feeded for each main person every epoch centered around this main 
    person( btw it is not working in michalfaber code, thats why quality of model is bit lower). 
    Secondary persons will not get such privilege, if they close to one of main persons they will be visible on crop,
     and heatmap/paf will be calculated, if they too far then ... bad luck, they never be machine learning star :)
    # https://github.com/anatolix/keras_Realtime_Multi-Person_Pose_Estimation/issues/14
    """
    # https://github.com/anatolix/keras_Realtime_Multi-Person_Pose_Estimation/issues/14

    for pers in all_persons:  # 最外层循环是所有的单个的人，从all person中选取main　person

        # skip this person if parts number is too low or if
        # segmentation area is too small
        if pers["num_keypoints"] < 5 or pers["segment_area"] < 32 * 32:
            #  we do not select the person, which is too small or has too few keypoints, as the main person
            # 用于居中图片的main person是用来训练网络的主力，所以关键点和人的大小要合理，关键点少的可能离其他main person近
            continue

        person_center = pers["objpos"]

        # skip this person if the distance to exiting person is too small
        flag = 0
        for pc in prev_center:  # prev_center 保存了person　center 以及人体框长和宽中的最大值
            a = np.expand_dims(pc[:2], axis=0)  # prev_center是一个坐标: (x, y)
            b = np.expand_dims(person_center, axis=0)
            dist = cdist(a, b)[0]   # by default, computing the euclidean distance
            # pc[2] 代表人体框长和宽中最大的那一边,原始程序中把距离main person特别近，<0.3的person不再作为下一个main person
            # 因为这样没有必要，离得很近，图片裁出来的部分基本一样
            if dist < pc[2] * 0.3:
                flag = 1
                continue

        if flag == 1:
            continue

        main_persons.append(pers)  # 若和之前已经存在的人距离不是非常近，并且标注点不少，则添加当前这个人，认为是一个main person，
        # 可能是为了避免生成差异较小的训练图片，因为如果距离很近的话，依然会包括相邻的人的
        # main_persions是一个list, pers是一个dic字典，排序在第一的main person将享有优先权
        prev_center.append(np.append(person_center, max(img_anns[p]["bbox"][2], img_anns[p]["bbox"][3])))
        # 保存了person　center 以及人体框长和宽中的最大值


    template = dict()
    template["dataset"] = dataset_type  # coco or coco_val

    if image_index < val_size and 'val' in dataset_type:  # notice: 'val' in 'COCOval'   >>>　True
        isValidation = 1
    else:
        isValidation = 0

    template["isValidation"] = isValidation
    template["img_width"] = w
    template["img_height"] = h  # 这个是整个图像的w, h
    template["image_id"] = img_id  # 将包含这些人姿态数据的image的id也保存起来
    template["annolist_index"] = image_index
    template["img_path"] = '%012d.jpg' % img_id

    # 外部大循环是每一张图片，内部（也就是下面这个）循环是一个图片中的所有main_persons, 也就是说每一个main_person都会轮流变成排序第一的人，将享有图片居中的特权
    for p, person in enumerate(main_persons):  # p是list的索引序号，person是dic类型的信息内容

        instance = template.copy()  # template is a dictionary type

        instance["objpos"] = [main_persons[p]["objpos"]]
        instance["joints"] = [main_persons[p]["joint"].tolist()]  # Return the array as a (possibly nested) list
        instance["scale_provided"] = [ main_persons[p]["scale_provided"] ]
        #  while training they scale main person to be approximately image size(368 pix in our case). But after
        #  it they do random scaling 0.6-1.1. So this is very logical network never learned libs(and PAFs) could be
        #  larger than half of image.
        lenOthers = 0

        for ot, operson in enumerate(all_persons):  # other person

            if person is operson:
                assert not "people_index" in instance, "several main persons? couldn't be"
                instance["people_index"] = ot
                continue

            if operson["num_keypoints"] == 0:
                continue

            instance["joints"].append(all_persons[ot]["joint"].tolist())
            instance["scale_provided"].append(all_persons[ot]["scale_provided"])
            instance["objpos"].append(all_persons[ot]["objpos"])

            lenOthers += 1

        assert "people_index" in instance, "No main person index"
        instance["numOtherPeople"] = lenOthers
        yield instance   # 带有yield关键字，是generator
        #  除了crowd和关键点很少的人以外，既打包了main person，也保存了其他非main person，对于一个instance，每次只有一个优先权main person


def writeImage(grp, img_grp, data, img, mask_miss, count, image_id, mask_grp=None):
    """
    Write hdf5 files
    :param grp: annotation hdf5 group
    :param img_grp: image hdf5 group
    :param data: annotation handled
    :param img: image returned by mask_mask()
    :param mask_miss: mask returned by mask_mask()
    :param count:
    :param image_id: image index
    :param mask_grp: mask hdf5 group
    :return: nothing
    """
    serializable_meta = data
    serializable_meta['count'] = count

    nop = data['numOtherPeople']

    assert len(serializable_meta['joints']) == 1 + nop, [len(serializable_meta['joints']), 1 + nop]
    assert len(serializable_meta['scale_provided']) == 1 + nop, [len(serializable_meta['scale_provided']), 1 + nop]
    assert len(serializable_meta['objpos']) == 1 + nop, [len(serializable_meta['objpos']), 1 + nop]

    img_key = "%012d" % image_id
    if not img_key in img_grp:

        if mask_grp is None:  # 为了兼容MPII没有mask的情形
            img_and_mask = np.concatenate((img, mask_miss[..., None]), axis=2)
            # create_dataset 返回创建的hdf5对象(此处为img_ds)，并且此对象被添加到img_key(若dataset name不为None)中
            img_ds = img_grp.create_dataset(img_key, data=img_and_mask, chunks=None)
        else:
            # _, img_bin = cv2.imencode(".jpg", img)  # encode compress, we do not need it actually, delete cv2.imencode
            # _, img_mask = cv2.imencode(".png", mask_miss) # data= img_bin, data = img_mask
            img_ds1 = img_grp.create_dataset(img_key, data=img, chunks=None)
            img_ds2 = mask_grp.create_dataset(img_key, data=mask_miss, chunks=None)

    key = '%07d' % count
    required = {'image':img_key, 'joints': serializable_meta['joints'], 'objpos': serializable_meta['objpos'], 'scale_provided': serializable_meta['scale_provided'] }
    ds = grp.create_dataset(key, data=json.dumps(required), chunks=None)
    ds.attrs['meta'] = json.dumps(serializable_meta)

    print('Writing sample %d' % count)


def process():

    tr_h5 = h5py.File(tr_hdf5_path, 'w')
    tr_grp = tr_h5.create_group("dataset")
    tr_write_count = 0
    tr_grp_img = tr_h5.create_group("images")
    tr_grp_mask = tr_h5.create_group("masks")  # in fact, is mask_concat rather than mask_miss  NOTICE !!!

    val_h5 = h5py.File(val_hdf5_path, 'w')
    val_grp = val_h5.create_group("dataset")
    val_write_count = 0
    val_grp_img = val_h5.create_group("images")
    val_grp_mask = val_h5.create_group("masks")

    for _, ds in enumerate(datasets):
        # datasets = [(val_anno_path, val_img_dir, "COCO_val"),(tr_anno_path, tr_img_dir, "COCO")]
        anno_path = ds[0]
        img_dir = ds[1]
        dataset_type = ds[2]

        coco = COCO(anno_path)
        ids = list(coco.imgs.keys())

        for image_index, img_id in enumerate(ids):
            ann_ids = coco.getAnnIds(imgIds=img_id)
            img_anns = coco.loadAnns(ann_ids)
            image_rec = coco.imgs[img_id]

            img = None
            mask_miss = None
            cached_img_id = None

            for data in process_image(image_rec, img_id, image_index, img_anns, dataset_type):
                # 由process_image中的val_size控制验证集的大小

                if cached_img_id != data['image_id']:
                    assert img_id == data['image_id']
                    cached_img_id = data['image_id']
                    img, mask_miss = make_mask(img_dir, cached_img_id, img_anns, coco)

                if data['isValidation']:  # 根据 isValidation 标志符确定是否作为val
                    writeImage(val_grp, val_grp_img, data, img, mask_miss, val_write_count, cached_img_id, val_grp_mask)
                    val_write_count += 1
                else:
                    writeImage(tr_grp, tr_grp_img, data, img, mask_miss, tr_write_count, cached_img_id, tr_grp_mask)
                    tr_write_count += 1
    tr_h5.close()
    val_h5.close()
    return tr_write_count, val_write_count


if __name__ == '__main__':
    start_time = time.time()
    tr_sample, val_sample = process()
    end_time = time.time()
    print('************************** \n')
    print('coco mask data process finished! consuming time: %.3f min' % ((end_time - start_time)/60))
    print('the size of train sample is: ', tr_sample)
    print('the size of val sample is: ', val_sample)
    # 大约需要处理30 min
