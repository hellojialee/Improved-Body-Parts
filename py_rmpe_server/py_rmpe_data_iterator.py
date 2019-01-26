# coding:utf-8
import h5py
import random
import json
import numpy as np
import cv2
from py_rmpe_transformer import Transformer, AugmentSelection
from py_rmpe_heatmapper import Heatmapper
from time import time


class RawDataIterator:
    def __init__(self, global_config, configs, shuffle=True, augment=True):

        self.global_config = global_config

        if not isinstance(configs, (list, tuple)):
            configs = [configs]

        self.h5files = [c.source() for c in configs]
        self.configs = configs
        self.h5s = [h5py.File(fname, "r") for fname in self.h5files]
        self.datums = [
            h5['datum'] if 'datum' in h5 else (h5['dataset'], h5['images'], h5['masks'] if 'masks' in h5 else None) for
            h5 in self.h5s]

        self.heatmapper = Heatmapper(global_config)
        self.transformer = Transformer(global_config)
        self.augment = augment
        self.shuffle = shuffle

        self.keys = []  # coco_mask_hdf5.py生成的dataset中的key包括所有（大约10k个人）sample的信息

        for n, d in enumerate(self.datums):
            if isinstance(d, (list, tuple)):
                k = list(d[0].keys())
            else:
                k = list(d.keys())

            print(len(k))

            self.keys += zip([n] * len(k), k)

    def gen(self, timing=False):
        print('Class类型: py_rmpe_data_iterator.RawDataIterator 被使用作为数据增强和groudtruth的生成')
        # 这个gen()函数是真正生成训练所需的groundtruth数据，并且在ds_generators.py中被调用，在那里数据被复制成多份满足多个stage的输入要求

        if self.shuffle:
            random.shuffle(self.keys)  # shuffle the self.keys

        for num, key in self.keys:

            read_start = time()
            image, mask, meta, debug = self.read_data(num, key)

            aug_start = time()

            # transform picture
            assert mask.dtype == np.uint8, mask.dtype  # verify the dtype of mask
            image, mask, meta = self.transformer.transform(image, mask, meta,
                                                           aug=None if self.augment else AugmentSelection.unrandom())
            assert mask.dtype == np.float, mask.dtype  # 因为在transformer.py中对mask做了立方插值的resize, 且　/225., 所以类型变成了float

            # we need layered mask on next stage
            mask = self.configs[num].convert_mask(mask, self.global_config, joints=meta['joints'])  # mask复制成了57个通道

            # create heatmaps and pafs
            labels = self.heatmapper.create_heatmaps(meta['joints'], mask)

            # normalize image to save gpu/cpu time for keras
            image = image / 256.0 - 0.5  # TODO: data preprocessing -- normalize the image

            if timing:  # TODO: determine whether to output the time of generating groundtruth
                yield image, mask, labels, meta['joints'], time() - read_start, time() - aug_start
            else:
                yield image, mask, labels, meta['joints']

    def num_keys(self):

        return len(self.keys)

    def read_data(self, num, key):

        config = self.configs[num]
        datum = self.datums[num]
        if isinstance(datum, (list, tuple)):
            dataset, images, masks = datum
            return self.read_data_new(dataset, images, masks, key, config)
        else:
            return self.read_data_old(datum, key, config)

    def read_data_old(self, datum, key, config):

        entry = datum[key]

        assert 'meta' in entry.attrs, "No 'meta' attribute in .h5 file. Did you generate .h5 with new code?"

        debug = json.loads(entry.attrs['meta'])
        meta = {}
        meta["objpos"] = debug["objpos"]
        meta["scale_provided"] = debug["scale_provided"]
        meta["joints"] = debug["joints"]

        meta = config.convert(meta, self.global_config)
        data = entry.value

        if data.shape[0] <= 6:
            # TODO: this is extra work, should write in store in correct format (not transposed)
            # can't do now because I want storage compatibility yet
            # fixme: we need image in classical not transposed format in this program for warp affine
            data = data.transpose([1, 2, 0])

        img = data[:, :, 0:3]
        mask_miss = data[:, :, 4]
        # mask = data[:,:,5]

        return img, mask_miss, meta, debug

    def read_data_new(self, dataset, images, masks, key, config):

        entry = dataset[key]

        assert 'meta' in entry.attrs, "No 'meta' attribute in .h5 file. Did you generate .h5 with new code?"

        meta = json.loads(entry.value)
        debug = json.loads(entry.attrs['meta'])
        meta = config.convert(meta, self.global_config)  # 改变数据定义，以满足CMU工作中的要求

        img = images[meta['image']].value
        mask_miss = None

        if len(img.shape) == 2 and img.shape[1] == 1:
            img = cv2.imdecode(img, flags=-1)

        if img.shape[2] > 3:
            mask_miss = img[:, :, 3]
            img = img[:, :, 0:3]

        if mask_miss is None:
            if masks is not None:
                mask_miss = masks[meta['image']].value
                if len(mask_miss.shape) == 2 and mask_miss.shape[1] == 1:
                    mask_miss = cv2.imdecode(mask_miss, flags=-1)

        if mask_miss is None:  # 对于没有mask的image，为了后面计算的形式上能够统一，制造一个全是255的mask，这是为了兼容MPII数据集
            mask_miss = np.full((img.shape[0], img.shape[1]), fill_value=255, dtype=np.uint8)

        return img, mask_miss, meta, debug

    def __del__(self):

        if 'h5s' in vars(self):
            for h5 in self.h5s:
                h5.close()
