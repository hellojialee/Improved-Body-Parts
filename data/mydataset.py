# coding: utf-8
""" MSCOCO Pytorch Dataset Class"""

from torch.utils.data import Dataset
from py_cocodata_server.py_data_iterator import RawDataIterator
import numpy as np
from config.config import GetConfig, COCOSourceConfig
from time import time
import matplotlib.pyplot as plt
import cv2
import torch.nn.functional as F
import torch


class MyDataset(Dataset):
    def __init__(self, global_config, config, shuffle=False, augment=True):
        """
        Initialize a DataIterator
        :param global_config: the configuration used in our project
        :param config:  the original COCO configuration
        :param shuffle:
        :param augment:
        """
        self.global_config = global_config
        self.config = config
        self.shuffle = shuffle
        self.augment = augment
        self.raw_data_iterator = RawDataIterator(self.global_config, self.config, shuffle=self.shuffle,
                                                 augment=self.augment)

    def __getitem__(self, index):
        # return entries: image, mask_miss, unmasked labels, offsets, mask_offset
        # Notice： numpy.random seed will fork the same value in multi-process, while python random will fork differently
        return self.raw_data_iterator.gen(index)

    def __len__(self):
        return self.raw_data_iterator.num_keys()


if __name__ == '__main__':  # for debug

    def test_augmentation_speed(train_client, show_image=True):
        start = time()
        batch = 0
        for index in range(train_client.__len__()):
            batch += 1

            image, mask_miss, labels = [v.numpy() for v in    # , offsets, mask_offset
                                            train_client.__getitem__(index)]

            # show the generated ground truth
            if show_image:
                show_labels = cv2.resize(labels.transpose((1, 2, 0)), image.shape[:2], interpolation=cv2.INTER_CUBIC)
                # offsets = cv2.resize(offsets.transpose((1, 2, 0)), image.shape[:2], interpolation=cv2.INTER_NEAREST)
                mask_miss = np.repeat(mask_miss.transpose((1, 2, 0)), 3, axis=2)
                # mask_miss = cv2.resize(mask_miss, image.shape[:2], interpolation=cv2.INTER_NEAREST)
                image = cv2.resize(image, mask_miss.shape[:2], interpolation=cv2.INTER_NEAREST)
                plt.imshow(image[:, :, [2, 1, 0]])   # Opencv image format: BGR
                plt.imshow(labels.transpose((1, 2, 0))[:,:,20], alpha=0.5)  # mask_all
                # plt.imshow(show_labels[:, :, 3], alpha=0.5)  # mask_all
                plt.show()
                t=2
        print("produce %d samples per second: " % (batch / (time() - start)))

    config = GetConfig("Canonical")
    soureconfig = COCOSourceConfig("../data/dataset/coco/link2coco2017/coco_val_dataset384.h5")

    val_client = MyDataset(config, soureconfig, shuffle=True, augment=True)  # shuffle in data loader
    # test the data generator
    test_augmentation_speed(val_client, True)
