# coding:utf-8
from configobj import ConfigObj
import numpy as np

# 为了完成demo.py和notebook的运行，从另一个fork的project里拷贝过来的


def config_reader():
    config = ConfigObj('/home/jia/Desktop/Improved-Body-Parts/utils/config')

    param = config['param']  # 继承了dict的一种字典类型
    model_id = param['modelID']
    model = config['models'][model_id]  # 因为config文件中，model部分又有一个[[1]]分支，所以又加上了model_id=1的索引
    model['boxsize'] = int(model['boxsize'])
    model['stride'] = int(model['stride'])
    model['max_downsample'] = int(model['max_downsample'])
    model['padValue'] = int(model['padValue'])
    #param['starting_range'] = float(param['starting_range'])
    #param['ending_range'] = float(param['ending_range'])
    param['remove_recon'] = int(param['remove_recon'])
    param['use_gpu'] = int(param['use_gpu'])
    param['starting_range'] = float(param['starting_range'])
    param['ending_range'] = float(param['ending_range'])
    param['scale_search'] = list(map(float, param['scale_search']))     # [float(param['scale_search'])]
    param['thre1'] = float(param['thre1'])
    param['thre2'] = float(param['thre2'])
    param['thre3'] = float(param['thre3'])
    param['connect_ration'] = float(param['connect_ration'])
    param['connection_tole'] = float(param['connection_tole'])
    param['mid_num'] = int(param['mid_num'])
    param['min_num'] = int(param['min_num'])
    param['len_rate'] = float(param['len_rate'])
    param['offset_radius'] = int(param['offset_radius'])
    param['crop_ratio'] = float(param['crop_ratio'])
    param['bbox_ratio'] = float(param['bbox_ratio'])
    param['GPUdeviceNumber'] = int(param['GPUdeviceNumber'])

    return param, model


if __name__ == "__main__":
    config_reader()
