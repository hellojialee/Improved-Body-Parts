import sys
sys.path.append("..")  # 包含上级目录
import json
import time
import numpy as np
from itertools import product
import torch
import torch.nn.functional as F
from data.mydataset import MyDataset
from torch.utils.data import DataLoader
from utils.config_reader import config_reader
from utils import util
from config.config import GetConfig, COCOSourceConfig, TrainingOpt
import matplotlib.pyplot as plt
from models.posenet import NetworkEval
import warnings
import os
import argparse


os.environ['CUDA_VISIBLE_DEVICES'] = "3"  # choose the available GPUs
warnings.filterwarnings("ignore")


parser = argparse.ArgumentParser(description='PoseNet Training')
parser.add_argument('--resume', '-r', action='store_true', default=True, help='resume from checkpoint')
parser.add_argument('--checkpoint_path', '-p',  default='checkpoints_parallel', help='save path')
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
soureconfig = COCOSourceConfig(opt.hdf5_train_data)
train_data = MyDataset(config, soureconfig, shuffle=False, augment=True)  # shuffle in data loader
train_loader = DataLoader(train_data, batch_size=opt.batch_size, shuffle=True, drop_last=True, num_workers=16,
                          pin_memory=True)  # num_workers is tuned according to project, too big or small is not good.

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
posenet.eval()   # set eval mode is important

params, model_params = config_reader()

batch_time = AverageMeter()
end = time.time()

with torch.no_grad():  # will save gpu memory and speed up
    for batch_idx, target_tuple in enumerate(train_loader):
        # images.requires_grad_()
        # loc_targets.requires_grad_()
        # conf_targets.requires_grad_()
        target_tuple = [target_tensor.cuda(non_blocking=True) for target_tensor in target_tuple]

        # target tensor shape: [8,512,512,3], [8, 1, 128,128], [8,43,128,128], [8,36,128,128], [8,36,128,128]
        images, mask_misses, heatmaps = target_tuple  # , offsets, mask_offsets

        # images = Variable(images)
        # loc_targets = Variable(loc_targets)
        # conf_targets = Variable(conf_targets)

        output_tuple = posenet(images)

        torch.cuda.synchronize()  # 因为所有GPU操作是异步的，应等待当前设备上所有流中的所有核心完成，测试的时间才正确
        batch_time.update((time.time() - end))
        end = time.time()
        print('==================>Test: [{0}/{1}]\t'
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'Speed {2:.3f} ({3:.3f})\t'.format(
            batch_idx, len(train_loader),
            opt.batch_size / batch_time.val,
            opt.batch_size / batch_time.avg,
            batch_time=batch_time))

# ==================>Test: [651/7497]	Time 0.435 (0.445)	Speed 36.740 (35.933)
# ==================>Test: [652/7497]	Time 0.437 (0.445)	Speed 36.591 (35.934)
# ==================>Test: [653/7497]	Time 0.439 (0.445)	Speed 36.410 (35.935)
#  .....                                                        Speed 38.5
