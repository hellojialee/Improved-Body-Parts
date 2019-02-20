import os
import argparse
import time
import torch.cuda
import torch.nn as nn
import torch.optim as optim
from config.config import GetConfig, COCOSourceConfig, TrainingOpt
from data.mydataset import MyDataset
from torch.utils.data import DataLoader
from models.posenet import PoseNet


parser = argparse.ArgumentParser(description='SSD Training')
parser.add_argument('--resume', '-r', action='store_true', default=False, help='resume from checkpoint')
args = parser.parse_args()


opt = TrainingOpt()
config = GetConfig(opt.config_name)
soureconfig = COCOSourceConfig(opt.hdf5_val_data)
val_data = MyDataset(config, soureconfig, shuffle=False, augment=True)  # shuffle in data loader
val_loader = DataLoader(val_data, batch_size=opt.batch_size, shuffle=True, drop_last=True, num_workers=4,
                        pin_memory=True)


use_cuda = torch.cuda.is_available()  # 判断GPU cuda是否可用
best_loss = float('inf')
start_epoch = 0  # 从0开始或者从上一个epoch开始

posenet = PoseNet(opt.hourglass_order, opt.hourglass_inp_dim, config.num_layers + config.offset_layers)

if args.resume:
    print(' # Resuming from checkpoint # ')
    checkpoint = torch.load(opt.ckpt_path)
    posenet.load_state_dict(checkpoint['net'])
    best_loss = checkpoint['loss']
    start_epoch = checkpoint['epoch']


criterion = nn.MSELoss()

if use_cuda:
    posenet.cuda()
    criterion.cuda()

    torch.backends.cudnn.benchmark = True  # 如果我们每次训练的输入数据的size不变，那么开启这个就会加快我们的训练速度

optimizer = optim.SGD(posenet.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)










