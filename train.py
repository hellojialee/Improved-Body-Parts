import os
import argparse
import time
import torch.cuda
import numpy as np
import torch.nn as nn
import torch.optim as optim
from config.config import GetConfig, COCOSourceConfig, TrainingOpt
from data.mydataset import MyDataset
from torch.utils.data import DataLoader
from models.posenet import PoseNet, PoseNet_easy
from models.loss_model import MultiTaskLoss
import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='SSD Training')
parser.add_argument('--resume', '-r', action='store_true', default=False, help='resume from checkpoint')
args = parser.parse_args()

opt = TrainingOpt()
config = GetConfig(opt.config_name)
soureconfig = COCOSourceConfig(opt.hdf5_train_data)
print('start loading the data......')
val_data = MyDataset(config, soureconfig, shuffle=False, augment=True)  # shuffle in data loader
print('loading the data finish......')
val_loader = DataLoader(val_data, batch_size=opt.batch_size, shuffle=True, drop_last=True, num_workers=8,
                        pin_memory=True)  # num_workers is tuned according to project, too big or small is not good.

# # ############# for debug  ###################
# if __name__ == '__main__':
#     t0 = time.time()
#     count = 0
#     print(torch.cuda.get_device_name(0))
#     torch.backends.cudnn.benchmark = True
#     for epoch in range(20):
#         for bath_id, data_dict in enumerate(train_loader):
#
#             t = data_dict[0].cuda()  # , non_blocking=True
#             count += opt.batch_size
#             print(bath_id, ' of ', epoch)
#             if count > 500:
#                 break
#     print('**************** ', count / (time.time() - t0))

use_cuda = torch.cuda.is_available()  # 判断GPU cuda是否可用
best_loss = float('inf')
start_epoch = 0  # 从0开始或者从上一个epoch开始

posenet = PoseNet_easy(opt.nstack, opt.hourglass_inp_dim, config.num_layers + config.offset_layers)

if args.resume:
    print(' # Resuming from checkpoint # ')
    checkpoint = torch.load(opt.ckpt_path)
    posenet.load_state_dict(checkpoint['net'])
    best_loss = checkpoint['loss']
    start_epoch = checkpoint['epoch']

# else:
#     print(' # Loading pretrained model # ')
#     posenet.load_state_dict(torch.load(opt.pretrained_model))

criterion = MultiTaskLoss(opt, config)

if use_cuda:
    posenet.cuda()  # module.cuda() only move the registered parameters to GPU.
    # criterion.cuda()

    torch.backends.cudnn.benchmark = True  # 如果我们每次训练的输入数据的size不变，那么开启这个就会加快我们的训练速度
    # torch.backends.cudnn.deterministic = True

optimizer = optim.SGD(posenet.parameters(), lr=opt.learning_rate, momentum=0.9, weight_decay=1e-4)

for param in posenet.parameters():
    if param.requires_grad == True:
        print('Parameters of network: Autograd')
        break


def train(epoch):
    print('\nTrain phase, Epoch: {}'.format(epoch))
    posenet.train()
    train_loss = 0
    for batch_idx, target_tuple in enumerate(val_loader):
        # images.requires_grad_()
        # loc_targets.requires_grad_()
        # conf_targets.requires_grad_()
        if use_cuda:
            target_tuple = [target_tensor.cuda() for target_tensor in target_tuple]

        # target tensor shape: [8,512,512,3], [8, 1, 128,128], [8,43,128,128], [8,36,128,128], [8,36,128,128]
        images, mask_misses, heatmaps, offsets, mask_offsets = target_tuple
        # images = Variable(images)
        # loc_targets = Variable(loc_targets)
        # conf_targets = Variable(conf_targets)

        optimizer.zero_grad()  # zero the gradient buff

        output_tuple = posenet(images)
        # print(loc_preds.requires_grad)
        # print(conf_preds.requires_grad)
        loss = criterion(output_tuple.permute(1,0,2,3,4), target_tuple[1:])
        # print(loss.requires_grad)
        loss.backward()  # retain_graph=True
        optimizer.step()

        train_loss += loss.item()  # 累加的loss　　　
        # 使用loss += loss.detach()来获取不需要梯度回传的部分。
        # 或者使用loss.item()直接获得所对应的python数据类型。
        print('  Train loss: %.3f, accumulated average loss: %.3f' % (loss.item(), train_loss / (batch_idx + 1)))


if __name__ == '__main__':
    for epoch in range(start_epoch, start_epoch + 2):
        train(epoch)
        torch.save(posenet.state_dict(), 'posemodel.pkl')
