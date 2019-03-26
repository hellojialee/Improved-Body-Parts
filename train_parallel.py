import os
import argparse
import time
import tqdm
import cv2
import matplotlib.pylab as plt
import torch.cuda
import numpy as np
import torch.nn as nn
import torch.optim as optim
from config.config import GetConfig, COCOSourceConfig, TrainingOpt
from data.mydataset import MyDataset
from torch.utils.data import DataLoader
from models.posenet import Network
import warnings


os.environ['CUDA_VISIBLE_DEVICES'] = "0, 1, 2, 3"  # choose the available GPUs
warnings.filterwarnings("ignore")
torch.cuda.empty_cache()

parser = argparse.ArgumentParser(description='PoseNet Training')
parser.add_argument('--resume', '-r', action='store_true', default=True, help='resume from checkpoint')
parser.add_argument('--checkpoint_path', '-p',  default='checkpoints_parallel', help='save path')
parser.add_argument('--max_grad_norm', default=5, type=float,
    help="If the norm of the gradient vector exceeds this, re-normalize it to have the norm equal to max_grad_norm")

args = parser.parse_args()

checkpoint_path = args.checkpoint_path
opt = TrainingOpt()
config = GetConfig(opt.config_name)
soureconfig = COCOSourceConfig(opt.hdf5_train_data)
train_data = MyDataset(config, soureconfig, shuffle=False, augment=True)  # shuffle in data loader
train_loader = DataLoader(train_data, batch_size=opt.batch_size, shuffle=True, drop_last=True, num_workers=16,
                          pin_memory=True)  # num_workers is tuned according to project, too big or small is not good.

soureconfig_val = COCOSourceConfig(opt.hdf5_val_data)
val_data = MyDataset(config, soureconfig_val, shuffle=False, augment=True)  # shuffle in data loader
val_loader = DataLoader(val_data, batch_size=opt.batch_size, shuffle=True, drop_last=True, num_workers=16,
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
#             t = data_dict[0].cuda(non_blocking=True)  # , non_blocking=True
#             count += opt.batch_size
#             print(bath_id, ' of ', epoch)
#             if count > 50:
#                 break
#     print('**************** ', count / (time.time() - t0))

use_cuda = torch.cuda.is_available()  # 判断GPU cuda是否可用
best_loss = float('inf')
start_epoch = 0  # 从0开始或者从上一个epoch开始

posenet = Network(opt, config, dist=False)
optimizer = optim.SGD(posenet.parameters(), lr=opt.learning_rate, momentum=0.9, weight_decay=1e-4)

if args.resume:
    print('Resuming from checkpoint ...... ')
    checkpoint = torch.load(opt.ckpt_path, map_location=torch.device('cpu'))  # map to cpu to save the gpu memory

    # # #################################################
    # from collections import OrderedDict
    # new_state_dict = OrderedDict()
    # for k, v in checkpoint['weights'].items():
    #     if 'out' in k or 'merge' in k:
    #         continue
    #     name = k  # add prefix `posenet.`
    #     new_state_dict[name] = v
    # posenet.load_state_dict(new_state_dict, strict=False)
    # # #################################################

    posenet.load_state_dict(checkpoint['weights'])  # 加入他人训练的模型，可能需要忽略部分层，则strict=False
    print('Network weights have been resumed from checkpoint...')

    optimizer.load_state_dict(checkpoint['optimizer_weight'])
    # We must convert the resumed state data of optimizer to gpu
    """It is because the previous training was done on gpu, so when saving the optimizer.state_dict, the stored
     states(tensors) are of cuda version. During resuming, when we load the saved optimizer, load_state_dict()
     loads this cuda version to cpu. But in this project, we use map_location to map the state tensors to cpu.
     In the training process, we need cuda version of state tensors, so we have to convert them to gpu."""
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.cuda()
    print('Optimizer has been resumed from checkpoint...')
    best_loss = checkpoint['train_loss']
    print('******************** Best loss resumed is :', best_loss, '  ************************')
    start_epoch = checkpoint['epoch'] + 1
    del checkpoint
torch.cuda.empty_cache()


if use_cuda:
    posenet = torch.nn.parallel.DataParallel(posenet.cuda())   # , device_ids=[0, 1, 2, 3]
    # module.cuda() only move the registered parameters to GPU.
    torch.backends.cudnn.benchmark = True  # 如果我们每次训练的输入数据的size不变，那么开启这个就会加快我们的训练速度
    # torch.backends.cudnn.deterministic = True

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1, last_epoch=-1)     # 设置学习率下降策略
for i in range(start_epoch):
    #  update the learning rate for start_epoch times
    scheduler.step()

for param in posenet.parameters():
    if param.requires_grad:
        print('Parameters of network: Autograd')
        break


def train(epoch):
    print('\n ############################# Train phase, Epoch: {} #############################'.format(epoch))
    posenet.train()
    train_loss = 0
    scheduler.step()
    print('\nLearning rate at this epoch is: %0.9f\n' % optimizer.param_groups[0]['lr'])  # scheduler.get_lr()[0]

    for batch_idx, target_tuple in enumerate(train_loader):
        # images.requires_grad_()
        # loc_targets.requires_grad_()
        # conf_targets.requires_grad_()
        if use_cuda:
            target_tuple = [target_tensor.cuda(non_blocking=True) for target_tensor in target_tuple]

        # target tensor shape: [8,512,512,3], [8, 1, 128,128], [8,43,128,128], [8,36,128,128], [8,36,128,128]
        images, mask_misses, heatmaps = target_tuple  # , offsets, mask_offsets
        # images = Variable(images)
        # loc_targets = Variable(loc_targets)
        # conf_targets = Variable(conf_targets)

        optimizer.zero_grad()  # zero the gradient buff

        loss_ngpu = posenet(images, target_tuple[1:])  # reduce losses of all GPUs on cuda 0
        loss = torch.sum(loss_ngpu) / opt.batch_size
        # print(loc_preds.requires_grad)
        # print(conf_preds.requires_grad)
        if loss.item() > 1e6:
            print("\nLoss is abnormal, drop this batch !")
            loss.zero_()
            continue
        # print(loss.requires_grad)
        loss.backward()  # retain_graph=True
        torch.nn.utils.clip_grad_norm(posenet.parameters(), args.max_grad_norm)
        optimizer.step()  # TODO：可以使用累加的loss变相增大batch size，但对于bn层需要减少默认的momentum

        train_loss += loss.item()  # 累加的loss !
        # 使用loss += loss.detach()来获取不需要梯度回传的部分。
        # 或者使用loss.item()直接获得所对应的python数据类型，但是仅仅限于only one element tensors can be converted to Python scalars
        print('########################### Epoch:', epoch, ', --  batch:',  batch_idx, '/', len(train_loader), ',   ',
              'Train loss: %.3f, accumulated average loss: %.3f ##############################' % (loss.item(), train_loss / (batch_idx + 1)))

    global best_loss
    train_loss /= len(train_loader)

    os.makedirs(checkpoint_path, exist_ok=True)
    logger = open(os.path.join('./' + checkpoint_path, 'log'), 'a+')
    logger.write('\nEpoch {}\ttrain_loss: {}'.format(epoch, train_loss))  # validation时不要\n换行
    logger.flush()
    logger.close()
    if train_loss < best_loss:
        best_loss = train_loss
        print('=====> Saving checkpoint...')
        state = {
            # not posenet.state_dict(). then, we don't ge the "module" string to begin with
            'weights': posenet.module.state_dict(),
            'optimizer_weight': optimizer.state_dict(),
            'train_loss': train_loss,
            'epoch': epoch,
        }
        torch.save(state, './' + checkpoint_path + '/PoseNet_' + str(epoch) + '_epoch.pth')


def test(epoch, show_image=False):
    print('\nTest phase, Epoch: {}'.format(epoch))
    posenet.eval()
    with torch.no_grad():  # will save gpu memory and speed up
        test_loss = 0
        for batch_idx, target_tuple in enumerate(val_loader):
            # images.requires_grad_()
            # loc_targets.requires_grad_()
            # conf_targets.requires_grad_()
            if use_cuda:
                target_tuple = [target_tensor.cuda(non_blocking=True) for target_tensor in target_tuple]

            # target tensor shape: [8,512,512,3], [8, 1, 128,128], [8,43,128,128], [8,36,128,128], [8,36,128,128]
            images, mask_misses, heatmaps = target_tuple  # , offsets, mask_offsets
            # images = Variable(images)
            # loc_targets = Variable(loc_targets)
            # conf_targets = Variable(conf_targets)

            output_tuple, loss_ngpu = posenet(images, target_tuple[1:])
            loss = torch.sum(loss_ngpu) / opt.batch_size

            test_loss += loss.item()  # 累加的loss
            print('  Test loss : %.3f, accumulated average loss: %.3f' % (loss.item(), test_loss / (batch_idx + 1)))
            if show_image:
                image, mask_miss, labels = [v.cpu().numpy() for v in target_tuple]  # , offsets, mask_offset
                output = output_tuple[-1][0].cpu().numpy()  # different scales can be shown
                # show the generated ground truth
                img = image[0]
                output = output[0].transpose((1, 2, 0))
                img = cv2.resize(img, output.shape[:2], interpolation=cv2.INTER_CUBIC)
                plt.imshow(img[:, :, [2, 1, 0]])  # Opencv image format: BGR
                plt.imshow(output[:, :, 28], alpha=0.5)  # mask_all
                # plt.imshow(mask_offset[:, :, 2], alpha=0.5)  # mask_all
                plt.show()

    os.makedirs(checkpoint_path, exist_ok=True)
    logger = open(os.path.join('./' + checkpoint_path, 'log'), 'a+')
    logger.write('\tval_loss: {}'.format(test_loss / len(val_loader)))  # validation时不要\n换行
    logger.flush()
    logger.close()


if __name__ == '__main__':
    for epoch in range(start_epoch, start_epoch + 200):
        train(epoch)
        test(epoch, show_image=False)

