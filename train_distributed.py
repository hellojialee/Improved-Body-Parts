import os
import argparse
import time
import tqdm
import cv2
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from config.config import GetConfig, COCOSourceConfig, TrainingOpt
from data.mydataset import MyDataset
from torch.utils.data import DataLoader
from models.posenet import Network
from models.loss_model import MultiTaskLoss
import warnings

try:
    from apex.parallel import DistributedDataParallel as DDP
    from apex.fp16_utils import *
    from apex import amp
    from apex.multi_tensor_apply import multi_tensor_applier
except ImportError:
    raise ImportError("Please install apex from https://www.github.com/nvidia/apex to run this example.")


warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='PoseNet Training')
parser.add_argument('--resume', '-r', action='store_true', default=True, help='resume from checkpoint')
parser.add_argument('--checkpoint_path', '-p',  default='checkpoints_distributed', help='save path')
parser.add_argument('--max_grad_norm', default=5, type=float,
                    help=("If the norm of the gradient vector exceeds this, "
                          "re-normalize it to have the norm equal to max_grad_norm"))
# FOR DISTRIBUTED:  Parse for the local_rank argument, which will be supplied automatically by torch.distributed.launch.
parser.add_argument("--local_rank", default=0, type=int)
parser.add_argument('--opt-level', type=str, default='O1')
parser.add_argument('--sync_bn', action='store_true', help='enabling apex sync BN.')  # 无触发为false， -c 触发为true
parser.add_argument('--keep-batchnorm-fp32', type=str, default=None)
parser.add_argument('--loss-scale', type=str, default=None)
parser.add_argument('--print-freq', '-f', default=10, type=int, metavar='N', help='print frequency (default: 10)')

torch.backends.cudnn.benchmark = True  # 如果我们每次训练的输入数据的size不变，那么开启这个就会加快我们的训练速度
use_cuda = torch.cuda.is_available()

args = parser.parse_args()

checkpoint_path = args.checkpoint_path
opt = TrainingOpt()
config = GetConfig(opt.config_name)
soureconfig = COCOSourceConfig(opt.hdf5_train_data)
train_data = MyDataset(config, soureconfig, shuffle=False, augment=True)  # shuffle in data loader

soureconfig_val = COCOSourceConfig(opt.hdf5_val_data)
val_data = MyDataset(config, soureconfig_val, shuffle=False, augment=True)  # shuffle in data loader


best_loss = float('inf')
start_epoch = 0  # 从0开始或者从上一个epoch开始

args.distributed = False
if 'WORLD_SIZE' in os.environ:
    args.distributed = int(os.environ['WORLD_SIZE']) > 1

args.gpu = 0
args.world_size = 1

# FOR DISTRIBUTED:  If we are running under torch.distributed.launch,
# the 'WORLD_SIZE' environment variable will also be set automatically.
if args.distributed:
    args.gpu = args.local_rank
    torch.cuda.set_device(args.gpu)
    # Initializes the distributed backend which will take care of synchronizing nodes/GPUs
    torch.distributed.init_process_group(backend='nccl', init_method='env://')
    args.world_size = torch.distributed.get_world_size()  # 获取分布式训练的进程数

assert torch.backends.cudnn.enabled, "Amp requires cudnn backend to be enabled."

posenet = Network(opt, config, dist=True)
# Actual working batch size on multi-GPUs is 4 times bigger than that on one GPU
# fixme: add up momentum if the batch grows?
optimizer = optim.SGD(posenet.parameters(), lr=opt.learning_rate * args.world_size, momentum=0.9, weight_decay=1e-4)
# 设置学习率下降策略, extract the "bare"  Pytorch optimizer before Apex wrapping.
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.2, last_epoch=-1)


if args.sync_bn:  # 用累计loss来达到sync bn 是不是更好，更改bn的momentum大小
    #  This should be done before model = DDP(model, delay_allreduce=True),
    #  because DDP needs to see the finalized model parameters
    import apex
    print("Using apex synced BN.")
    posenet = apex.parallel.convert_syncbn_model(posenet)

posenet.cuda()

# Initialize Amp.  Amp accepts either values or strings for the optional override arguments,
# for convenient interoperation with argparse.
# For distributed training, wrap the model with apex.parallel.DistributedDataParallel.
# This must be done AFTER the call to amp.initialize.
model, optimizer = amp.initialize(posenet, optimizer,
                                  opt_level=args.opt_level,
                                  keep_batchnorm_fp32=args.keep_batchnorm_fp32,
                                  loss_scale=args.loss_scale)  # Dynamic loss scaling is used by default.
# delay_allreduce delays all communication to the end of the backward pass.


if args.distributed:
    # By default, apex.parallel.DistributedDataParallel overlaps communication with computation in the backward pass.
    # model = DDP(model)
    # delay_allreduce delays all communication to the end of the backward pass.
    # DDP模块同时也计算整体的平均梯度, 这样我们就不需要在训练步骤计算平均梯度。
    model = DDP(model, delay_allreduce=True)


if args.resume:
    # Use a local scope to avoid dangling references
    # dangling references: a variable that refers to an object that was deleted prematurely
    def resume():
        if os.path.isfile(opt.ckpt_path):
            print('Resuming from checkpoint ...... ')
            checkpoint = torch.load(opt.ckpt_path, map_location=torch.device('cpu'))  # map to cpu to save the gpu memory

            # #################################################
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in checkpoint['weights'].items():
                # if 'out' in k or 'merge' in k:
                #     continue
                name = 'module.' + k  # add prefix 'module.'
                new_state_dict[name] = v
            model.load_state_dict(new_state_dict)  # , strict=False
            # # #################################################

            # model.load_state_dict(checkpoint['weights'])  # 加入他人训练的模型，可能需要忽略部分层，则strict=False
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
            global best_loss, start_epoch  # global declaration. otherwise best_loss and start_epoch can not be changed
            best_loss = checkpoint['train_loss']
            print('******************** Best loss resumed is :', checkpoint['train_loss'], '  ************************')
            start_epoch = checkpoint['epoch'] + 1
            print("========> Resume and start training from Epoch {} ".format(start_epoch))
            del checkpoint
        else:
            print("========> No checkpoint found at '{}'".format(opt.ckpt_path))
    resume()


train_sampler = None
val_sampler = None
# Restricts data loading to a subset of the dataset exclusive to the current process
# Create DistributedSampler to handle distributing the dataset across nodes when training 创建分布式采样器来控制训练中节点间的数据分发
# This can only be called after distributed.init_process_group is called 这个只能在 distributed.init_process_group 被调用后调用
# 这个对象控制进入分布式环境的数据集以确保模型不是对同一个子数据集训练，以达到训练目标。
if args.distributed:
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_data)

# 创建数据加载器，在训练和验证步骤中喂数据
train_loader = torch.utils.data.DataLoader(train_data, batch_size=opt.batch_size, shuffle=(train_sampler is None),
                                           num_workers=16, pin_memory=True, sampler=train_sampler, drop_last=True)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=opt.batch_size, shuffle=False,
                                         num_workers=4, pin_memory=True, sampler=val_sampler, drop_last=True)

for param in model.parameters():
    if param.requires_grad:
        print('Parameters of network: Autograd')
        break


#  Update the learning rate for start_epoch times
for i in range(start_epoch):
    scheduler.step()


def train(epoch):
    print('\n ############################# Train phase, Epoch: {} #############################'.format(epoch))
    posenet.train()
    # DistributedSampler 中记录目前的 epoch 数， 因为采样器是根据 epoch 来决定如何打乱分配数据进各个进程
    if args.distributed:
        train_sampler.set_epoch(epoch)
    # train_loss = 0
    scheduler.step()
    print('\nLearning rate at this epoch is: %0.9f\n' % optimizer.param_groups[0]['lr'])  # scheduler.get_lr()[0]

    batch_time = AverageMeter()
    losses = AverageMeter()
    end = time.time()

    for batch_idx, target_tuple in enumerate(train_loader):
        # images.requires_grad_()
        # loc_targets.requires_grad_()
        # conf_targets.requires_grad_()
        if use_cuda:
            #  这允许异步 GPU 复制数据也就是说计算和数据传输可以同时进.
            target_tuple = [target_tensor.cuda(non_blocking=True) for target_tensor in target_tuple]

        # target tensor shape: [8,512,512,3], [8, 1, 128,128], [8,43,128,128], [8,36,128,128], [8,36,128,128]
        images, mask_misses, heatmaps, offsets, mask_offsets = target_tuple
        # images = Variable(images)
        # loc_targets = Variable(loc_targets)
        # conf_targets = Variable(conf_targets)

        loss = model(images, target_tuple[1:])
        optimizer.zero_grad()  # zero the gradient buff

        if loss.item() > 1e6:
            print("\nLoss is abnormal, drop this batch !")
            loss.zero_()
            continue

        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()

        torch.nn.utils.clip_grad_norm(model.parameters(), args.max_grad_norm)
        optimizer.step()  # TODO：可以使用累加的loss变相增大batch size，但对于bn层需要减少默认的momentum

        # train_loss += loss.item()  # 累加的loss !
        # 使用loss += loss.detach()来获取不需要梯度回传的部分。
        # 或者使用loss.item()直接获得所对应的python数据类型，但是仅仅限于only one element tensors can be converted to Python scalars
        if batch_idx % args.print_freq == 0:
            # Every print_freq iterations, check the loss, accuracy, and speed.
            # For best performance, it doesn't make sense to print these metrics every
            # iteration, since they incur an allreduce and some host<->device syncs.
            # print 会触发allreduce，而这个操作比较费时
            if args.distributed:
                # We manually reduce and average the metrics across processes. In-place reduce tensor.
                reduced_loss = reduce_tensor(loss.data)
            else:
                reduced_loss = loss.data

            # to_python_float incurs a host<->device sync
            losses.update(to_python_float(reduced_loss), images.size(0))  # update needs average and number
            torch.cuda.synchronize()  # 因为所有GPU操作是异步的，应等待当前设备上所有流中的所有核心完成，测试的时间才正确
            batch_time.update((time.time() - end) / args.print_freq)
            end = time.time()

            if args.local_rank == 0:  # Print them in the Process 0
                print('==================> Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Speed {3:.3f} ({4:.3f})\t'
                      'Loss {loss.val:.10f} ({loss.avg:.4f}) <================ \t'.format(
                        epoch, batch_idx, len(train_loader),
                        args.world_size * opt.batch_size / batch_time.val,
                        args.world_size * opt.batch_size / batch_time.avg,
                        batch_time=batch_time,
                        loss=losses))

    global best_loss
    # DistributedSampler控制进入分布式环境的数据集以确保模型不是对同一个子数据集训练，以达到训练目标。
    # train_loss /= (len(train_loader))  # Each GPU process can only see 1/(world_size) training samples per epoch

    if args.local_rank == 0:
        # Write the log file each epoch.
        os.makedirs(checkpoint_path, exist_ok=True)
        logger = open(os.path.join('./' + checkpoint_path, 'log'), 'a+')
        logger.write('\nEpoch {}\ttrain_loss: {}'.format(epoch, losses.avg))  # validation时不要\n换行
        logger.flush()
        logger.close()

        if losses.avg < best_loss:
            # Update the best_loss if the average loss drops
            best_loss = losses.avg
            print('Saving model checkpoint...')
            state = {
                # not posenet.state_dict(). then, we don't ge the "module" string to begin with
                'weights': model.module.state_dict(),
                'optimizer_weight': optimizer.state_dict(),
                'train_loss': losses.avg,
                'epoch': epoch
            }
            torch.save(state, './' + checkpoint_path + '/PoseNet_' + str(epoch) + '_epoch.pth')


def test(epoch):
    print('\n ############################# Test phase, Epoch: {} #############################'.format(epoch))
    posenet.eval()
    # DistributedSampler 中记录目前的 epoch 数， 因为采样器是根据 epoch 来决定如何打乱分配数据进各个进程
    if args.distributed:
        train_sampler.set_epoch(epoch)  # 验证集太小，不够4个划分
    batch_time = AverageMeter()
    losses = AverageMeter()
    end = time.time()

    for batch_idx, target_tuple in enumerate(val_loader):
        # images.requires_grad_()
        # loc_targets.requires_grad_()
        # conf_targets.requires_grad_()
        if use_cuda:
            #  这允许异步 GPU 复制数据也就是说计算和数据传输可以同时进.
            target_tuple = [target_tensor.cuda(non_blocking=True) for target_tensor in target_tuple]

        # target tensor shape: [8,512,512,3], [8, 1, 128,128], [8,43,128,128], [8,36,128,128], [8,36,128,128]
        images, mask_misses, heatmaps, offsets, mask_offsets = target_tuple

        with torch.no_grad():
            _, loss = model(images, target_tuple[1:])

        if args.distributed:
            # We manually reduce and average the metrics across processes. In-place reduce tensor.
            reduced_loss = reduce_tensor(loss.data)
        else:
            reduced_loss = loss.data

        # to_python_float incurs a host<->device sync
        losses.update(to_python_float(reduced_loss), images.size(0))  # update needs average and number
        torch.cuda.synchronize()  # 因为所有GPU操作是异步的，应等待当前设备上所有流中的所有核心完成，测试的时间才正确
        batch_time.update((time.time() - end))
        end = time.time()

        if args.local_rank == 0:  # Print them in the Process 0
            print('==================>Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Speed {2:.3f} ({3:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                   batch_idx, len(val_loader),
                   args.world_size * opt.batch_size / batch_time.val,
                   args.world_size * opt.batch_size / batch_time.avg,
                   batch_time=batch_time, loss=losses))

    if args.local_rank == 0:  # Print them in the Process 0
        # Write the log file each epoch.
        os.makedirs(checkpoint_path, exist_ok=True)
        logger = open(os.path.join('./' + checkpoint_path, 'log'), 'a+')
        logger.write('\tval_loss: {}'.format(losses.avg))  # validation时不要\n换行
        logger.flush()
        logger.close()


def adjust_learning_rate(optimizer, epoch, step, len_epoch):
    """LR schedule that should yield 76% converged accuracy with batch size 256"""
    factor = epoch // 30

    if epoch >= 80:
        factor = factor + 1

    lr = args.lr*(0.1**factor)

    """Warmup"""
    if epoch < 5:
        lr = lr*float(1 + step + epoch*len_epoch)/(5.*len_epoch)  # len_epoch=len(train_loader)

    # if(args.local_rank == 0):
    #     print("epoch = {}, step = {}, lr = {}".format(epoch, step, lr))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


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


def reduce_tensor(tensor):
    # Reduces the tensor data across all machines
    # If we print the tensor, we can get:
    # tensor(334.4330, device='cuda:1') *********************, here is cuda:  cuda:1
    # tensor(359.1895, device='cuda:3') *********************, here is cuda:  cuda:3
    # tensor(263.3543, device='cuda:2') *********************, here is cuda:  cuda:2
    # tensor(340.1970, device='cuda:0') *********************, here is cuda:  cuda:0
    rt = tensor.clone()  # The function operates in-place.
    dist.all_reduce(rt, op=dist.reduce_op.SUM)
    rt /= args.world_size
    return rt


if __name__ == '__main__':
    for epoch in range(start_epoch, start_epoch + 200):
        train(epoch)
        test(epoch)

