import time
import torch
from torch import nn
import torch.nn.functional as F


class MultiTaskLoss(nn.Module):

    def __init__(self, opt, config, heatmap_weight=1, offset_weight=1, **kwargs):
        super(MultiTaskLoss, self).__init__()
        self.nstck = opt.nstack
        self.batch_size = opt.batch_size
        self.offset_start = config.offset_start
        self.multi_task_weight = opt.multi_task_weight
        self.scale_weight = opt.scale_weight
        self.nstack_weight = opt.nstack_weight
        self.heatmap_weight = heatmap_weight
        self.offset_weight = offset_weight

    def forward(self, pred_tuple, target_tuple):
        """
        Compute the multi-task total loss
        :param pred_tuple: [nstack * [(bacth,C,128,128), (bacth,C,64,64), (bacth,C,32,32),  (bacth,C,16,16)], (bacth,C,8,8)]
        :param target_tuple: target tensors, i.e.,
         mask_misses,   heatmaps,       offsets,       mask_offsets,
        [batch,1,128,128], [batch,43,128,128], [batch,36,128,128], [batch,36,128,128]
        :return: scalar tensor
        """
        # we use 4 stacks, 5 scales
        # TODO: 是用5个不同scale好还是4个scale监督好？
        assert self.batch_size == target_tuple[0].shape[0], 'batch size {} not match'.format(pred_tuple[0].shape[0])
        pred_scale_tensors = [pred_tuple]
        loss_scales = [self._loss_per_scale(pred_scale_tensors[i], target_tuple) * self.scale_weight[i] for i in
                       range(1)]
        loss_per_batch = sum(loss_scales) / len(self.scale_weight)
        return loss_per_batch

    def _loss_per_scale(self, pred, target):
        """
        Compute the loss on a particular scale.
        :param pred: tensor (nstack, bacth, C, H, W)
        :param target: mask_misses, heatmaps, offsets, mask_offsets of shape (N, C, H, W)
        :return:
        """
        pred_heatmap = pred[:, :, :self.offset_start]
        pred_offset = pred[:, :, self.offset_start:]

        # TODO: Have a try  ------ F.adaptive_avg_pool2d ?  ------
        target_this_scale = [F.adaptive_max_pool2d(iter_tensor, output_size=pred.shape[-2:]) for iter_tensor in target]
        # heatmap = target_this_scale[1][0,...].cpu().numpy().squeeze()
        # mask = target_this_scale[0][0,...].cpu().numpy().squeeze()
        #
        # import matplotlib.pylab as plt
        # plt.imshow(heatmap[11,:,:]) # mask_all
        # # plt.imshow(mask_offset[:, :, 2], alpha=0.5)  # mask_all
        # plt.show()
        heatmap_loss = self.focal_l2_loss(pred_heatmap, target_this_scale[1][None, ...], target_this_scale[0][None, ...]
                                          , nstack_weight=self.nstack_weight)
        offset_loss = self.l1_loss(pred_offset, target_this_scale[2][None, ...], target_this_scale[3][None, ...]
                                   , nstack_weight=self.nstack_weight)
        multi_task_loss = heatmap_loss * self.multi_task_weight[0] + offset_loss * self.multi_task_weight[1]
        return multi_task_loss / 2

    @staticmethod
    def focal_l2_loss(s, sxing, mask_miss, gamma=2, nstack_weight=[1, 1, 1, 1]):
        """
        Compute the focal L2 loss between predicted and groundtruth score maps.
        :param s:  predicted tensor (nstack, batch, channel, height, width), predicted score maps
        :param sxing: target tensor (nstack, batch, channel, height, width)
        :param mask_miss: tensor (nstack, batch, 1, height, width)
        :param gamma: focusing parameter
        :return: a scalar tensor
        """
        eps = 1e-12
        s = torch.clamp(s, eps, 1. - eps)  # improve the stability of the focal loss
        st = torch.where(torch.ge(sxing, 0.01), s, 1 - s)
        factor = (1. - st) ** gamma
        # multiplied by mask_miss via broadcast operation
        out = (s - sxing) ** 2 * mask_miss  # type: torch.Tensor
        # sum over the feature map, should divide by batch afterwards
        loss_nstack = out.sum(dim=(1, 2, 3, 4))
        assert len(loss_nstack) == len(nstack_weight), nstack_weight
        weight_loss = [loss_nstack[i] * nstack_weight[i] for i in range(len(nstack_weight))]
        loss = sum(weight_loss) / len(nstack_weight)
        return loss

    @staticmethod
    def l1_loss(pred, target, mask_offset, nstack_weight=[1, 1, 1, 1]):
        """
        Compute the smooth L1 loss of offset feature maps
        :param pred: predicted tensor (nstack, batch, channel, height, width), predicted feature maps
        :param target: target tensor (nstack, batch, channel, height, width)
        :param mask_offset: tensor (nstack, batch, channel, height, width)
        :param nstack_weight:
        :return:
        """
        out = torch.abs(pred - target) * mask_offset  # type: torch.Tensor
        # sum over the feature map, should divide by batch afterwards
        loss_nstack = out.sum(dim=(1, 2, 3, 4))
        assert len(loss_nstack) == len(nstack_weight), nstack_weight
        weight_loss = [loss_nstack[i] * nstack_weight[i] for i in range(len(nstack_weight))]
        loss = sum(weight_loss) / len(nstack_weight)
        return loss



