import time
import torch
import torch.nn
import torch.nn as nn
import torch.nn.functional as F


def focal_l2_loss(s, sxing):
    st = torch.where(torch.ge(sxing, 0.01), s, 1 - s)
    factor = (1. - st) ** 2
    out = (s - sxing) ** 2 * factor
    return out




