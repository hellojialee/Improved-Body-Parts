import numpy as np
import matplotlib.pyplot as plt
import torch

def focal_loss(s,sxing):
    s = torch.tensor(s,requires_grad=True)
    sxing = torch.tensor(sxing)
    gamma=2.0
    st=torch.where(torch.ge(sxing, 0.01), s, 1-s)
    factor = torch.pow(1. - st, gamma)
    print('the factor is \n', factor)
    out = torch.mul((s-sxing), (s-sxing)) * factor
    out.backward()
    print(s.grad)


focal_loss(0.2, 0.85)
