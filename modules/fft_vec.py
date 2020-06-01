import torch
import torch.nn as nn
import numpy as np
import math
from torch.autograd import Variable

def fft2vec(x_fft, apply_scaling):
    real = x_fft[:, :, :, 0].clone()
    imag = x_fft[:, :, :, 1].clone()

    n = x_fft.shape[1]
    hn = n // 2

    # apply scaling
    if apply_scaling:
        real[:, 0, 0] = real[:, 0, 0] / math.sqrt(2)
        real[:, 0, hn] = real[:, 0, hn] / math.sqrt(2)
        real[:, hn, 0] = real[:, hn, 0] / math.sqrt(2)
        real[:, hn, hn] = real[:, hn, hn] / math.sqrt(2)

    bs = x_fft.shape[0]

    out = torch.cat((real[:, 0:hn+1, 0], real[:, 0:hn+1, hn], real[:, :, 1:hn].transpose(1,2).contiguous().view(bs, -1),
                        imag[:, 1:hn, 0], imag[:, 1:hn, hn], imag[:, :, 1:hn].transpose(1,2).contiguous().view(bs, -1)), 1)

    if apply_scaling:
        out = out * math.sqrt(2)

    return out

if __name__ == '__main__':
    tensor = torch.randn(2, 4, 4, 2).cuda()
    tensor_fft = torch.fft(tensor, 2)
    tensor_vec = fft2vec(tensor_fft)
