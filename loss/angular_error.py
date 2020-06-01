import torch
import torch.nn as nn
from torch.autograd import Variable
from core.utils import *

class AngularError(nn.Module):
    def __init__(self, conf, compute_acos, illuminant_key = 'illuminant',
                gt_key = 'illuminant'):
        super(AngularError, self).__init__()
        self._conf = conf
        self._illuminant_key = illuminant_key
        self._gt_key = gt_key
        self._compute_acos = compute_acos

    def forward(self, outputs, data, model):
        labels = Variable(data[self._gt_key])
        pred = outputs[self._illuminant_key]

        # angular_error_gradsafe computes differentiable angular error,
        # arccos(x) is not differentiable at -1 and +1. We handle that,
        # as well as 0 vector.
        err = angular_error_gradsafe(pred, labels, compute_acos=self._compute_acos)

        return err.mean()
