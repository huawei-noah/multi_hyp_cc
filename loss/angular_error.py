#Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

#This program is free software; you can redistribute it and/or modify it under the terms of the BSD 0-Clause License.

#This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD 0-Clause License for more details.

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
