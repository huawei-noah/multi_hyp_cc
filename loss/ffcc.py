# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved. THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.

import torch
import math
import torch.nn as nn
from torch.autograd import Variable
from core.utils import *
from numpy.linalg import norm
import torch.nn.functional as F

# google FFCC loss
class Ffcc(nn.Module):
    def __init__(self, conf, logistic_loss_epochs,
                logistic_loss_mult=2.5, bvm_mult=2.5,
                regularization_mult=0.5):
        logistic_loss_mult = 2**logistic_loss_mult
        bvm_mult = 2**bvm_mult

        super(Ffcc, self).__init__()
        self._conf = conf
        self._bin_size = self._conf['log_uv_warp_histogram']['bin_size']

        self._logistic_loss_epochs = logistic_loss_epochs
        self._logistic_loss_mult = logistic_loss_mult
        self._bvm_mult = bvm_mult
        self._regularization_mult = regularization_mult

    def forward(self, outputs, data, model):
        labels = Variable(data['illuminant_log_uv'], requires_grad=False)
        mu = outputs['mu']
        sigma = outputs['sigma']

        regularization_term = 0
        for name, param in model.named_parameters():
            if 'conv' not in name:
                regularization_term += (param*param).sum()

        # they actually use 2 losses, logistic regression for some epochs,
        # then, BVM
        if data['epoch'] < self._logistic_loss_epochs:
            # logistic loss
            gt_pdf = data['gt_pdf']
            bin_probability_logits = outputs['bin_probability_logits'].squeeze(1)
            logsoft = F.log_softmax(bin_probability_logits.view(bin_probability_logits.shape[0], -1), 1).view_as(bin_probability_logits)
            logistic_loss_positive = (gt_pdf*logsoft).view(bin_probability_logits.shape[0], -1).sum(1)
            data_term = -self._logistic_loss_mult*logistic_loss_positive.mean()
        else:
            # bivariate von mises
            dif = (labels - mu).unsqueeze(-1)

            sigma_inv = torch.inverse(sigma)
            fitting_loss = torch.sum(torch.mul(torch.matmul(sigma_inv, dif), dif).squeeze(-1), 1)
            logdet = batch_logdet2x2(sigma)
            loss_bvm = 0.5*(fitting_loss + logdet + 2*math.log(2*math.pi))
            loss_bvm_min = math.log(2*math.pi*outputs['bivariate_von_mises_epsilon']*self._bin_size*self._bin_size)
            l = loss_bvm - loss_bvm_min
            data_term = self._bvm_mult*l.mean()

        return data_term + self._regularization_mult*regularization_term
