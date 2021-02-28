#Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

#This program is free software; you can redistribute it and/or modify it under the terms of the BSD 0-Clause License.

#This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD 0-Clause License for more details.

import torch.nn as nn
import torch
import math
from torch.autograd import Variable
from torch.nn import init
from core.utils import batch_logdet2x2, uv_to_rgb_torch
from modules.bivariate_von_mises import BivariateVonMises
from modules.fft_vec import fft2vec
from modules.vec_fft import Vec2Fft
import torch.nn.functional as F

class Ffcc(nn.Module):

    def __init__(self, conf, pretrained=False, local_absolute_deviation=True,
                gain=False, bivariate_von_mises_epsilon=1,
                mult_F=-22.5, add_F=-43, mult_B=-23, add_B=-39.25,
                mult_G=-22.5, add_G=-43, mult_F_lab=-25, add_F_lab=-40.75):
        # all parameters are 2**x because Matlab FFCC does the same
        bivariate_von_mises_epsilon = 2**bivariate_von_mises_epsilon
        mult_F = 2**mult_F
        add_F = 2**add_F
        mult_B = 2**mult_B
        add_B = 2**add_B
        mult_G = 2**mult_G
        add_G = 2**add_G
        mult_F_lab = 2**mult_F_lab
        add_F_lab = 2**add_F_lab

        arch = conf['network']['subarch']
        # get FFCC histograms settings
        conf_hist = conf['log_uv_warp_histogram']
        self._num_bins = conf_hist['num_bins'] # numbers of bins (num_bins*num_bins image)
        self._bin_size = conf_hist['bin_size'] # size of the bin in log [g/r, g/b] space
        self._starting_uv = conf_hist['starting_uv'] # starting point of the histogram
        # this setting is to emulate inverse UV of Hisilicon's Matlab FFCC:
        # instead of g/r and g/b, they use r/g and b/g
        self._inverse_uv = 'inverse_uv' in conf_hist and conf_hist['inverse_uv']
        if arch == 'ffcc':
            super(Ffcc, self).__init__()
        else:
            raise Exception('Wrong architecture')

        # no pretrained model
        if pretrained:
            raise Exception('Wrong architecture')

        self._gain = gain
        self._local_absolute_deviation = local_absolute_deviation
        self._bivariate_von_mises_epsilon = bivariate_von_mises_epsilon

        # compute minimum entropy: taking into account epsilon
        # (they add epsilon*Identity matrix) to the Sigma matrix (see Eq. 17)
        sqrt_det_identity_epsilon = self._bin_size*self._bin_size*self._bivariate_von_mises_epsilon
        self._min_entropy = 0.5*math.log(sqrt_det_identity_epsilon*sqrt_det_identity_epsilon)

        # Bivariate von Mises
        self.bivariate_von_mises = BivariateVonMises(self._num_bins, bivariate_von_mises_epsilon)

        # F are our convolution weights
        F_data = torch.zeros(1, self._num_bins * self._num_bins)
        self.F = nn.Parameter(F_data)
        if self._local_absolute_deviation:
            # F_lab are the convolution weights for the filtered image (Eq. 19)
            F_lab_data = torch.zeros(1, self._num_bins * self._num_bins)
            self.F_lab = nn.Parameter(F_lab_data)
        self.softmax = nn.Softmax(2)
        # Bias
        self.B = nn.Parameter(torch.Tensor(1, self._num_bins * self._num_bins))
        # Gain
        if self._gain:
            self.G = nn.Parameter(torch.Tensor(1, self._num_bins * self._num_bins))

        # see FFCC supp. material for this
        self.fft_vec = fft2vec
        self.vec_fft = Vec2Fft.apply

        # regularizer
        # We call cuda() here to ensure we use the same implementation of
        # fft() and ifft(), CPU and GPU version could be slightly different
        u_variation = torch.zeros(1, self._num_bins, self._num_bins)
        u_variation[0, 0, 0] = -1.0 / math.sqrt(math.sqrt(self._num_bins))
        u_variation[0, 1, 0] = 1.0 / math.sqrt(math.sqrt(self._num_bins))
        if conf['use_gpu']:
            u_variation = u_variation.cuda()
        u_variation = torch.rfft(u_variation, 2, onesided=False)
        u_variation = u_variation[0, :, :, 0]*u_variation[0, :, :, 0] + u_variation[0, :, :, 1]*u_variation[0, :, :, 1]

        v_variation = torch.zeros(1, self._num_bins, self._num_bins)
        v_variation[0, 0, 0] = -1.0 / math.sqrt(math.sqrt(self._num_bins))
        v_variation[0, 0, 1] = 1.0 / math.sqrt(math.sqrt(self._num_bins))
        if conf['use_gpu']:
            v_variation = v_variation.cuda()
        v_variation = torch.rfft(v_variation, 2, onesided=False)
        v_variation = v_variation[0, :, :, 0]*v_variation[0, :, :, 0] + v_variation[0, :, :, 1]*v_variation[0, :, :, 1]

        total_variation = (u_variation + v_variation).view(1, self._num_bins, self._num_bins, 1)
        total_variation = torch.cat((total_variation, total_variation), -1)

        # preconditioners
        preconditioner_F = self._compute_preconditioner(total_variation, mult_F, add_F)
        self.register_buffer('preconditioner_F', preconditioner_F)
        preconditioner_B = self._compute_preconditioner(total_variation, mult_B, add_B)
        self.register_buffer('preconditioner_B', preconditioner_B)
        if self._gain:
            preconditioner_G = self._compute_preconditioner(total_variation, mult_G, add_G)
            self.register_buffer('preconditioner_G', preconditioner_G)
        if self._local_absolute_deviation:
            preconditioner_F_lab = self._compute_preconditioner(total_variation, mult_F_lab, add_F_lab)
            self.register_buffer('preconditioner_F_lab', preconditioner_F_lab)

        # we init all to zero!
        init.constant_(self.F, 0)
        if self._local_absolute_deviation:
            init.constant_(self.F_lab, 0)

        init.constant_(self.B, 0)
        if self._gain:
            init.constant_(self.G, 0)

    def initialize(self, illuminnts = None):
        pass

    def _compute_preconditioner(self, total_variation, mult, add):
        # see FFCC supp. material
        #out = torch.sqrt(1.0 / self.fft_vec(total_variation*mult + add, False))
        regularizer = total_variation*mult + add
        out = torch.sqrt(math.sqrt(2) / self.fft_vec(regularizer, True))
        # We need to special-case the 4 FFT elements that are present only once.
        n = total_variation.shape[1]
        hn = n // 2

        out[:, 0] = out[:, 0] / math.sqrt(math.sqrt(2))
        out[:, hn] = out[:, hn] / math.sqrt(math.sqrt(2))
        out[:, hn+1] = out[:, hn+1] / math.sqrt(math.sqrt(2))
        out[:, n+1] = out[:, n+1] / math.sqrt(math.sqrt(2))

        return out

    def complex_mul(self, a, b):
        # Complex number multiplication
        # pytorch didn't support complex numbers when this was implemented
        real_a = a[:, :, :, 0]
        imag_a = a[:, :, :, 1]
        real_b = b[:, :, :, 0]
        imag_b = b[:, :, :, 1]

        real = (real_a*real_b - imag_a*imag_b).unsqueeze(-1)
        imag = (real_a*imag_b + imag_a*real_b).unsqueeze(-1)

        mul = torch.cat((real, imag), 3)

        return mul

    def forward(self, data):
        # get histogram for image
        im_hist = data['log_uv_histogram_wrapped']
        if self._local_absolute_deviation:
            # and histogram for filtered image
            lab_hist = data['log_uv_histogram_wrapped_local_abs_dev']

        im_hist = im_hist.squeeze(1)
        lab_hist = lab_hist.squeeze(1)

        # 1. do fft
        x_im = torch.rfft(im_hist, 2, onesided=False)
        if self._local_absolute_deviation:
            x_lab = torch.rfft(lab_hist, 2, onesided=False)

        # get all weights, F, F_lab, B and G
        F = self.F
        B = self.B
        if self._local_absolute_deviation:
            F_lab = self.F_lab
        if self._gain:
            G = self.G

        # 2. convolution
        F_fft = self.vec_fft(self.F*self.preconditioner_F)
        # complex multiplication!
        x_im = self.complex_mul(F_fft, x_im)
        if self._local_absolute_deviation:
            # convolution for filtered image histogram
            F_lab_fft = self.vec_fft(F_lab*self.preconditioner_F_lab)
            # complex multiplication!
            x_lab = self.complex_mul(F_lab_fft, x_lab)
            # Add both activations
            x = x_im + x_lab
        else:
            x = x_im

        # 3. inverse fft
        x = torch.irfft(x, 2, onesided=False)
        x = x.unsqueeze(1)

        # 4. affine transformation
        # convert B and G to real numbers
        B_real = torch.irfft(self.vec_fft(B*self.preconditioner_B), 2, onesided=False).unsqueeze(1)
        if self._gain:
            G_real = torch.exp(torch.irfft(self.vec_fft(G*self.preconditioner_G), 2, onesided=False).unsqueeze(1))
            # aff. trans.
            x = torch.mul(x, G_real) + B_real
        else:
            # aff. trans. (G=1)
            x = x + B_real

        # 5. softmax
        bin_probability_logits = x
        bin_probability = self.softmax(x.view(*x.size()[:2], -1)).view_as(x) # b, 1, u, v

        # 6. fit bivariate von mises
        bvm_out = self.bivariate_von_mises(bin_probability)
        sigma_idx = bvm_out['sigma']
        mu_idx = bvm_out['mu']

        # convert from ids to log uv
        mu = self._starting_uv + self._bin_size*mu_idx
        bin_size_squared = self._bin_size*self._bin_size
        sigma = torch.mul(sigma_idx, bin_size_squared)

        # 7. compute the confidence
        entropy = 0.5*batch_logdet2x2(sigma)
        confidence = torch.exp(self._min_entropy - entropy)

        # 8. convert mean log uv to rgb
        illuminant = uv_to_rgb_torch(mu, self._inverse_uv)

        # save output to dictionary
        d = {'illuminant': illuminant, 'bin_probability': bin_probability, 'mu_idx': mu_idx,
            'mu': mu, 'sigma': sigma, 'bias': B_real.squeeze(1), 'confidence': confidence}
        d['F_visualization'] = torch.irfft(F_fft, 2, onesided=False)
        d['bin_probability_logits'] = bin_probability_logits
        d['bivariate_von_mises_epsilon'] = self._bivariate_von_mises_epsilon
        if self._local_absolute_deviation:
            d['F_lab_visualization'] = torch.irfft(F_lab_fft, 2, onesided=False)
        if self._gain:
            d['gain'] = G_real.squeeze(1)

        return d
