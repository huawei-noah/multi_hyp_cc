#Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

#This program is free software; you can redistribute it and/or modify it under the terms of the BSD 0-Clause License.

#This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD 0-Clause License for more details.

import torch
import torch.nn as nn
import numpy as np
import math

# FFCC paper, final step: fit a "Gaussian" in a torus
# see section 4: https://storage.googleapis.com/pub-tools-public-publication-data/pdf/6002678db6270c4a69f26d6fcef820e44b134951.pdf
class BivariateVonMises(nn.Module):
    def __init__(self, num_bins, epsilon=1.0):
        super(BivariateVonMises, self).__init__()

        self._num_bins = num_bins
        self._epsilon = epsilon
        theta = torch.Tensor( (2*math.pi*np.arange(num_bins))/num_bins )
        self.register_buffer('theta', theta)

        bins = torch.Tensor( np.arange(num_bins) )
        self.register_buffer('bins', bins)

    def forward(self, bin_probability):
        # Fit approximate bivariate von Mises
        # i -> u, j -> v
        pi = torch.sum(bin_probability, 3, keepdim=True)
        pj = torch.sum(bin_probability, 2, keepdim=True)

        sin_theta = torch.sin(self.theta)
        cos_theta = torch.cos(self.theta)
        sin_theta_pi = sin_theta.unsqueeze(0).unsqueeze(0).unsqueeze(-1).expand_as(pi)
        cos_theta_pi = cos_theta.unsqueeze(0).unsqueeze(0).unsqueeze(-1).expand_as(pi)
        sin_theta_pj = sin_theta.unsqueeze(0).unsqueeze(0).unsqueeze(0).expand_as(pj)
        cos_theta_pj = cos_theta.unsqueeze(0).unsqueeze(0).unsqueeze(0).expand_as(pj)

        yi = torch.sum(torch.mul(pi, sin_theta_pi), 2, keepdim=True)
        xi = torch.sum(torch.mul(pi, cos_theta_pi), 2, keepdim=True)

        yj = torch.sum(torch.mul(pj, sin_theta_pj), 3, keepdim=True)
        xj = torch.sum(torch.mul(pj, cos_theta_pj), 3, keepdim=True)

        # 1. compute the mean (with gray light de-alisaing)
        m_u_idx = torch.remainder(torch.mul(torch.atan2(yi, xi), self._num_bins)/(2*math.pi), self._num_bins).squeeze(-1).squeeze(-1)
        m_v_idx = torch.remainder(torch.mul(torch.atan2(yj, xj), self._num_bins)/(2*math.pi), self._num_bins).squeeze(-1).squeeze(-1)
        mu_idx = torch.cat([m_u_idx, m_v_idx], 1)

        # 2. compute the covariance matrix
        warp_i = torch.remainder(self.bins - m_u_idx + (self._num_bins/2) - 1, self._num_bins).unsqueeze(1).unsqueeze(-1)
        warp_j = torch.remainder(self.bins - m_v_idx + (self._num_bins/2) - 1, self._num_bins).unsqueeze(1).unsqueeze(1)

        E_i = torch.sum(pi * warp_i, 2, keepdim=True)
        E_ii = torch.sum(pi * warp_i * warp_i, 2, keepdim=True)
        E_j = torch.sum(pj * warp_j, 3, keepdim=True)
        E_jj = torch.sum(pj * warp_j * warp_j, 3, keepdim=True)

        sigma_00 = self._epsilon + E_ii - E_i*E_i
        sigma_11 = self._epsilon + E_jj - E_j*E_j
        sigma_o = torch.mul(bin_probability, torch.matmul(warp_i, warp_j)).sum(2, keepdim=True).sum(3, keepdim=True) - E_i*E_j

        sigma = torch.cat([sigma_00, sigma_o, sigma_o, sigma_11], 1).squeeze(-1).squeeze(-1)
        sigma = sigma.view(sigma.shape[0], 1, 2, 2).squeeze(1)

        out = {'mu': mu_idx, 'sigma': sigma}
        return out

if __name__ == '__main__':
    import cv2

    n_bins = 1024
    bin_size = 1
    starting_uv = 0
    mean = np.array([100, 600])
    covariance = np.array([[500.5886, 400],[0, 500.7801]])
    n = 10000
    dist = np.random.multivariate_normal(mean, covariance, n)
    dist = np.round(dist).astype(np.int)
    u = dist[:, 0]
    v = dist[:, 1]

    hist, xedges, yedges = np.histogram2d(u, v, n_bins, [[0, n_bins], [0, n_bins]])
    hist = hist / hist.sum()

    hist_torch = torch.FloatTensor(hist).unsqueeze(0).unsqueeze(0)
    bvm = BivariateVonMises(n_bins, bin_size, starting_uv)
    output = bvm(hist_torch)

    id = 0
    u = int(output['mu'][id, 0])
    v = int(output['mu'][id, 1])
    sigma = output['sigma'].data.cpu().numpy()[id, :, :] # B x 2 x 2
    print(u, v)
    print(sigma)
    eigenvalues, eigenvectors = np.linalg.eig(sigma)
    angle = math.atan2(eigenvectors[1, 0], eigenvectors[0, 0])

    if angle < 0:
        angle += 2*math.pi

    angle = angle*180.0/math.pi

    chisquare_val = 2.4477 # 95% confidence interval
    axis_len = (round(chisquare_val*math.sqrt(eigenvalues[1])), round(chisquare_val*math.sqrt(eigenvalues[0])))
    mean = (v, u)

    hist_im = np.stack(((255*hist/hist.max()).astype(np.uint8),)*3, axis=2)
    hist_im[u, v, :] = (0, 255, 0)
    cv2.ellipse(hist_im, mean, axis_len, -angle, 0.0, 360.0, (0, 255, 0), 1)

    cv2.imwrite('hist.png', hist_im)
