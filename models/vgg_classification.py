#Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

#This program is free software; you can redistribute it and/or modify it under the terms of the BSD 0-Clause License.

#This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD 0-Clause License for more details.

import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch
import math
import torch.nn.functional as F
from torch.autograd import Variable
from core.utils import angular_error_degress_np, import_shortcut
import torchvision
from torchvision.models.vgg import make_layers, VGG
try:
    from torchvision.models.vgg import cfgs
except:
    from torchvision.models.vgg import cfg as cfgs
import numpy as np

import time

# Multi-Hypothesis model

class VggClassification(VGG):

    def __init__(self, conf, pretrained=False, dropout = 0.5,
                fix_base_network = False,
                final_affine_transformation = False,
                n_fc_layers = 3, n_pointwise_conv = 2, n_features = 128,
                **kwargs):
        """Multi-Hypothesis model constructor.

        Args:
            conf (dict): Dictionary of the configuration file (json).
            pretrained (:obj:`bool`, optional): Whether used pretrained model
                on ImageNet.
            dropout (:obj:`float`): Probability of dropout.
            fix_base_network (:obj:`bool`): If true, we don't learn the first
                convolution, just use the weights from pretrained model.
            final_affine_transformation (:obj:`bool`): Whether to learn an
                affine transformation of the output of the network.
            n_fc_layers (:obj:`int`): Number of FC layers after conv layers.
            n_pointwise_conv (:obj:`int`): Number of 1x1 conv layers after
                first 3x3 convolution.
            n_features (:obj:`int`): Number of feature for the final 1x1 conv.
        """
        arch = conf['network']['subarch']
        # vgg11 or vgg11 with batch norm
        if arch == 'vgg11':
            super(VggClassification, self).__init__(make_layers(cfgs['A']), **kwargs)
        elif arch == 'vgg11_bn':
            super(VggClassification, self).__init__(make_layers(cfgs['A'], batch_norm=True), **kwargs)
        else:
            raise Exception('Wrong architecture')

        if pretrained:
            self.load_state_dict(model_zoo.load_url(torchvision.models.vgg.model_urls[arch]))

        # this is used to remove all cameras not included in this setting,
        # if defined in the conf file.
        if 'cameras' in conf:
            self._cameras = conf['cameras']
        else:
            self._cameras = None

        # load candidate selection method:
        # see folder modules/candidate_selection/ for available methods.
        class_obj = import_shortcut('modules.candidate_selection',
                                    conf['candidate_selection']['name'])
        self.candidate_selection = class_obj(conf, **conf['candidate_selection']['params'])
        self._final_affine_transformation = final_affine_transformation

        # we keep only the first VGG convolution!
        self.conv1 = self.features[0]
        self.relu1 = self.features[1]

        # then, we have N="n_pointwise_conv" number of 1x1 convs
        N = n_features
        pointwise_layers = []
        for n_layers in range(n_pointwise_conv):
            n_output = 64
            if n_layers == n_pointwise_conv - 1:
                n_output = N
            pointwise_layers.append(nn.Conv2d(64, n_output, kernel_size=1))
            pointwise_layers.append(nn.ReLU(inplace=True))

        self.pointwise_conv = nn.Sequential(*pointwise_layers)

        # remove VGG features and classifier (FCs of the net)
        del self.features
        del self.classifier

        # if this option is enabled, we don't learn the first conv weights,
        # they are copied from VGG pretrained on Imagenet (from pytorch),
        # the weights: /torch_model_zoo/vgg11-bbd30ac9.pth
        if fix_base_network:
            # do not learn any parameters from VGG
            included = ['conv1.bias', 'conv1.weight']
            for name, param in self.named_parameters():
                if name in included:
                    #print('name',name, param.shape)
                    param.requires_grad = False

        # probability of dropout
        self.dropout = nn.Dropout(dropout)

        # final FC layers: from N to 1 (probability for illuminant)
        final_n = 1

        # N: initial feature size
        # n_fc_layers: number of FC layers
        # final_n: size of final prediction
        self.fc = self._fc_layers(N, n_fc_layers, final_n)
        self.softmax = nn.Softmax(1)
        self.logsoftmax = nn.LogSoftmax(1)

    def _fc_layers(self, n_features, n_fc_layers, n_final):
        # generate "n_fc_layers" FC layers,
        # initially, we have "n_features" features and,
        # we convert that to n_final
        # we reduce the number of features //2 every time
        fc_layers = []
        n_curr = n_features
        for fc_i in range(n_fc_layers):
            next_n = n_curr // 2
            if fc_i == n_fc_layers-1:
                next_n = n_final

            fc_layers.append(nn.Linear(n_curr, next_n, bias=True))
            if fc_i != n_fc_layers-1:
                fc_layers.append(nn.ReLU(inplace=True))

            n_curr = next_n

        return nn.Sequential(*fc_layers)

    def _fc_layers_affine(self, n_features, n_fc_layers, n_layers):
        # same as previous function but we specify the size
        # of intermediate features
        fc_layers = []
        n_curr = n_features
        for fc_i in range(n_fc_layers):
            N = n_layers[fc_i]
            fc_layers.append(nn.Linear(n_curr, N, bias=True))
            if fc_i != n_fc_layers-1:
                fc_layers.append(nn.ReLU(inplace=True))

            n_curr = N

        return nn.Sequential(*fc_layers)

    def initialize(self, illuminants = None):
        # when running the constructor, we don't have the
        # training set illuminants yet, so, we run initialize
        # to do the candidate selection
        if illuminants is not None:
            # here we save one candidate set per camera
            for key in illuminants.keys():
                candidates = self.candidate_selection.initialize(illuminants[key])
                self.register_buffer('clusters_'+key, candidates)
        else:
            # if illuminants is None, we init the candidate set with
            # default values (zeros). This is used for inference scripts,
            # we don't have the illuminants when we call this function,
            # but it does not matter because we will load it from
            # the checkpoint file.
            for key in self._cameras:
                candidates = self.candidate_selection.initialize(None)
                self.register_buffer('clusters_'+key, candidates)

        # Number of candidate illuminants
        self.n_output = candidates.shape[1]

        # Final affine transformation (B and G in the paper)
        if self._final_affine_transformation:
            self.mult = nn.Parameter(torch.ones(1, self.n_output))
            self.add = nn.Parameter(torch.zeros(1, self.n_output))

    def _inference(self, input, gt_illuminant, sensor, candidates, do_dropout = True):
        # output logics and confs
        logits = torch.zeros(input.shape[0], candidates.shape[1]).type(input.type())

        # loop illuminants
        for i in range(candidates.shape[1]):
            # get illuminant "i"
            illuminant = candidates[:, i, :].unsqueeze(-1).unsqueeze(-1)
            # generate candidate image! image / illuminant
            input2 = input / illuminant

            # avoid log(0)
            input2 = torch.log(input2 + 0.000001)

            # first conv (from VGG)
            x = self.conv1(input2)
            x = self.relu1(x)

            # point-wise convs (1x1)
            x = self.pointwise_conv(x)

            # global average pooling, from 64x64x128 to 1x1x128
            x = F.adaptive_avg_pool2d(x, (1, 1))

            # dropout (in paper=0.5)
            x = self.dropout(x)

            # reshape 1x1x128 -> 128
            x = x.view(x.size(0), -1)
            # FC layers: 128 -> 1
            x = self.fc(x)

            # save log-likelihood for this illuminant
            logits[:, i] = x[:, 0]

        # apply affine transformation (G and B)
        if self._final_affine_transformation:
            # just get B and G
            mult = self.mult
            add = self.add

            # save softmax of the pre-affine output (for visualization)
            bin_probability_pre_affine = self.softmax(logits)
            # apply aff. trans.
            logits = mult*logits + add

        # softmax!
        bin_probability = self.softmax(logits)

        # do linear comb. in RGB
        illuminant_r = torch.sum(bin_probability*candidates[:, :, 0], 1, keepdim=True)
        illuminant_g = torch.sum(bin_probability*candidates[:, :, 1], 1, keepdim=True)
        illuminant_b = torch.sum(bin_probability*candidates[:, :, 2], 1, keepdim=True)
        illuminant = torch.cat([illuminant_r, illuminant_g, illuminant_b], 1)

        # l2-normalization of the vector
        norm = torch.norm(illuminant, p=2, dim=1, keepdim=True)
        illuminant = illuminant.div(norm.expand_as(illuminant))

        # save output to dictionary
        d = {'illuminant': illuminant, 'bin_probability2': bin_probability,
            'logits': logits, 'candidates': candidates}

        if self._final_affine_transformation:
            d["bias2"] = add
            d["gain2"] = mult
            d["bin_probability_preaffine"] = bin_probability_pre_affine

        return d

    def forward(self, data, image_index=0):
        input = data['rgb']
        if 'illuminant' in data:
            gt_illuminant = Variable(data['illuminant'])
        else:
            gt_illuminant = None
        sensor = data['sensor']

        # only one image
        input = input[:, image_index, ...]

        # get candidates
        candidates_cam = getattr(self, 'clusters_' + sensor[0]['camera_name'][0])

        # run image specific tunning of the candidate set
        # this only applies for Minkowski candidate selection
        # (modules/candidate_selection/minkowski.py)
        candidates = self.candidate_selection.run(input, candidates_cam)

        # do inference
        return self._inference(input, gt_illuminant, sensor, candidates, self.training)
