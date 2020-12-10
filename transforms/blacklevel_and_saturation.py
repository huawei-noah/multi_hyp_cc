#Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

#This program is free software; you can redistribute it and/or modify it under the terms of the BSD 0-Clause License.

#This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD 0-Clause License for more details.

import numpy as np
from core.utils import blacklevel_saturation_correct

# black level correction: subtract the constant black level from input images,
# this is camera dependent. We also want to avoid having saturated pixels,
# because they don't behave linearly.
class BlacklevelAndSaturation():
    def __init__(self, worker, saturation_scale = 0.95):
        # saturation threshold
        self._saturation_scale = saturation_scale

    def __call__(self, input_dict):
        im = input_dict['rgb']
        sensor = input_dict['sensor']

        # correct all images in the batch
        for i in range(im.shape[0]):
            im[i, ...] = blacklevel_saturation_correct(im[i, ...], sensor[i], saturation_scale = self._saturation_scale)

        input_dict['rgb'] = im
        return input_dict
