# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved. THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.

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
