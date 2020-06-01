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
