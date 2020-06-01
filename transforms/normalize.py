import numpy as np
import torch

# normalize to [0, 1]
class Normalize():
    def __init__(self, worker=None):
        pass

    def _is_numpy_image(self, img):
        return isinstance(img, np.ndarray) and (img.ndim in {2, 3})

    def __call__(self, input_dict):
        input = input_dict['rgb'].astype(np.float32)
        max_value = np.iinfo(input_dict['rgb'].dtype).max

        for i in range(input.shape[0]):
            pic = input[i, ...]
            if not self._is_numpy_image(pic):
                raise TypeError('pic should be ndarray. Got {}'.format(type(pic)))

            # handle numpy array
            if pic.ndim == 2:
                pic = pic[:, :, None]

            # normalize
            pic = pic.astype(np.float32) / max_value

            input[i, ...] = pic

        input = input.transpose((0, 3, 1, 2))
        input_dict['rgb'] = input

        return input_dict
