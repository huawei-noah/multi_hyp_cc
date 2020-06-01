import cv2
import numpy as np

class Resize():
    def __init__(self, worker, size, dict_resize={'rgb':'linear', 'mask':'nearest'}):
        self._size = size
        self._dict_resize = dict_resize

    def _str_interpolation(self, string):
        if string == 'linear':
            return cv2.INTER_LINEAR
        elif string == 'nearest':
            return cv2.INTER_NEAREST
        else:
            raise Exception('Wrong interpolation type: ' + str(string))

    def __call__(self, input_dict):
        for key in self._dict_resize.keys():
            input = input_dict[key]
            output = []
            for i in range(input.shape[0]):
                im = cv2.resize(input[i, ...], (self._size[1], self._size[0]), interpolation=self._str_interpolation(self._dict_resize[key]))
                im = im[np.newaxis, ...]
                output.append(im)
            input_dict[key] = np.concatenate(output, 0)

        return input_dict
