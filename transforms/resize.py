#Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

#This program is free software; you can redistribute it and/or modify it under the terms of the BSD 0-Clause License.

#This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD 0-Clause License for more details.

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
