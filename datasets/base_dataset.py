# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved. THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.#Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

#This program is free software; you can redistribute it and/or modify it under the terms of the BSD 0-Clause License.

#This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD 0-Clause License for more details.

import torch
from torch.utils.data.dataset import Dataset
import cv2
import numpy as np
from datasets.sensor import Sensor
import os
import sys
import copy
from core.utils import log_uv_histogram_checks, find_loguv_warp_conf
from transforms.normalize import Normalize

class BaseDataset(Dataset):
    def __init__(self, subdataset, data_conf, dataset_file,
                dataset_class, conf, gpu,
                transform, required_input,
                cache_manager, is_valtest, verbose):

        dataset_cache = cache_manager.dataset()

        self._dataset = dataset_class(subdataset, data_conf, dataset_file, dataset_cache)
        self._conf = conf
        self._gpu = gpu
        self._transform = transform
        self._required_input = required_input
        if self._required_input is not None:
            self._required_input += ['sensor', 'path']
        self._cache_transforms = cache_manager.transforms()
        self._is_valtest = is_valtest
        self._normalize = Normalize()

        if verbose:
            illuminants = self._dataset.get_illuminants()
            log_uv_warp_histogram = find_loguv_warp_conf(self._transform)
            if log_uv_warp_histogram is not None:
                log_uv_histogram_checks(log_uv_warp_histogram, illuminants)

    def get_illuminants(self):
        return self._dataset.get_illuminants()

    def get_illuminants_by_sensor(self):
        return self._dataset.get_illuminants_by_sensor()

    def get_rgb_by_path(self, path):
        return self._dataset.get_rgb_by_path(path)

    def get_rgb(self, index):
        return self._dataset.get_rgb(index)

    def __getitem__(self, index):
        filename = self._dataset.get_filename(index)
        if self._cache_transforms.is_cached(filename):
            dict = self._cache_transforms.read(filename)
        else:
            dict = self._dataset[index]

            if self._transform is not None:
                dict = self._transform(copy.deepcopy(dict))

            # remove not required keys to save memory
            if self._required_input is not None:
                keys = list(dict.keys())
                for k in keys:
                    if k not in self._required_input:
                        del dict[k]

            # convert to pytorch tensor
            keys = list(dict.keys())
            for k in keys:
                if isinstance(dict[k], (np.ndarray, np.generic)):
                    dict[k] = torch.from_numpy(dict[k])

            self._cache_transforms.save(filename, dict)

        sensor = dict['sensor']
        path = dict['path']
        exclude_gpu = ['sensor', 'path']

        final_dict = {}
        if self._gpu is not None:
            for k in dict.keys():
                if k not in exclude_gpu:
                    final_dict[k] = dict[k].cuda(self._gpu)
        else:
            final_dict = dict.copy()
            del final_dict['sensor']
            del final_dict['path']

        for i in range(len(sensor)):
            if isinstance(sensor[i], Sensor):
                sensor[i] = sensor[i].to_dict()

        final_dict['sensor'] = sensor
        final_dict['path'] = path

        return final_dict

    def __len__(self):
        return len(self._dataset)
