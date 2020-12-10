#Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

#This program is free software; you can redistribute it and/or modify it under the terms of the BSD 0-Clause License.

#This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD 0-Clause License for more details.

import torch
from torch.utils.data.dataset import Dataset
import cv2
import numpy as np
import os
import sys
import pdb

from core.utils import import_shortcut

# Class used to train/test with multiple datasets, see example: data/multidataset/nus.txt
class Multidataset(Dataset):
    def __init__(self, subdataset, data_conf, file, cache):
        self._datasets = []
        self._cache = cache

        if type(file) is list:
            for f in file:
                self._read_list(os.path.join(data_conf['base'], f), data_conf)
        else:
            self._read_list(os.path.join(data_conf['base'], file), data_conf)

        self._datasets_n = []
        n = 0
        for dataset in self._datasets:
            n += len(dataset)
            self._datasets_n.append(n)

    def _get_dataset(self, index):
        prev_n = 0
        i = 0
        for n in self._datasets_n:
            if index < n:
                return self._datasets[i], index-prev_n
            i += 1
            prev_n = n

        return None, None

    def get_filename(self, index):
        dataset, i = self._get_dataset(index)
        return dataset.get_filename(i)

    def get_illuminants(self):
        illuminants = []
        for dataset in self._datasets:
            illuminants += dataset.get_illuminants()

        return illuminants

    def get_illuminants_by_sensor(self):
        illuminants = {}
        for dataset in self._datasets:
            dataset_ill = dataset.get_illuminants_by_sensor()
            for key in dataset_ill.keys():
                # if we already had some images from the same camera,
                # add new images to the list
                if key in illuminants:
                    illuminants[key] += dataset_ill[key]
                else:
                    illuminants[key] = dataset_ill[key]

        return illuminants

    def _read_list(self, file, data_conf):
        with open(file, 'r') as f:
            content = f.readlines()
            for line in content:
                elements = line.strip().split(' ')
                dataset_class_name = elements[0]
                subdataset = elements[1]
                dataset_file = elements[2]

                dataset_class = import_shortcut('datasets', dataset_class_name)
                self._datasets.append(dataset_class(subdataset, data_conf, dataset_file, self._cache))

    def get_rgb_by_path(self, path):
        for dataset in self._datasets:
            if path in dataset._rgbs:
                return dataset.get_rgb_by_path(path)
        return None

    def get_rgb(self, index):
        dataset, i = self._get_dataset(index)
        path = dataset.get_filename(i)

        return dataset.get_rgb_by_path(path)

    def __getitem__(self, index):
        im, mask, sensor = self.get_rgb(index)
        dataset, i = self._get_dataset(index)
        path = dataset.get_filename(i)

        illuminant = np.array(dataset._illuminants[i], dtype=np.float32)

        dict = {'rgb': im, 'sensor': sensor, 'mask': mask,
                'illuminant': illuminant, 'path': path}

        return dict

    def __len__(self):
        return self._datasets_n[-1]
