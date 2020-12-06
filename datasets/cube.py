# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved. THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.

import torch
from torch.utils.data.dataset import Dataset
import cv2
import numpy as np
from datasets.sensor import Sensor
import os
import sys
import time
import math
import scipy.io

with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'ccm', 'canon_eos_550d.txt'), 'r') as f:
    CCM = torch.FloatTensor(np.loadtxt(f))

class CanonEOS550D(Sensor):
    def __init__(self, saturation):
        super(CanonEOS550D, self).__init__(2048, saturation, CCM, 'CanonEOS550D')

# Cube (and Cube+) dataset: https://ipg.fer.hr/ipg/resources/color_constancy
class Cube(Dataset):
    def __init__(self, subdataset, data_conf, file, cache):
        self._rgbs = []
        self._illuminants = []
        self._secondary_illuminants = []
        self._base_path = data_conf['cube_'+subdataset]
        self._subdataset = subdataset

        gt, second_gt = self._read_gt()

        if type(file) is list:
            for f in file:
                self._read_list(os.path.join(data_conf['base'], f), gt, second_gt)
        else:
            self._read_list(os.path.join(data_conf['base'], file), gt, second_gt)

        self._cache = cache

    def get_filename(self, index):
        return self._rgbs[index]

    def get_illuminants(self):
        return self._illuminants

    def get_illuminants_by_sensor(self):
        dict = {'CanonEOS550D': self._illuminants}
        return dict

    def _read_gt(self):
        gt_dir = os.path.join(self._base_path, self._subdataset, 'gt')
        gt = self._read_gt_file(os.path.join(gt_dir, 'gt.txt'))

        left_file = os.path.join(gt_dir, 'left_gt.txt')
        right_file = os.path.join(gt_dir, 'right_gt.txt')
        left_gt = None
        if os.path.isfile(left_file):
            left_gt = self._read_gt_file(left_file)
        right_gt = None
        if os.path.isfile(right_file):
            right_gt = self._read_gt_file(right_file)

        if left_gt is not None and right_gt is not None:
            second_gt = {}
            for key in gt.keys():
                gt_ill = gt[key]
                left_ill = left_gt[key]
                right_ill = right_gt[key]

                if gt_ill == left_ill:
                    second_gt[key] = right_ill
                elif gt_ill == right_ill:
                    second_gt[key] = left_ill
                else:
                    raise Exception('GT should be equal to either left GT or right GT')
        else:
            second_gt = None

        return gt, second_gt

    def _read_gt_file(self, file):
        gt = {}
        with open(file, 'r') as f:
            content = f.readlines()
            for i in range(len(content)):
                line = content[i]
                illuminant = line.strip().split(' ')
                illuminant = [float(e) for e in illuminant]
                length = math.sqrt(sum([e*e for e in illuminant]))
                illuminant = [e / length for e in illuminant]
                filename = str(i+1) + '.png'
                gt[filename] = illuminant
        return gt


    def _read_list(self, file, gt, second_gt):
        with open(file, 'r') as f:
            content = f.readlines()
            for line in content:
                filename = line.strip()
                rgb_path = os.path.join(self._base_path, filename)
                self._rgbs.append(rgb_path)

                illuminant = gt[os.path.basename(filename)]
                self._illuminants.append(illuminant)

                if second_gt is not None:
                    illuminant2 = second_gt[os.path.basename(filename)]
                    self._secondary_illuminants.append(illuminant2)

    def get_rgb_by_path(self, filename):
        if self._cache.is_cached(filename):
            im, mask, sensor = self._cache.read(filename)
        else:
            im = cv2.imread(filename, -1)
            if im is None:
                raise Exception('File not found: ' + filename)
            sensor = CanonEOS550D(im.max()-2)

            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

            # get mask of valid pixels
            mask = np.ones(im.shape[:2], dtype = np.float32)
            rc = np.array(([2049, im.shape[1]-1, im.shape[1]-1, 2049], [1049, 1049, im.shape[0]-1, im.shape[0]-1])).T
            ctr = rc.reshape((-1,1,2)).astype(np.int32)
            cv2.drawContours(mask, [ctr], 0, 0, -1)

            # set CCB pixels to zero
            # TODO: ideally, downsampling should consider the mask
            # and then, apply the mask as a final step
            im[mask == 0] = [0, 0, 0]

            self._cache.save(filename, (im, mask, sensor))

        im = im[np.newaxis, ...]
        mask = mask[np.newaxis, ...]
        sensor = [sensor]

        return im, mask, sensor

    def get_rgb(self, index):
        path = self._rgbs[index]

        return self.get_rgb_by_path(path)

    def __getitem__(self, index):
        filename = self._rgbs[index]

        im, mask, sensor = self.get_rgb(index)

        illuminant = np.array(self._illuminants[index], dtype=np.float32)
        secondary_illuminant = None
        if index in self._secondary_illuminants:
            secondary_illuminant = np.array(self._secondary_illuminants[index], dtype=np.float32)

        dict = {'rgb': im, 'sensor': sensor, 'mask': mask,
                'illuminant': illuminant,
                'path': filename}

        if secondary_illuminant is not None:
            dict['secondary_illuminant'] = secondary_illuminant

        return dict

    def __len__(self):
        return len(self._rgbs)
