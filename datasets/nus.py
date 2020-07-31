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

CCM_NUS = {}
with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'ccm', 'nus', 'canon_eos_1D_mark3.txt'), 'r') as f:
    CCM_NUS['canon_eos_1D_mark3'] = torch.FloatTensor(np.loadtxt(f))

with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'ccm', 'nus', 'canon_eos_600D.txt'), 'r') as f:
    CCM_NUS['canon_eos_600D'] = torch.FloatTensor(np.loadtxt(f))

with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'ccm', 'nus', 'fuji.txt'), 'r') as f:
    CCM_NUS['fuji'] = torch.FloatTensor(np.loadtxt(f))

with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'ccm', 'nus', 'nikonD40.txt'), 'r') as f:
    CCM_NUS['nikonD40'] = torch.FloatTensor(np.loadtxt(f))

with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'ccm', 'nus', 'nikonD5200.txt'), 'r') as f:
    CCM_NUS['nikonD5200'] = torch.FloatTensor(np.loadtxt(f))

with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'ccm', 'nus', 'olympus.txt'), 'r') as f:
    CCM_NUS['olympus'] = torch.FloatTensor(np.loadtxt(f))

with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'ccm', 'nus', 'panasonic.txt'), 'r') as f:
    CCM_NUS['panasonic'] = torch.FloatTensor(np.loadtxt(f))

with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'ccm', 'nus', 'samsung.txt'), 'r') as f:
    CCM_NUS['samsung'] = torch.FloatTensor(np.loadtxt(f))

with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'ccm', 'nus', 'sony.txt'), 'r') as f:
    CCM_NUS['sony'] = torch.FloatTensor(np.loadtxt(f))

class CanonEos1DMark3Sensor(Sensor):
    def __init__(self, black_level, saturation):
        super(CanonEos1DMark3Sensor, self).__init__(black_level, saturation, CCM_NUS['canon_eos_1D_mark3'], 'CanonEos1DMark3')

class CanonEos600DSensor(Sensor):
    def __init__(self, black_level, saturation):
        super(CanonEos600DSensor, self).__init__(black_level, saturation, CCM_NUS['canon_eos_600D'], 'CanonEos600D')

class FujiSensor(Sensor):
    def __init__(self, black_level, saturation):
        super(FujiSensor, self).__init__(black_level, saturation, CCM_NUS['fuji'], 'FujifilmXM1')

class NikonD40Sensor(Sensor):
    def __init__(self, black_level, saturation):
        super(NikonD40Sensor, self).__init__(black_level, saturation, CCM_NUS['nikonD40'], 'NikonD40')

class NikonD5200Sensor(Sensor):
    def __init__(self, black_level, saturation):
        super(NikonD5200Sensor, self).__init__(black_level, saturation, CCM_NUS['nikonD5200'], 'NikonD5200')

class OlympusSensor(Sensor):
    def __init__(self, black_level, saturation):
        super(OlympusSensor, self).__init__(black_level, saturation, CCM_NUS['olympus'], 'OlympusEPL6')

class PanasonicSensor(Sensor):
    def __init__(self, black_level, saturation):
        super(PanasonicSensor, self).__init__(black_level, saturation, CCM_NUS['panasonic'], 'PanasonicGX1')

class SamsungSensor(Sensor):
    def __init__(self, black_level, saturation):
        super(SamsungSensor, self).__init__(black_level, saturation, CCM_NUS['samsung'], 'SamsungNX2000')

class SonySensor(Sensor):
    def __init__(self, black_level, saturation):
        super(SonySensor, self).__init__(black_level, saturation, CCM_NUS['sony'], 'SonyA57')

CAMERA_CLASS_NUS = {}
CAMERA_CLASS_NUS['canon_eos_1D_mark3'] = CanonEos1DMark3Sensor
CAMERA_CLASS_NUS['canon_eos_600D'] = CanonEos600DSensor
CAMERA_CLASS_NUS['fuji'] = FujiSensor
CAMERA_CLASS_NUS['nikonD40'] = NikonD40Sensor
CAMERA_CLASS_NUS['nikonD5200'] = NikonD5200Sensor
CAMERA_CLASS_NUS['olympus'] = OlympusSensor
CAMERA_CLASS_NUS['panasonic'] = PanasonicSensor
CAMERA_CLASS_NUS['samsung'] = SamsungSensor
CAMERA_CLASS_NUS['sony'] = SonySensor

# NUS dataset: http://cvil.eecs.yorku.ca/projects/public_html/illuminant/illuminant.html
class Nus(Dataset):
    def __init__(self, subdataset, data_conf, file, cache):
        self._rgbs = []
        self._illuminants = []
        self._x = {}
        self._y = {}
        self._base_path = data_conf['nus_'+subdataset]

        if type(file) is list:
            for f in file:
                self._read_list(os.path.join(data_conf['base'], f))
        else:
            self._read_list(os.path.join(data_conf['base'], file))

        gt = scipy.io.loadmat(os.path.join(self._base_path, 'gt', 'gt.mat'))

        self._darkness_level = int(gt['darkness_level'][0][0])
        self._saturation_level = int(gt['saturation_level'][0][0])
        self._class_dataset = CAMERA_CLASS_NUS[subdataset]

        image_names = gt['all_image_names'].tolist()
        image_names = [e[0][0] for e in image_names]
        groundtruth_illuminants = gt['groundtruth_illuminants']
        CC_coords = gt['CC_coords']

        for i in range(len(self._rgbs)):
            basename_rgb = os.path.basename(self._rgbs[i].replace('.PNG', ''))
            index = image_names.index(basename_rgb)
            self._illuminants.append(groundtruth_illuminants[index, :])

        self._cache = cache

    def get_filename(self, index):
        return self._rgbs[index]

    def get_illuminants(self):
        return self._illuminants

    def get_illuminants_by_sensor(self):
        dict = {self._class_dataset(None,None).camera_name: self._illuminants}
        return dict

    def _read_list(self, file):
        with open(file, 'r') as f:
            content = f.readlines()
            for line in content:
                filename = line.strip()
                rgb_path = os.path.join(self._base_path, 'PNG', filename)
                self._rgbs.append(rgb_path)

                txt = filename.replace('.PNG','_mask.txt')
                mask_path = os.path.join(self._base_path, 'mask', txt)
                with open(mask_path, 'r') as file_txt:
                    coordinates = file_txt.readlines()[0].split(',')
                x = float(coordinates[0])
                width = float(coordinates[2])
                y = float(coordinates[1])
                height = float(coordinates[3])
                self._x[rgb_path] = [x, x+width, x+width, x]
                self._y[rgb_path] = [y, y, y+height, y+height]

    def get_rgb_by_path(self, filename):
        sensor = self._class_dataset(self._darkness_level, self._saturation_level)
        if self._cache.is_cached(filename):
            im, mask = self._cache.read(filename)
        else:
            im = cv2.imread(filename, -1)
            if im is None:
                raise Exception('File not found: ' + filename)

            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

            # get mask of valid pixels
            mask = np.ones(im.shape[:2], dtype = np.float32)
            rc = np.array((self._x[filename], self._y[filename])).T
            ctr = rc.reshape((-1,1,2)).astype(np.int32)
            cv2.drawContours(mask, [ctr], 0, 0, -1)

            # set CCB pixels to zero
            # TODO: ideally, downsampling should consider the mask
            # and then, apply the mask as a final step
            im[mask == 0] = [0, 0, 0]

            # rotate 90 degrees so that all images have the same resolution
            if im.shape[0] == 2820:
                im = cv2.transpose(im)
                im = cv2.flip(im, flipCode=0)
                mask = cv2.transpose(mask)
                mask = cv2.flip(mask, flipCode=0)

            self._cache.save(filename, (im, mask))

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

        dict = {'rgb': im, 'sensor': sensor, 'mask': mask,
                'illuminant': illuminant, 'path': filename}

        return dict

    def __len__(self):
        return len(self._rgbs)


if __name__ == '__main__':
    import scipy
    import scipy.io

    path = 'data/nus/'
    camera = scipy.io.loadmat(os.path.join(path, 'cv_metadata.mat'))['cv_metadata'][0][0]
    cameras_list = ['canon_eos_1D_mark3', 'canon_eos_600D', 'fuji',
                    'nikonD40', 'nikonD5200', 'olympus', 'panasonic',
                    'samsung', 'sony']

    camera_folds = []
    for i in range(len(cameras_list)):
        camera_folds.append([[],[],[]])

    for i in range(len(camera)):
        files = camera[i]
        for j in range(len(files)):
            file = files[j]
            filename = file[0][0][0]
            scene_idx = file[0][1][0][0] - 1
            cv_fold = file[0][2][0][0] - 1
            camera_folds[i][cv_fold].append(filename)

    for i in range(len(camera)):
        camera_name = cameras_list[i]
        with open(os.path.join(path, camera_name + '.txt'), 'w') as f:
            f.write('Nus\n')
            f.write('data/nus/splits/'+camera_name+'/fold1.txt\n')
            f.write('data/nus/splits/'+camera_name+'/fold2.txt\n')
            f.write('data/nus/splits/'+camera_name+'/fold3.txt\n')

        for fold in range(3):
            camera_folder = os.path.join(path, 'splits', camera_name)
            os.makedirs(camera_folder, exist_ok=True)
            with open(os.path.join(camera_folder, 'fold'+str(fold+1)+'.txt'), 'w') as f:
                files = camera_folds[i][fold]
                for file in files:
                    f.write(file+'\n')
