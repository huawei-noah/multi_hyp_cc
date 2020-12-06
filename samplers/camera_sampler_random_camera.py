# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved. THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.

from torch.utils.data.sampler import Sampler
import random

# same as CameraSampler but:
# we sample the cameras randomly (also the images within each camera dataset)
class CameraSamplerRandomCamera(Sampler):
    def __init__(self, data, batch_size):
        data_camera = {}

        for i in range(len(data)):
            camera = data[i]['sensor'][0]['camera_name']
            if camera not in data_camera:
                data_camera[camera] = []

            data_camera[camera].append(i)

        self.n = 0
        for key in data_camera.keys():
            self.n += len(data_camera[key])

        # shuffle the unique images
        for key in data_camera.keys():
            random.shuffle(data_camera[key])

        self.batches = []
        batch = []
        for key in data_camera.keys():
            for i in range(len(data_camera[key])):
                data = data_camera[key][i]
                batch.append(data)
                if len(batch) >= batch_size:
                    self.batches.append(batch)
                    batch = []
            if len(batch) > 0:
                self.batches.append(batch)
                batch = []

    def __iter__(self):
        random.shuffle(self.batches)

        for batch in self.batches:
            yield batch

    def __len__(self):
        return self.n
