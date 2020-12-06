# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved. THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.

from torch.utils.data.sampler import Sampler
import random

# when training a multi-device model, we want to sample image for a batch
# so that all images belong to the same camera dataset.
# Note that here, we sample the cameras in order
class CameraSampler(Sampler):
    def __init__(self, data, batch_size):
        self.batch_size = batch_size
        self.data_camera = {}

        # iterate all data
        for i in range(len(data)):
            camera = data[i]['sensor'][0]['camera_name']
            # save all cameras
            if camera not in self.data_camera:
                self.data_camera[camera] = []

            # assign image "i" to its camera
            self.data_camera[camera].append(i)

        # sum all the camera datasets length
        self.n = 0
        for key in self.data_camera.keys():
            self.n += len(self.data_camera[key])

    def __iter__(self):
        # shuffle the unique images inside every camera dataset
        for key in self.data_camera.keys():
            random.shuffle(self.data_camera[key])

        # init an empty batch
        batch = []
        # iterate all cameras IN ORDER!
        for key in self.data_camera.keys():
            # iterate all images within a camera dataset
            for i in range(len(self.data_camera[key])):
                # add to batch
                data = self.data_camera[key][i]
                batch.append(data)
                # if batch size is enough, yield
                if len(batch) >= self.batch_size:
                    yield batch
                    batch = []
            # the last batch can be < batch_size
            if len(batch) > 0:
                yield batch
                batch = []

    def __len__(self):
        return self.n
