#Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

#This program is free software; you can redistribute it and/or modify it under the terms of the BSD 0-Clause License.

#This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD 0-Clause License for more details.

import numpy as np
from tensorboardX import SummaryWriter
from PIL import Image
from io import BytesIO
from core.utils import axis_numpy, axis_pytorch

# log output to tensorboard folder
class TensorBoardLogger(object):
    def __init__(self, log_dir):
        """Create a summary writer logging to log_dir."""
        self.writer = SummaryWriter(log_dir)

    def scalar_summary(self, tag, value, step):
        """Log a scalar variable."""
        self.writer.add_scalar(tag, value, step)

    def image_summary(self, tag, images, step):
        """Log a list of images."""

        img_summaries = []
        for i, img in enumerate(images):
            # Write the image to a string
            if len(img.shape) == 3:
                np_im = img
            else:
                images = []
                for j in range(img.shape[0]):
                    if len(img.shape) == 5:
                        np_im = np.moveaxis(img[j, ...], 1, 3)
                    elif len(img.shape) == 4:
                        np_im = axis_numpy(img[j, ...])
                    else:
                        np_im = img[j, ...]
                    images.append(np_im)
                np_im = np.hstack(images)
                np_im = axis_pytorch(np_im)

            # Create an Image object
            self.writer.add_image('%s/%d' % (tag, i), np_im, step)

    def histo_summary(self, tag, values, step, bins=1000):
        """Log a histogram of the tensor of values."""
        self.writer.add_histogram(tag, values, step, bins=bins)
