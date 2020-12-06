# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved. THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.

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
