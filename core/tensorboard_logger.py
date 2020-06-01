import tensorflow as tf
import numpy as np
from PIL import Image
from io import BytesIO
from core.utils import axis_numpy

# log output to tensorboard folder
class TensorBoardLogger(object):
    def __init__(self, log_dir):
        """Create a summary writer logging to log_dir."""
        self.writer = tf.summary.FileWriter(log_dir)

    def scalar_summary(self, tag, value, step):
        """Log a scalar variable."""
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
        self.writer.add_summary(summary, step)

    def image_summary(self, tag, images, step):
        """Log a list of images."""

        img_summaries = []
        for i, img in enumerate(images):
            # Write the image to a string
            s = BytesIO()

            if len(img.shape) == 3:
                np_im = axis_numpy(img)
                if np_im.shape[-1] == 1:
                    np_im = np_im.squeeze(-1)
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

            #print('fromarray',tag,np_im.shape)
            im_pillow = Image.fromarray(np_im)
            im_pillow.save(s, format="png")

            # Create an Image object
            width, height = im_pillow.size
            img_sum = tf.Summary.Image(encoded_image_string=s.getvalue(),
                                       height=height,
                                       width=width)
            # Create a Summary value
            img_summaries.append(tf.Summary.Value(tag='%s/%d' % (tag, i), image=img_sum))

        # Create and write Summary
        summary = tf.Summary(value=img_summaries)
        self.writer.add_summary(summary, step)

    def histo_summary(self, tag, values, step, bins=1000):
        """Log a histogram of the tensor of values."""

        # Create a histogram using numpy
        try:
            counts, bin_edges = np.histogram(values, bins=bins)
        except Exception as e:
            print('tag: ', tag, ' values:', values)

        # Fill the fields of the histogram proto
        hist = tf.HistogramProto()
        hist.min = float(np.min(values))
        hist.max = float(np.max(values))
        hist.num = int(np.prod(values.shape))
        hist.sum = float(np.sum(values))
        hist.sum_squares = float(np.sum(values**2))

        # Drop the start of the first bin
        bin_edges = bin_edges[1:]

        # Add bin edges and counts
        for edge in bin_edges:
            hist.bucket_limit.append(edge)
        for c in counts:
            hist.bucket.append(c)

        # Create and write Summary
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, histo=hist)])
        self.writer.add_summary(summary, step)
        self.writer.flush()
