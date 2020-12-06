# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved. THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.

import numpy as np
from core.utils import rgb_to_uv

def log_uv_histogram_wrapped(im, mask, conf):
    num_bins = conf['num_bins']
    bin_size = conf['bin_size']
    starting_uv = conf['starting_uv']
    min_intensity = conf['min_intensity']
    normalization = conf['normalization']
    postprocess = conf['postprocess']

    r = im[:, :, 0].astype(np.float)
    g = im[:, :, 1].astype(np.float)
    b = im[:, :, 2].astype(np.float)

    max_value = np.iinfo(im.dtype).max
    min_value = min_intensity*max_value

    # ignore black pixels
    bigger_zero = np.logical_and(np.logical_and(r >= 1, g >= 1), b >= 1)
    # ignore pixels smaller than minimum
    bigger_min = np.logical_or(np.logical_or(r >= min_value, g >= min_value), b >= min_value)
    # ignore saturated pixels
    not_saturated = np.logical_and(np.logical_and(r < max_value, g < max_value), b < max_value)

    invalid = np.logical_not(np.logical_and(np.logical_and(bigger_zero, bigger_min), np.logical_and(not_saturated, mask>0)))

    r[invalid] = 1
    g[invalid] = 1
    b[invalid] = 1

    log_r = np.log(r)
    log_g = np.log(g)
    log_b = np.log(b)

    # ignore invalid pixels
    invalid_log = np.logical_not(np.logical_and(np.logical_and(np.isfinite(log_r), np.isfinite(log_g)), np.isfinite(log_b)))

    invalid = np.logical_or(invalid, invalid_log)
    valid = np.logical_not(invalid)

    log_r[invalid] = 0
    log_g[invalid] = 0
    log_b[invalid] = 0

    u = log_g - log_r
    v = log_g - log_b

    weight = np.ones(u.shape)

    # set invalid pixels weight to 0
    weight[invalid] = 0

    weight_flat = weight[valid].flatten()
    u_flat = u[valid].flatten()
    v_flat = v[valid].flatten()

    # FFCC: wrap log uv!
    wrapped_u = np.mod(np.round((u_flat - starting_uv) / bin_size), num_bins)
    wrapped_v = np.mod(np.round((v_flat - starting_uv) / bin_size), num_bins)

    hist, xedges, yedges = np.histogram2d(wrapped_u, wrapped_v, num_bins, [[0, num_bins], [0, num_bins]], weights = weight_flat)

    hist = hist.astype(np.float)
    if normalization is None:
        div_hist = 1.0
    elif normalization == 'sum':
        div_hist = float(hist.sum())
    elif normalization == 'max':
        div_hist = float(hist.max())
    else:
        raise Exception('not a valid histogram normalization: ' + normalization)

    hist = hist / max(div_hist, 0.00001)

    if postprocess is not None:
        if postprocess == 'sqrt':
            hist = np.sqrt(hist)
        else:
            raise Exception('not a valid histogram postprocess: ' + postprocess)

    hist = hist.astype(np.float32)

    return hist, weight

def local_abs_dev(im, mask, min_intensity):
    original_type = im.dtype
    max_value = np.iinfo(im.dtype).max
    min_value = min_intensity*max_value

    im = im.astype(np.int32)
    r = im[:, :, 0]
    g = im[:, :, 1]
    b = im[:, :, 2]

    # ignore black pixels
    bigger_zero = np.logical_and(np.logical_and(r >= 1, g >= 1), b >= 1)
    # ignore pixels smaller than minimum
    bigger_min = np.logical_or(np.logical_or(r >= min_value, g >= min_value), b >= min_value)
    # ignore saturated pixels
    not_saturated = np.logical_and(np.logical_and(r < max_value, g < max_value), b < max_value)

    mask = np.logical_and(np.logical_and(bigger_zero, bigger_min), np.logical_and(not_saturated, mask>0))

    # ignore zeros
    mask = mask.astype(np.int32)

    pad_mask = np.pad(mask, ((1, 1), (1, 1)), 'constant', constant_values = 0)
    pad_r = np.pad(r, ((1, 1), (1, 1)), 'edge')
    pad_g = np.pad(g, ((1, 1), (1, 1)), 'edge')
    pad_b = np.pad(b, ((1, 1), (1, 1)), 'edge')
    pad = np.concatenate((np.expand_dims(pad_r, axis=2), np.expand_dims(pad_g, axis=2), np.expand_dims(pad_b, axis=2)), 2)

    output = np.zeros(im.shape)
    for c in range(im.shape[2]):
        cim = pad[:, :, c]
        res = (im.shape[0], im.shape[1])
        out = np.zeros(res, dtype=np.int32)
        n = np.zeros(res, dtype=np.int32)
        for i in [-1,0,1]:
            for j in [-1,0,1]:
                if i == 0 and j == 0:
                    continue

                im_shift = cim[i+1:im.shape[0]+i+1, j+1:im.shape[1]+j+1]
                mask_shift = pad_mask[i+1:im.shape[0]+i+1, j+1:im.shape[1]+j+1]

                curr_mask = mask * mask_shift
                out += curr_mask * np.abs(im_shift - im[:, :, c])
                n += curr_mask

        n[n == 0] = 1
        output[:, :, c] = out / n

    output = output.astype(original_type)
    return output

class LogUvHistogramWrap():
    def __init__(self, worker, num_bins = 64, bin_size = 0.03125,
                starting_uv = -0.4375, min_intensity = 0.0,
                normalization = "sum", postprocess = None):
        self._conf = {}
        self._conf['num_bins'] = num_bins
        self._conf['bin_size'] = bin_size
        self._conf['starting_uv'] = starting_uv
        self._conf['min_intensity'] = min_intensity
        self._conf['normalization'] = normalization
        self._conf['postprocess'] = postprocess

    def _ground_truth_pdf(self, illuminant, conf):
        num_bins = conf['num_bins']
        pdf = np.zeros((num_bins, num_bins)).astype(np.int)

        pdf[illuminant[0], illuminant[1]] = 1

        return pdf.astype(np.float32)

    def _log_uv_histogram_wrapped(self, im, mask, conf):
        return log_uv_histogram_wrapped(im, mask, conf)

    def _local_abs_dev(self, im, mask):
        min_intensity = self._conf['min_intensity']
        return local_abs_dev(im, mask, min_intensity)

    def __call__(self, input_dict):
        im = input_dict['rgb']
        mask = input_dict['mask']
        illuminant = None
        if 'illuminant' in input_dict:
            illuminant = input_dict['illuminant']

        hist_list = []
        weights_list = []
        hist_local_abs_dev_list = []
        for i in range(im.shape[0]):
            im_i = im[i, ...]
            mask_i = mask[i, ...]

            hist, weights = self._log_uv_histogram_wrapped(im_i, mask_i, self._conf)
            local_abs_dev = self._local_abs_dev(im_i, mask_i)
            # no weight for local abs dev image
            hist_local_abs_dev, _ = self._log_uv_histogram_wrapped(local_abs_dev, mask_i, self._conf)
            hist_list.append(hist[np.newaxis, ...])
            weights_list.append(weights[np.newaxis, ...])
            hist_local_abs_dev_list.append(hist_local_abs_dev[np.newaxis, ...])

        hist = np.concatenate(hist_list, 0)
        weights = np.concatenate(weights_list, 0)
        hist_local_abs_dev = np.concatenate(hist_local_abs_dev_list, 0)

        illuminant_warped = gt_pdf = None
        if illuminant is not None:
            illuminant, illuminant_warped = rgb_to_uv(illuminant, self._conf)
            gt_pdf = self._ground_truth_pdf(illuminant_warped, self._conf)

        input_dict['log_uv_histogram_wrapped'] = hist
        input_dict['log_uv_histogram_weights'] = weights
        input_dict['log_uv_histogram_wrapped_local_abs_dev'] = hist_local_abs_dev
        if illuminant is not None:
            input_dict['gt_pdf'] = gt_pdf
            input_dict['illuminant_log_uv'] = illuminant

        return input_dict
