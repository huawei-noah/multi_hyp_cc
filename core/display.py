#Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

#This program is free software; you can redistribute it and/or modify it under the terms of the BSD 0-Clause License.

#This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD 0-Clause License for more details.

import numpy as np
import torch
import os
import cv2
from core.utils import *

# All the information of a predicted illuminant
class ImageResult():
    def __init__(self, path, error, prediction, gt):
        self.path = path # image path
        self.error = error # angular error
        self.prediction = prediction # RGB illuminant
        self.gt = gt # GT RGB illuminant

class Display():
    def __init__(self, conf):
        self._conf = conf

    # save output to disk
    def save_output(self, data, output, err, dataset, results_dir, save_full_res):
        output_np = output['illuminant'].data.cpu().numpy()
        illuminant_np = None
        if 'illuminant' in data:
            illuminant_np = data['illuminant'].cpu().numpy()

        confidence_np = None
        if 'confidence' in output:
            confidence_np = output['confidence'].data.cpu().numpy()

        bin_probability_preaffine_np = None
        if 'bin_probability_preaffine' in output:
            bin_probability_preaffine_np = output['bin_probability_preaffine'].data.cpu().numpy()

        bin_probability_np = None
        if 'bin_probability2' in output:
            bin_probability_np = output['bin_probability2'].data.cpu().numpy()
        candidates_np = None
        if 'candidates' in output:
            candidates_np = output['candidates'].data.cpu().numpy()

        bias2_np = None
        if 'bias2' in output:
            bias2_np = output['bias2'].squeeze().data.cpu().numpy()
        gain2_np = None
        if 'gain2' in output:
            gain2_np = output['gain2'].squeeze().data.cpu().numpy()

        illuminants_np = None
        if 'illuminants' in output:
            illuminants_np = output['illuminants'].squeeze().data.cpu().numpy()

        info = self.get_images(data, output)

        res = []
        for j in range(output_np.shape[0]):
            illuminant = output_np[j, :]
            confidence = None
            if confidence_np is not None:
                confidence = confidence_np[j]
            bin_probability = None
            if bin_probability_np is not None:
                bin_probability = bin_probability_np[j, :]

            bin_probability_preaffine = None
            if bin_probability_preaffine_np is not None:
                bin_probability_preaffine = bin_probability_preaffine_np[j, :]

            candidates = None
            if candidates_np is not None:
                if candidates_np.shape[0] == 1:
                    candidates = candidates_np[0, ...]
                else:
                    candidates = candidates_np[j, ...]

            if bias2_np is None:
                bias2 = None
            else:
                if len(bias2_np.shape) == 1:
                    bias2 = bias2_np
                else:
                    bias2 = bias2_np[j, ...]

            if gain2_np is None:
                gain2 = None
            else:
                if len(gain2_np.shape) == 1:
                    gain2 = gain2_np
                else:
                    gain2 = gain2_np[j, ...]
            illuminants = None
            if illuminants_np is not None:
                illuminants = illuminants_np[j, ...]

            if 'illuminant' in data:
                error = err[j]
                gt_illuminant = illuminant_np[j, :]
            else:
                error = None
                gt_illuminant = None

            path_name = data['path'][j]
            save_name = path_name
            if 'save_name' in data:
                save_name = data['save_name'][j]

            r = ImageResult(save_name, error, illuminant, gt_illuminant)
            res.append(r)

            self._save_output(path_name, save_name, info, j, gt_illuminant,
                            illuminant, error, dataset, results_dir,
                            save_full_res, confidence, bin_probability,
                            candidates, bias2, gain2, illuminants,
                            bin_probability_preaffine)

        return res

    # save image to disk
    def _save_im(self, name, info, key, batch):
        if key in info:
            num_img = info[key].shape[1]
            for i in range(num_img):
                if len(info[key].shape) == 5:
                    im = axis_numpy(info[key][batch, i, :, :, :])
                    im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
                else:
                    im = info[key][batch, i, :, :]
                img_name = name + '.jpg'
                if num_img > 1:
                    img_name = name + '_' + str(i) + '.jpg'
                cv2.imwrite(img_name, im)

    # save output probabilities to text file
    def _save_candidate_prob(self, name, candidates, prob):
        with open(name, 'w') as f:
            for i in range(candidates.shape[0]):
                f.write(str(candidates[i,0])+','+str(candidates[i,1])+','+str(candidates[i,2])+' '+str(prob[i])+'\n')

    # plot probabilities of candidates + GT + inference
    def _plot_prob(self, name, prob, candidates, gt_illuminant=None, illuminant=None):
        import matplotlib as mpl
        mpl.use('Agg')
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        import matplotlib.cm as cm

        r = prob.max() - prob.min()
        if r == 0:
            r = 1
        prob_norm = (prob - prob.min()) / r
        colors = cm.jet(prob_norm)

        fig = plt.figure()
        ax = plt.axes()

        for i in range(candidates.shape[0]):
            c = colors[i,:]
            c = c[np.newaxis,:]
            ax.scatter(candidates[i,0] / candidates[i,1], candidates[i,2] / candidates[i,1], marker='^', c=c)

        if gt_illuminant is not None:
            ax.scatter(gt_illuminant[0] / gt_illuminant[1], gt_illuminant[2] / gt_illuminant[1], marker='o', c='tab:green')

        if illuminant is not None:
            ax.scatter(illuminant[0] / illuminant[1], illuminant[2] / illuminant[1], marker='o', c='tab:blue')

        ax.set_xlabel('r/g')
        ax.set_ylabel('b/g')

        cax, _ = mpl.colorbar.make_axes(ax)
        cmap = mpl.cm.get_cmap('jet')
        normalize = mpl.colors.Normalize(vmin=prob.min(), vmax=prob.max())
        cbar = mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=normalize)

        fig.savefig(name, dpi=fig.dpi)
        plt.close(fig)

    # save output of the method
    def _save_output(self, orig_name, save_name, info, j, gt_illuminant, illuminant,
                    error, dataset, results_dir, save_full_res, confidence,
                    bin_probability, candidates, bias, gain, illuminants,
                    bin_probability_preaffine):
        name = os.path.join(results_dir, os.path.basename(save_name)[:-4])

        self._save_im(name+'_input', info, 'Input Image', j)
        self._save_im(name+'_prediction', info, 'Predicted Image', j)
        self._save_im(name+'_true', info, 'GT Image', j)
        self._save_im(name+'_loguv_histogram', info, 'Log UV Histogram', j)
        self._save_im(name+'_loguv_binprobability', info, 'Log UV Bin Probability', j)
        self._save_im(name+'_loguv_gtpdf', info, 'Log UV GT PDF', j)
        self._save_im(name+'_conf', info, 'Confidence Map', j)
        self._save_im(name+'_log_uv_histogram_weights', info, 'Log UV Weights', j)

        if candidates is not None:
            self._plot_prob(name+'_plot.jpg', bin_probability, candidates, gt_illuminant, illuminant)
            if bin_probability_preaffine is not None:
                self._plot_prob(name+'_plot_preaffine.jpg', bin_probability_preaffine, candidates, gt_illuminant, illuminant)

        if bias is not None:
            self._plot_prob(name+'_plot_bias.jpg', bias, candidates)

        if gain is not None:
            self._plot_prob(name+'_plot_gain.jpg', gain, candidates)

        # save full resolution image
        if save_full_res:
            im, mask, sensor = dataset.get_rgb_by_path(orig_name)
            num_img = im.shape[0]
            for i in range(num_img):
                im_i = im[i, ...]
                mask_i = mask[i, ...]
                sensor_i = sensor[i]
                im_i = blacklevel_saturation_correct(im_i, sensor_i, saturation_scale = 1.0)
                im_i = im_i.astype(np.float) / np.iinfo(im_i.dtype).max
                im_i = convert_for_display(im_i, illuminant, sensor_i.ccm.numpy())
                im_i = (255*im_i).astype(np.uint8)
                im_i = cv2.cvtColor(im_i, cv2.COLOR_RGB2BGR)
                img_name = name + '_prediction_fullres.png'
                if num_img > 1:
                    img_name = name + '_' + str(i) + '_prediction_fullres.png'
                cv2.imwrite(img_name, im_i)

        illuminant = illuminant.squeeze()
        with open(name+'_illum.txt', 'w') as f:
            for c in illuminant:
                f.write(str(c)+'\n')

        if confidence is not None:
            with open(name+'_confidence.txt', 'w') as f:
                f.write(str(confidence)+'\n')

        # Available ground truth
        if gt_illuminant is not None:
            with open(name+'_error.txt', 'w') as f:
                f.write(str(error)+'\n')

            with open(name+'_illum_true.txt', 'w') as f:
                for c in gt_illuminant:
                    f.write(str(c)+'\n')

        if bin_probability is not None:
            self._save_candidate_prob(name+'_candidates.txt', candidates, bin_probability)

        if illuminants is not None:
            #max_ae = 0
            #for i in range(illuminants.shape[0]):
            #    ae = angular_error_degress_np(illuminant, illuminants[i, :])
            #    max_ae = max(max_ae, ae)
            covar = np.cov(illuminants, rowvar=False)
            variance = np.linalg.det(covar)

            with open(name+'_variance.txt', 'w') as f:
                f.write(str(variance)+'\n')

    # generate all output images
    def get_images(self, data, output):
        input_np = data['rgb'].cpu().numpy()
        log_uv_histogram = None

        if 'log_uv_histogram' in data:
            log_uv_histogram = data['log_uv_histogram']
        elif 'log_uv_histogram_wrapped' in data:
            log_uv_histogram = data['log_uv_histogram_wrapped']

        if 'log_uv_histogram' in output:
            log_uv_histogram = output['log_uv_histogram'].data

        if log_uv_histogram is not None:
            log_uv_histogram = log_uv_histogram.squeeze(2).cpu().numpy()

        gt_pdf = None
        if 'gt_pdf' in data:
            gt_pdf = data['gt_pdf'].cpu().numpy()
            gt_pdf = (255*gt_pdf).astype(np.uint8)
            gt_pdf = np.stack((gt_pdf,)*3, axis=1)

        illuminant_np = output['illuminant'].squeeze(-1).squeeze(-1).data.cpu().numpy()
        conf_np = None
        if 'conf' in output:
            conf_np = output['conf'].data.cpu().numpy()
        illuminant_weight = None
        if 'illuminant_weight' in output:
            illuminant_weight = output['illuminant_weight'].data.cpu().numpy()
        log_uv_histogram_weights_np = None
        if 'log_uv_histogram_weights' in data:
            log_uv_histogram_weights_np = data['log_uv_histogram_weights'].data.cpu().numpy()
        bin_probability = None
        if 'bin_probability' in output:
            bin_probability = output['bin_probability'].data.cpu().numpy()
            bin_probability = bin_probability.squeeze(1)
            bin_probability = np.stack((bin_probability,)*3, axis=1)
        F = None
        if 'F_visualization' in output:
            F = (output['F_visualization'].data.cpu().numpy())
            F = (255*F).astype(np.uint8)
        F2 = None
        if 'F2_visualization' in output:
            F2 = (output['F2_visualization'].data.cpu().numpy())
            F2 = (255*F2).astype(np.uint8)
        F_lab = None
        if 'F_lab_visualization' in output:
            F_lab = (output['F_lab_visualization'].data.cpu().numpy())
            F_lab = (255*F_lab).astype(np.uint8)
        bias = None
        if 'bias' in output:
            bias = (output['bias'].data.cpu().numpy())
            bias = (255*bias).astype(np.uint8)
        gain = None
        if 'gain' in output:
            gain = (output['gain'].data.cpu().numpy())
            gain = (255*gain).astype(np.uint8)

        gt_im_np = None
        if 'illuminant' in data:
            gt_illuminant_np = data['illuminant'].cpu().numpy()
            gt_im_np = np.zeros(input_np.shape)

        mu_idx_np = None
        sigma_np = None
        if 'mu_idx' in output and 'sigma' in output:
            mu_idx_np = output['mu_idx'].data.cpu().numpy()
            sigma_np = output['sigma'].data.cpu().numpy()

        predicted_im_np = np.zeros(input_np.shape)

        for id in range(input_np.shape[0]):
            if bin_probability is not None:
                bin_probability[id, :, :, :] = (bin_probability[id, :, :, :] / bin_probability[id, :, :, :].max())
                bin_probability[id, :, :, :] = (255*bin_probability[id, :, :, :])

            if conf_np is not None:
                conf_np[id, ...] = conf_np[id, ...] / conf_np[id, ...].max()

            if 'mu_idx' in output and 'sigma' in output:
                max_bin = self._conf['log_uv_warp_histogram']['num_bins'] - 1
                u = min(max_bin, int(mu_idx_np[id, 0]))
                v = min(max_bin, int(mu_idx_np[id, 1]))
                if gt_pdf is not None:
                    gt_pdf[id, 1, u, v] = 255
                bin_probability[id, 0, u, v] = 0
                bin_probability[id, 1, u, v] = 255
                bin_probability[id, 2, u, v] = 0

                bin_size = self._conf['log_uv_warp_histogram']['bin_size']
                sigma = sigma_np[id, :, :]/(bin_size*bin_size) # B x 2 x 2
                eigenvalues, eigenvectors = np.linalg.eig(sigma)
                angle = math.atan2(eigenvectors[1, 0], eigenvectors[0, 0])
                if angle < 0:
                    angle += 2*math.pi

                angle = angle*180.0/math.pi

                chisquare_val = 2.4477 # 95% confidence interval
                axis_len = (round(chisquare_val*math.sqrt(eigenvalues[1])), round(chisquare_val*math.sqrt(eigenvalues[0])))
                mean = (v, u)

                bin_prob_id = np.copy(axis_numpy(bin_probability[id, :, :, :]), order='C')
                cv2.ellipse(bin_prob_id, mean, axis_len, -angle, 0.0, 360.0, (0, 255, 0), 1)
                bin_probability[id, :, :, :] = axis_pytorch(bin_prob_id)
                if gt_pdf is not None:
                    gt_pdf_id = np.copy(axis_numpy(gt_pdf[id, :, :, :]), order='C')
                    cv2.ellipse(gt_pdf_id, mean, axis_len, -angle, 0.0, 360.0, (0, 255, 0), 1)
                    gt_pdf[id, :, :, :] = axis_pytorch(gt_pdf_id)

            for n in range(input_np.shape[1]):
                if log_uv_histogram is not None:
                    log_uv_histogram[id, n, :, :] = (255*(log_uv_histogram[id, n, :, :]/log_uv_histogram[id, n, :, :].max()))

                input = axis_numpy(input_np[id, n, :, :, :].squeeze())
                ccm = data['sensor'][n]['ccm'].cpu().numpy()
                ccm_im = ccm[id, :, :].squeeze()
                predicted_im_np[id, n, :, :, :] = axis_pytorch(convert_for_display(input, illuminant_np[id, :].squeeze(), ccm_im))
                if 'illuminant' in data:
                    gt_im_np[id, n, :, :, :] = axis_pytorch(convert_for_display(input, gt_illuminant_np[id, :].squeeze(), ccm_im))

                input_np[id, n, :, :, :] = axis_pytorch(convert_for_display(input, None, ccm_im))

        info = { 'Input Image': (255*input_np).astype(np.uint8), 'Predicted Image': (255*predicted_im_np).astype(np.uint8) }
        if log_uv_histogram is not None:
            info['Log UV Histogram'] = log_uv_histogram.astype(np.uint8)
        if conf_np is not None:
            info['Confidence Map'] = (255*conf_np).astype(np.uint8)
        if illuminant_weight is not None:
            info['Illuminant Weight'] = (255*illuminant_weight).astype(np.uint8)
        if log_uv_histogram_weights_np is not None:
            info['Log UV Weights'] = (255*log_uv_histogram_weights_np).astype(np.uint8)
        if bin_probability is not None:
            info['Log UV Bin Probability'] = bin_probability.astype(np.uint8)
        if gt_pdf is not None:
            info['Log UV GT PDF'] = gt_pdf
        if F is not None:
            info['F'] = F[np.newaxis, ...]
        if F2 is not None:
            info['F2'] = F2[np.newaxis, ...]
        if F_lab is not None:
            info['F_lab'] = F_lab[np.newaxis, ...]
        if bias is not None:
            if len(bias.shape) == 3:
                info['Bias'] = bias[np.newaxis, ...]
            else:
                for l in range(bias.shape[0]):
                    info['Bias '+str(l)] = bias[l, :, :]
                    info['Bias '+str(l)] = info['bias_'+str(l)][np.newaxis, ...]
        if gain is not None:
            if len(gain.shape) == 3:
                info['Gain'] = gain[np.newaxis, ...]
            else:
                for l in range(gain.shape[0]):
                    info['Gain '+str(l)] = gain[l, :, :]
                    info['Gain '+str(l)] = info['gain_'+str(l)][np.newaxis, ...]

        if gt_im_np is not None:
            info['GT Image'] = (255*gt_im_np).astype(np.uint8)

        return info
