import torch
import numpy as np

from core.utils import normalize_illuminant

# uniform sampling: get max and min in r/g, b/g
# and sample uniformly
class Uniform():
    def __init__(self, conf, k, color_space = 'rg_bg'):
        self.k = k
        self.color_space = color_space
        implemented_cs = ['rg_bg']

        if self.color_space not in implemented_cs:
            raise Exception('Unkwnown color space: '+ str(color_space))

    def initialize(self, illuminants):
        if illuminants is None:
            clusters = torch.FloatTensor(np.zeros((self.k*self.k, 3))).unsqueeze(0)
            return clusters
        else:
            illuminants = np.array(illuminants)
            min_rg = 100
            max_rg = 0
            min_bg = 100
            max_bg = 0
            for i in range(illuminants.shape[0]):
                rg = illuminants[i,0] / illuminants[i,1] # r / g
                bg = illuminants[i,2] / illuminants[i,1] # b / g

                min_rg = min(min_rg, rg)
                max_rg = max(max_rg, rg)

                min_bg = min(min_bg, bg)
                max_bg = max(max_bg, bg)

            real_k = self.k*self.k
            clusters_rg_bg = np.zeros((real_k, 2))

            step_rg = (max_rg - min_rg) / self.k
            step_bg = (max_bg - min_bg) / self.k

            for i in range(self.k):
                rg = min_rg + step_rg*i
                for j in range(self.k):
                    bg = min_bg + step_bg*j
                    rg_bg = np.array([rg, bg])
                    #print('i: '+str(i)+' j: '+str(j)+' '+str(rg_bg))
                    clusters_rg_bg[i*self.k+j, :] = rg_bg

            clusters = np.zeros((clusters_rg_bg.shape[0], 3))

            # convert to rgb
            for i in range(clusters.shape[0]):
                rgb_vector = np.array([clusters_rg_bg[i, 0], 1.0, clusters_rg_bg[i, 1]])
                clusters[i, :] = normalize_illuminant(rgb_vector)

            clusters = torch.FloatTensor(clusters).unsqueeze(0)
            return clusters

    def run(self, image, clusters):
        return clusters
