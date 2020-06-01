import torch
import math
import numpy as np
import sklearn.cluster

from core.utils import normalize_illuminant

class KMeans():
    def __init__(self, conf, k, color_space = 'rgb'):
        self.k = k # number of clusters
        self.color_space = color_space
        # implemented color spaces
        implemented_cs = ['rgb', 'rg_bg', 'log_rg_bg']

        # check we use an implemented color space
        if self.color_space not in implemented_cs:
            raise Exception('Unkwnown color space: '+ str(color_space))

    def initialize(self, illuminants):
        if illuminants is None:
            # if no illuminants are provided, init with zeros
            clusters = torch.FloatTensor(np.zeros((self.k, 3))).unsqueeze(0)
            return clusters
        else:
            # convert illuminants to the desired color space
            illuminants = np.array(illuminants)
            if self.color_space == 'rg_bg':
                illuminants_new = np.zeros((illuminants.shape[0], 2))
                for i in range(illuminants.shape[0]):
                    illuminants_new[i, 0] = illuminants[i,0] / illuminants[i,1] # r / g
                    illuminants_new[i, 1] = illuminants[i,2] / illuminants[i,1] # b / g

                illuminants = illuminants_new
            elif self.color_space == 'log_rg_bg':
                illuminants_new = np.zeros((illuminants.shape[0], 2))
                for i in range(illuminants.shape[0]):
                    illuminants_new[i, 0] = math.log(illuminants[i,0]) - math.log(illuminants[i,1]) # log(r / g)
                    illuminants_new[i, 1] = math.log(illuminants[i,2]) - math.log(illuminants[i,1]) # log(b / g)

                illuminants = illuminants_new

            # run K-Means
            kmeans = sklearn.cluster.KMeans(n_clusters = self.k).fit(illuminants)

            # convert back to RGB color space
            if self.color_space == 'rgb':
                clusters = np.copy(kmeans.cluster_centers_)
                # normalize illuminants
                for i in range(clusters.shape[0]):
                    clusters[i, :] = normalize_illuminant(clusters[i, :])
            elif self.color_space == 'rg_bg':
                clusters_rg_bg = np.copy(kmeans.cluster_centers_)
                clusters = np.zeros((clusters_rg_bg.shape[0], 3))

                # convert to rgb
                for i in range(clusters.shape[0]):
                    rgb_vector = np.array([clusters_rg_bg[i, 0], 1.0, clusters_rg_bg[i, 1]])
                    clusters[i, :] = normalize_illuminant(rgb_vector)
            elif self.color_space == 'log_rg_bg':
                clusters_rg_bg = np.copy(kmeans.cluster_centers_)
                clusters = np.zeros((clusters_rg_bg.shape[0], 3))

                # convert to rgb
                for i in range(clusters.shape[0]):
                    rgb_vector = np.array([math.exp(clusters_rg_bg[i, 0]), 1.0, math.exp(clusters_rg_bg[i, 1])])
                    clusters[i, :] = normalize_illuminant(rgb_vector)

            # output: pytorch tensor
            clusters = torch.FloatTensor(clusters).unsqueeze(0)
            return clusters

    def run(self, image, clusters):
        # do nothing: no candidate tunning for each image
        return clusters
