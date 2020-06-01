import torch
import numpy as np
from sklearn import mixture

from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt

from core.utils import normalize_illuminant

class Gmm():
    def __init__(self, conf, k, n_components, color_space = 'rg_bg'):
        self.k = k # number of candidates
        self.n_components = n_components # number of gaussians
        self.color_space = color_space # color space
        implemented_cs = ['rgb', 'rg_bg']

        if self.color_space not in implemented_cs:
            raise Exception('Unkwnown color space: '+ str(color_space))

    # draw ellipse around mean
    def _draw_ellipse(self, position, covariance, ax=None, **kwargs):
        """Draw an ellipse with a given position and covariance"""
        ax = ax or plt.gca()

        # convert covariance to principal axes
        if covariance.shape == (2, 2):
            U, s, Vt = np.linalg.svd(covariance)
            angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
            width, height = 2 * np.sqrt(s)
        else:
            angle = 0
            width, height = 2 * np.sqrt(covariance)

        # draw the Ellipse
        for nsig in range(1, 4):
            ax.add_patch(Ellipse(position, nsig * width, nsig * height, angle, **kwargs))

    # plot GMM
    def _plot_gmm(self, gmm, X, label=False, ax=None):
        ax = ax or plt.gca()
        labels = gmm.fit(X).predict(X)
        if label:
            ax.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis', zorder=2)
        else:
            ax.scatter(X[:, 0], X[:, 1], s=40, zorder=2)
            ax.axis('equal')

        w_factor = 0.2 / gmm.weights_.max()
        for pos, covar, w in zip(gmm.means_, gmm.covars_, gmm.weights_):
            self._draw_ellipse(pos, covar, alpha=w * w_factor)

        plt.show()
        ax.figure.savefig('gmm.png', dpi=ax.figure.dpi)
        plt.close(ax.figure)

    def initialize(self, illuminants):
        # init to zero if no illuminants are provided
        if illuminants is None:
            clusters = torch.FloatTensor(np.zeros((self.k, 3))).unsqueeze(0)
            return clusters
        else:
            # convert to desired color space first
            illuminants = np.array(illuminants)
            if self.color_space == 'rg_bg':
                illuminants_new = np.zeros((illuminants.shape[0], 2))
                for i in range(illuminants.shape[0]):
                    illuminants_new[i, 0] = illuminants[i,0] / illuminants[i,1] # r / g
                    illuminants_new[i, 1] = illuminants[i,2] / illuminants[i,1] # b / g

                illuminants = illuminants_new

            # fit GMM
            g = mixture.GMM(n_components=self.n_components)
            g.fit(illuminants)
            self._plot_gmm(g, illuminants)

            # convert results to RGB
            if self.color_space == 'rgb':
                clusters = g.sample(self.k) # sample GMM
                # normalize illuminants
                for i in range(clusters.shape[0]):
                    clusters[i, :] = normalize_illuminant(clusters[i, :])
            elif self.color_space == 'rg_bg':
                clusters_rg_bg = g.sample(self.k) # sample GMM
                clusters = np.zeros((clusters_rg_bg.shape[0], 3))

                # convert to rgb
                for i in range(clusters.shape[0]):
                    rgb_vector = np.array([clusters_rg_bg[i, 0], 1.0, clusters_rg_bg[i, 1]])
                    clusters[i, :] = normalize_illuminant(rgb_vector)

            # pytorch candidates
            clusters = torch.FloatTensor(clusters).unsqueeze(0)
            return clusters

    def run(self, image, clusters):
        # do nothing: no candidate tunning for each image
        return clusters
