import unittest
import numpy as np
from PCAfold import preprocess
from PCAfold import reduction
from PCAfold import analysis

class Reduction(unittest.TestCase):

    def test_reduction__plot_cumulative_variance__allowed_calls(self):

        X = np.random.rand(100,5)

        try:
            pca_X = reduction.PCA(X, scaling='auto', n_components=2)
            plt = reduction.plot_cumulative_variance(pca_X.L, n_components=0, title=None, save_filename=None)
            plt.close()
        except Exception:
            self.assertTrue(False)

        try:
            pca_X = reduction.PCA(X, scaling='auto', n_components=2)
            plt = reduction.plot_cumulative_variance(pca_X.L, n_components=2, title=None, save_filename=None)
            plt.close()
        except Exception:
            self.assertTrue(False)

        try:
            pca_X = reduction.PCA(X, scaling='auto', n_components=2)
            plt = reduction.plot_cumulative_variance(pca_X.L, n_components=3, title='Title', save_filename=None)
            plt.close()
        except Exception:
            self.assertTrue(False)

# ------------------------------------------------------------------------------
