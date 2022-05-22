import unittest
import numpy as np
from PCAfold import preprocess
from PCAfold import reduction
from PCAfold import analysis

class Reduction(unittest.TestCase):

    def test_reduction__plot_eigenvalue_distribution__allowed_calls(self):

        X = np.random.rand(100,5)

        try:
            pca_X = reduction.PCA(X, scaling='auto', n_components=2)
            plt = reduction.plot_eigenvalue_distribution(pca_X.L, normalized=False, title=None, save_filename=None)
            plt.close()
        except Exception:
            self.assertTrue(False)

        try:
            pca_X = reduction.PCA(X, scaling='auto', n_components=2)
            plt = reduction.plot_eigenvalue_distribution(pca_X.L, normalized=True, title=None, save_filename=None)
            plt.close()
        except Exception:
            self.assertTrue(False)

        try:
            pca_X = reduction.PCA(X, scaling='auto', n_components=2)
            plt = reduction.plot_eigenvalue_distribution(pca_X.L, normalized=True, title='Title', save_filename=None)
            plt.close()
        except Exception:
            self.assertTrue(False)

# ------------------------------------------------------------------------------
