import unittest
import numpy as np
from PCAfold import preprocess
from PCAfold import reduction
from PCAfold import analysis

class Reduction(unittest.TestCase):

    def test_reduction__plot_mode__allowed_calls(self):

        X = np.random.rand(100,5)

        try:
            pca_X = reduction.PCA(X, scaling='auto', n_components=2)
            plt_handle = reduction.plot_mode(pca_X.A[:,0])
            plt_handle.close()
        except Exception:
            self.assertTrue(False)

        try:
            pca_X = reduction.PCA(X, scaling='auto', n_components=2)
            plt_handle = reduction.plot_mode(pca_X.A[:,0], mode_name='PC', plot_absolute=True, rotate_label=True, bar_color='r', figure_size=(10,2), title='A', save_filename=None)
            plt_handle.close()
        except Exception:
            self.assertTrue(False)

# ------------------------------------------------------------------------------
