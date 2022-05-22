import unittest
import numpy as np
from PCAfold import preprocess
from PCAfold import reduction
from PCAfold import analysis

class Reduction(unittest.TestCase):

    def test_reduction__plot_eigenvectors__allowed_calls(self):

        X = np.random.rand(100,5)

        try:
            pca_X = reduction.PCA(X, scaling='auto', n_components=2)
            plts = reduction.plot_eigenvectors(pca_X.A, eigenvectors_indices=[], variable_names=[], plot_absolute=False, bar_color=None, title=None, save_filename=None)
            for i in range(0, len(plts)):
                plts[i].close()
        except Exception:
            self.assertTrue(False)

        try:
            pca_X = reduction.PCA(X, scaling='auto', n_components=2)
            plts = reduction.plot_eigenvectors(pca_X.A[:,0], eigenvectors_indices=[], variable_names=[], plot_absolute=False, bar_color=None, title=None, save_filename=None)
            for i in range(0, len(plts)):
                plts[i].close()
        except Exception:
            self.assertTrue(False)

        try:
            pca_X = reduction.PCA(X, scaling='auto', n_components=2)
            plts = reduction.plot_eigenvectors(pca_X.A[:,2:4], eigenvectors_indices=[], variable_names=[], plot_absolute=False, bar_color=None, title=None, save_filename=None)
            for i in range(0, len(plts)):
                plts[i].close()
        except Exception:
            self.assertTrue(False)

        try:
            pca_X = reduction.PCA(X, scaling='auto', n_components=2)
            plts = reduction.plot_eigenvectors(pca_X.A[:,0], eigenvectors_indices=[0], variable_names=['a', 'b', 'c', 'd', 'e'], plot_absolute=True, bar_color='r', title='Title', save_filename=None)
            for i in range(0, len(plts)):
                plts[i].close()
        except Exception:
            self.assertTrue(False)

# ------------------------------------------------------------------------------
