import unittest
import numpy as np
from PCAfold import preprocess
from PCAfold import reduction
from PCAfold import analysis

class Reduction(unittest.TestCase):

    def test_reduction__PCA_r2_convergence__allowed_calls(self):

        X = np.random.rand(100,3)

        pca = reduction.PCA(X, scaling='auto')

        try:
            r2 = pca.r2_convergence(X, 3, variable_names=[], print_width=10, verbose=False, save_filename=None)
        except Exception:
            self.assertTrue(False)

        try:
            r2 = pca.r2_convergence(X, 3, variable_names=['a', 'b', 'c'], print_width=10, verbose=False, save_filename=None)
        except Exception:
            self.assertTrue(False)

        try:
            r2 = pca.r2_convergence(X, 1, variable_names=[], print_width=10, verbose=False, save_filename=None)
        except Exception:
            self.assertTrue(False)

        try:
            r2 = pca.r2_convergence(X, 1, variable_names=['a', 'b', 'c'], print_width=10, verbose=False, save_filename=None)
        except Exception:
            self.assertTrue(False)

# ------------------------------------------------------------------------------

    def test_reduction__PCA_r2_convergence__not_allowed_calls(self):

        pass

# ------------------------------------------------------------------------------
