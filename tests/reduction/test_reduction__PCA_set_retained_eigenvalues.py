import unittest
import numpy as np
from PCAfold import preprocess
from PCAfold import reduction
from PCAfold import analysis

class Reduction(unittest.TestCase):

    def test_reduction__PCA_set_retained_eigenvalues__allowed_calls(self):

        X = np.random.rand(100,10)

        pca = reduction.PCA(X, scaling='auto')

        try:
            pca_new = pca.set_retained_eigenvalues(method='TOTAL VARIANCE', option=0.5)
        except Exception:
            self.assertTrue(False)

        try:
            pca_new = pca.set_retained_eigenvalues(method='INDIVIDUAL VARIANCE', option=0.5)
        except Exception:
            self.assertTrue(False)

        try:
            pca_new = pca.set_retained_eigenvalues(method='BROKEN STICK')
        except Exception:
            self.assertTrue(False)

# ------------------------------------------------------------------------------

    def test_reduction__PCA_set_retained_eigenvalues__not_allowed_calls(self):

        X = np.random.rand(100,10)

        pca = reduction.PCA(X, scaling='auto')

        with self.assertRaises(ValueError):
            pca.set_retained_eigenvalues(method='Method')
        with self.assertRaises(ValueError):
            pca.set_retained_eigenvalues(method='TOTAL VARIANCE', option=1.1)
        with self.assertRaises(ValueError):
            pca.set_retained_eigenvalues(method='TOTAL VARIANCE', option=-0.1)
        with self.assertRaises(ValueError):
            pca.set_retained_eigenvalues(method='INDIVIDUAL VARIANCE', option=1.1)
        with self.assertRaises(ValueError):
            pca.set_retained_eigenvalues(method='INDIVIDUAL VARIANCE', option=-0.1)

# ------------------------------------------------------------------------------
