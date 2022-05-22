import unittest
import numpy as np
from PCAfold import preprocess
from PCAfold import reduction
from PCAfold import analysis

class Reduction(unittest.TestCase):

    def test_reduction__PCA_principal_variables__allowed_calls(self):

        X = np.random.rand(100,10)

        pca = reduction.PCA(X, scaling='auto')

        try:
            principal_variables_indices = pca.principal_variables(method='B2')
        except Exception:
            self.assertTrue(False)

        try:
            principal_variables_indices = pca.principal_variables(method='B4')
        except Exception:
            self.assertTrue(False)

        try:
            principal_variables_indices = pca.principal_variables(method='M2', x=X)
        except Exception:
            self.assertTrue(False)

# ------------------------------------------------------------------------------

    def test_reduction__PCA_principal_variables__not_allowed_calls(self):

        X = np.random.rand(100,10)

        pca = reduction.PCA(X, scaling='auto')

        with self.assertRaises(ValueError):
            pca.principal_variables(method='M2')
        with self.assertRaises(ValueError):
            pca.principal_variables(method='Method')

# ------------------------------------------------------------------------------
