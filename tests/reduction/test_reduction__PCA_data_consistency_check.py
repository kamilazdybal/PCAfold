import unittest
import numpy as np
from PCAfold import preprocess
from PCAfold import reduction
from PCAfold import analysis

class Reduction(unittest.TestCase):

    def test_reduction__PCA_data_consistency_check__allowed_calls(self):

        X = np.random.rand(100,20)
        pca_X = reduction.PCA(X, scaling='auto', n_components=10)

        try:
            X_1 = np.random.rand(50,20)
            is_consistent = pca_X.data_consistency_check(X_1)
            self.assertTrue(is_consistent==True)
        except Exception:
            self.assertTrue(False)

        try:
            X_2 = np.random.rand(100,10)
            is_consistent = pca_X.data_consistency_check(X_2)
            self.assertTrue(is_consistent==False)
        except Exception:
            self.assertTrue(False)

        X_3 = np.random.rand(100,10)
        with self.assertRaises(ValueError):
            is_consistent = pca_X.data_consistency_check(X_3, errors_are_fatal=True)

        try:
            X_4 = np.random.rand(80,20)
            is_consistent = pca_X.data_consistency_check(X_4, errors_are_fatal=True)
            self.assertTrue(is_consistent==True)
        except Exception:
            self.assertTrue(False)

# ------------------------------------------------------------------------------

    def test_reduction__PCA_data_consistency_check__not_allowed_calls(self):

        X = np.random.rand(100,20)
        pca_X = reduction.PCA(X, scaling='auto', n_components=10)

        with self.assertRaises(ValueError):
            is_consistent = pca_X.data_consistency_check(X, errors_are_fatal=1)

        with self.assertRaises(ValueError):
            is_consistent = pca_X.data_consistency_check(X, errors_are_fatal=0)

# ------------------------------------------------------------------------------
