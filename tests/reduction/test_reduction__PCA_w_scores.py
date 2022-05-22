import unittest
import numpy as np
from PCAfold import preprocess
from PCAfold import reduction
from PCAfold import analysis

class Reduction(unittest.TestCase):

    def test_reduction__PCA_w_scores__allowed_calls(self):

        X = np.random.rand(100,10)

        pca = reduction.PCA(X, scaling='auto')

        try:
            w_scores = pca.w_scores(X)
        except Exception:
            self.assertTrue(False)

        try:
            pca.n_components = 5
            w_scores = pca.w_scores(X)
            (n_observations, n_w_scores) = np.shape(w_scores)
            self.assertTrue(n_w_scores == 5)
        except Exception:
            self.assertTrue(False)

        try:
            pca.n_components = 0
            w_scores = pca.w_scores(X)
            (n_observations, n_w_scores) = np.shape(w_scores)
            self.assertTrue(n_w_scores == 10)
        except Exception:
            self.assertTrue(False)

# ------------------------------------------------------------------------------

    def test_reduction__PCA_w_scores__not_allowed_calls(self):

        X = np.random.rand(20,4)
        X_2 = np.random.rand(20,3)
        X_3 = np.random.rand(20,5)

        pca = reduction.PCA(X, scaling='auto')

        with self.assertRaises(ValueError):
            u_scores = pca.w_scores(X_2)

        with self.assertRaises(ValueError):
            u_scores = pca.w_scores(X_3)

# ------------------------------------------------------------------------------
