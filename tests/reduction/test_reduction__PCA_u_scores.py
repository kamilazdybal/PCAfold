import unittest
import numpy as np
from PCAfold import preprocess
from PCAfold import reduction
from PCAfold import analysis

class Reduction(unittest.TestCase):

    def test_reduction__PCA_u_scores__allowed_calls(self):

        X = np.random.rand(100,10)

        try:
            pca = reduction.PCA(X, scaling='auto')
            u_scores = pca.u_scores(X)
            (n_obs, n_u_scores) = np.shape(u_scores)
            self.assertTrue(n_u_scores==10)
        except Exception:
            self.assertTrue(False)

        try:
            pca = reduction.PCA(X, scaling='auto', n_components=4)
            u_scores = pca.u_scores(X)
            (n_obs, n_u_scores) = np.shape(u_scores)
            self.assertTrue(n_u_scores==4)
        except Exception:
            self.assertTrue(False)

        try:
            pca = reduction.PCA(X, scaling='auto', n_components=1)
            u_scores = pca.u_scores(X)
            (n_obs, n_u_scores) = np.shape(u_scores)
            self.assertTrue(n_u_scores==1)
        except Exception:
            self.assertTrue(False)

        try:
            pca = reduction.PCA(X, scaling='pareto', n_components=10)
            u_scores = pca.u_scores(X)
            (n_obs, n_u_scores) = np.shape(u_scores)
            self.assertTrue(n_u_scores==10)
        except Exception:
            self.assertTrue(False)

        try:
            pca = reduction.PCA(X, scaling='auto', n_components=4)
            X_new = np.random.rand(50,10)
            u_scores = pca.u_scores(X_new)
            (n_obs, n_u_scores) = np.shape(u_scores)
            self.assertTrue(n_u_scores==4)
        except Exception:
            self.assertTrue(False)

# ------------------------------------------------------------------------------

    def test_reduction__PCA_u_scores__not_allowed_calls(self):

        X = np.random.rand(20,4)
        X_2 = np.random.rand(20,3)
        X_3 = np.random.rand(20,5)

        pca = reduction.PCA(X, scaling='auto')

        with self.assertRaises(ValueError):
            u_scores = pca.u_scores(X_2)

        with self.assertRaises(ValueError):
            u_scores = pca.u_scores(X_3)

# ------------------------------------------------------------------------------
