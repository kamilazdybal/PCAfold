import unittest
import numpy as np
from PCAfold import preprocess
from PCAfold import reduction
from PCAfold import analysis

class Analysis(unittest.TestCase):

    def test_analysis__coefficient_of_determination__allowed_calls(self):

        X = np.random.rand(100,5)
        pca_X = reduction.PCA(X, scaling='auto', n_components=2)
        X_rec = pca_X.reconstruct(pca_X.transform(X))

        try:
            r2 = analysis.coefficient_of_determination(X[:,0], X_rec[:,0])
        except Exception:
            self.assertTrue(False)

# ------------------------------------------------------------------------------

    def test_analysis__coefficient_of_determination__not_allowed_calls(self):

        X = np.random.rand(100,5)

        with self.assertRaises(ValueError):
            plt = analysis.coefficient_of_determination(X[:,0:2], X[:,1])

        with self.assertRaises(ValueError):
            plt = analysis.coefficient_of_determination(X[:,0], X[:,1:3])

        with self.assertRaises(ValueError):
            plt = analysis.coefficient_of_determination([1,2,3], X[:,1])

        with self.assertRaises(ValueError):
            plt = analysis.coefficient_of_determination(X[:,0], [1,2,3])

# ------------------------------------------------------------------------------

    def test_analysis__coefficient_of_determination__computation(self):

        X = np.random.rand(100,5)
        pca_X = reduction.PCA(X, scaling='auto', n_components=5)
        X_rec = pca_X.reconstruct(pca_X.transform(X))

        try:
            r2 = analysis.coefficient_of_determination(X[:,0], X_rec[:,0])
            self.assertTrue(r2==1)
        except Exception:
            self.assertTrue(False)

        try:
            r2 = analysis.coefficient_of_determination(X[:,0], X[:,0])
            self.assertTrue(r2==1)
        except Exception:
            self.assertTrue(False)

        pca_X = reduction.PCA(X, scaling='auto', n_components=4)
        X_rec = pca_X.reconstruct(pca_X.transform(X))

        try:
            r2 = analysis.coefficient_of_determination(X[:,0], X_rec[:,0])
            self.assertTrue(r2<1)
        except Exception:
            self.assertTrue(False)

        try:
            r2 = analysis.coefficient_of_determination(X[:,0], X[:,1])
            self.assertTrue(r2<1)
        except Exception:
            self.assertTrue(False)

        obs = np.array([0., 1., 2.])
        offset = 1.e-1
        expected = 1. - 0.5*(3 * offset**2)
        self.assertTrue(np.abs(analysis.coefficient_of_determination(obs, obs + offset) - expected) < 1.e-6)
        self.assertTrue(np.abs(analysis.coefficient_of_determination(obs, obs) - 1.0) < 1.e-6)

# ------------------------------------------------------------------------------
