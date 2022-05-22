import unittest
import numpy as np
from PCAfold import preprocess
from PCAfold import reduction
from PCAfold import analysis

class Reduction(unittest.TestCase):

    def test_reduction__PCA_reconstruct__allowed_calls(self):

        X = np.random.rand(100,10)

        try:
            pca_X = reduction.PCA(X, scaling='auto')
            principal_components = pca_X.transform(X)
            X_rec = pca_X.reconstruct(principal_components)
        except Exception:
            self.assertTrue(False)

        try:
            pca_X = reduction.PCA(X, scaling='auto', n_components=5)
            principal_components = pca_X.transform(X)
            X_rec = pca_X.reconstruct(principal_components)
        except Exception:
            self.assertTrue(False)

        try:
            pca_X = reduction.PCA(X, scaling='auto', n_components=2)
            principal_components = pca_X.transform(X)
            X_rec = pca_X.reconstruct(principal_components)
        except Exception:
            self.assertTrue(False)

        try:
            pca_X = reduction.PCA(X, scaling='auto', n_components=2, nocenter=True)
            principal_components = pca_X.transform(X)
            X_rec = pca_X.reconstruct(principal_components)
        except Exception:
            self.assertTrue(False)

        try:
            pca_X = reduction.PCA(X, scaling='auto', n_components=2)
            principal_components = pca_X.transform(X, nocenter=True)
            X_rec = pca_X.reconstruct(principal_components, nocenter=True)
        except Exception:
            self.assertTrue(False)

        try:
            pca_X = reduction.PCA(X, scaling='auto', n_components=2)
            X_2 = np.random.rand(200,10)
            principal_components = pca_X.transform(X_2, nocenter=True)
            X_rec = pca_X.reconstruct(principal_components, nocenter=True)
        except Exception:
            self.assertTrue(False)

        try:
            pca_X = reduction.PCA(X, scaling='auto', n_components=2)
            X_2 = np.random.rand(200,10)
            principal_components = pca_X.transform(X_2)
            X_rec = pca_X.reconstruct(principal_components)
        except Exception:
            self.assertTrue(False)

# ------------------------------------------------------------------------------

    def test_reduction__PCA_reconstruct__not_allowed_calls(self):

        X = np.random.rand(100,10)
        fake_PCs = np.random.rand(100,11)

        pca = reduction.PCA(X, scaling='auto')
        with self.assertRaises(ValueError):
            X_rec = pca.reconstruct(fake_PCs)

        pca = reduction.PCA(X, scaling='auto', n_components=4)
        with self.assertRaises(ValueError):
            X_rec = pca.reconstruct(fake_PCs)

# ------------------------------------------------------------------------------
