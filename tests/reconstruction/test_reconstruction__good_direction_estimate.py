import unittest
import numpy as np
from PCAfold import preprocess
from PCAfold import reduction
from PCAfold import reconstruction

class Reconstruction(unittest.TestCase):

    def test_reconstruction__good_direction_estimate__allowed_calls(self):

        X = np.random.rand(100,5)
        pca_X = reduction.PCA(X, scaling='auto', n_components=2)
        X_rec = pca_X.reconstruct(pca_X.transform(X))

        try:
            (GD, GDE) = reconstruction.good_direction_estimate(X, X_rec)
        except Exception:
            self.assertTrue(False)

        try:
            (GD, GDE) = reconstruction.good_direction_estimate(X[:,0:2], X_rec[:,0:2])
        except Exception:
            self.assertTrue(False)

# ------------------------------------------------------------------------------

    def test_reconstruction__good_direction_estimate__not_allowed_calls(self):

        X = np.random.rand(100,5)
        pca_X = reduction.PCA(X, scaling='auto', n_components=2)
        X_rec = pca_X.reconstruct(pca_X.transform(X))

        with self.assertRaises(ValueError):
            (GD, GDE) = reconstruction.good_direction_estimate(X[:,0], X_rec[:,0:2])

        with self.assertRaises(ValueError):
            (GD, GDE) = reconstruction.good_direction_estimate(X[:,0:2], X_rec[:,0])

        with self.assertRaises(ValueError):
            (GD, GDE) = reconstruction.good_direction_estimate(X[:,0:2], X_rec[:,0:1])

        with self.assertRaises(ValueError):
            (GD, GDE) = reconstruction.good_direction_estimate(X[:,0:1], X_rec[:,0:1])

        with self.assertRaises(ValueError):
            (GD, GDE) = reconstruction.good_direction_estimate(X[:,0:2], [1,2,3])

        with self.assertRaises(ValueError):
            (GD, GDE) = reconstruction.good_direction_estimate([1,2,3], X_rec[:,0:2])

        with self.assertRaises(ValueError):
            (GD, GDE) = reconstruction.good_direction_estimate(X, X_rec, tolerance=[1])

# ------------------------------------------------------------------------------

    def test_reconstruction__good_direction_estimate__computation(self):

        X = np.random.rand(100,5)
        pca_X = reduction.PCA(X, scaling='auto', n_components=5)
        X_rec = pca_X.reconstruct(pca_X.transform(X))
        tolerance = 0.0000001

        try:
            (GD, GDE) = reconstruction.good_direction_estimate(X, X_rec, tolerance=tolerance)
            self.assertTrue(GDE==100)
            self.assertTrue(np.all(GD<=1+tolerance) and np.all(GD>=1-tolerance))
        except Exception:
            self.assertTrue(False)

        pca_X = reduction.PCA(X, scaling='auto', n_components=4)
        X_rec = pca_X.reconstruct(pca_X.transform(X))

        try:
            (GD, GDE) = reconstruction.good_direction_estimate(X, X_rec, tolerance=0.001)
            self.assertTrue(GDE<100)
            self.assertTrue(np.all(GD<=1))
        except Exception:
            self.assertTrue(False)

# ------------------------------------------------------------------------------
