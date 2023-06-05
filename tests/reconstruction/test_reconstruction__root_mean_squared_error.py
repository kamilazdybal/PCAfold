import unittest
import numpy as np
from PCAfold import preprocess
from PCAfold import reduction
from PCAfold import reconstruction
from sys import modules

try:
    from sklearn.metrics import mean_squared_error
except ImportError:
    pass

class Analysis(unittest.TestCase):

    def test_reconstruction__root_mean_squared_error__allowed_calls(self):

        X = np.random.rand(100,5)
        pca_X = reduction.PCA(X, scaling='auto', n_components=2)
        X_rec = pca_X.reconstruct(pca_X.transform(X))

        try:
            RMSE = reconstruction.root_mean_squared_error(X[:,0], X_rec[:,0])
        except Exception:
            self.assertTrue(False)

        try:
            RMSE = reconstruction.root_mean_squared_error(X[:,0], X[:,0])
        except Exception:
            self.assertTrue(False)

# ------------------------------------------------------------------------------

    def test_reconstruction__root_mean_squared_error__not_allowed_calls(self):

        X = np.random.rand(100,5)

        with self.assertRaises(ValueError):
            RMSE = reconstruction.root_mean_squared_error(X[:,0:2], X[:,0])

        with self.assertRaises(ValueError):
            RMSE = reconstruction.root_mean_squared_error(X[:,0], X[:,0:2])

        with self.assertRaises(ValueError):
            RMSE = reconstruction.root_mean_squared_error([1,2,3], X[:,0])

        with self.assertRaises(ValueError):
            RMSE = reconstruction.root_mean_squared_error(X[:,0], [1,2,3])

# ------------------------------------------------------------------------------

    def test_reconstruction__root_mean_squared_error__computation(self):

        X = np.random.rand(100,5)
        pca_X = reduction.PCA(X, scaling='auto', n_components=5)
        X_rec = pca_X.reconstruct(pca_X.transform(X))

        try:
            RMSE = reconstruction.root_mean_squared_error(X[:,0], X_rec[:,0])
            self.assertTrue(RMSE<10**-15)
        except Exception:
            self.assertTrue(False)

        try:
            RMSE = reconstruction.root_mean_squared_error(X[:,0], X[:,0])
            self.assertTrue(RMSE<10**-15)
        except Exception:
            self.assertTrue(False)

        pca_X = reduction.PCA(X, scaling='auto', n_components=4)
        X_rec = pca_X.reconstruct(pca_X.transform(X))

        try:
            RMSE = reconstruction.root_mean_squared_error(X[:,0], X_rec[:,0])
            self.assertTrue(RMSE>0)
        except Exception:
            self.assertTrue(False)

        try:
            RMSE = reconstruction.root_mean_squared_error(X[:,0], X[:,1])
            self.assertTrue(RMSE>0)
        except Exception:
            self.assertTrue(False)

# ------------------------------------------------------------------------------

    def test_reconstruction__root_mean_squared_error__check_against_sklearn(self):

        n_repeat_scenario = 5

        if 'sklearn' in modules:

            tol = np.finfo(float).eps

            for i in range(0,n_repeat_scenario):
                X = np.random.rand(100,2)
                rmse = reconstruction.root_mean_squared_error(X[:,0], X[:,1])
                rmse_sklearn = mean_squared_error(X[:,0], X[:,1], squared=False)

                if np.any(abs(rmse - rmse_sklearn) > tol):
                    self.assertTrue(False)

# ------------------------------------------------------------------------------
