import unittest
import numpy as np
from PCAfold import preprocess
from PCAfold import reduction
from PCAfold import analysis

class Analysis(unittest.TestCase):

    def test_analysis__max_absolute_error__allowed_calls(self):

        X = np.random.rand(100,5)
        pca_X = reduction.PCA(X, scaling='auto', n_components=2)
        X_rec = pca_X.reconstruct(pca_X.transform(X))

        try:
            MSE = analysis.max_absolute_error(X[:,0], X_rec[:,0])
        except Exception:
            self.assertTrue(False)

        try:
            MSE = analysis.max_absolute_error(X[:,0], X[:,0])
        except Exception:
            self.assertTrue(False)

# ------------------------------------------------------------------------------

    def test_analysis__max_absolute_error__not_allowed_calls(self):

        X = np.random.rand(100,5)

        with self.assertRaises(ValueError):
            MSE = analysis.max_absolute_error(X[:,0:2], X[:,0])

        with self.assertRaises(ValueError):
            MSE = analysis.max_absolute_error(X[:,0], X[:,0:2])

        with self.assertRaises(ValueError):
            MSE = analysis.max_absolute_error([1,2,3], X[:,0])

        with self.assertRaises(ValueError):
            MSE = analysis.max_absolute_error(X[:,0], [1,2,3])

# ------------------------------------------------------------------------------

    def test_analysis__max_absolute_error__computation(self):

        X = np.random.rand(100,5)
        pca_X = reduction.PCA(X, scaling='auto', n_components=5)
        X_rec = pca_X.reconstruct(pca_X.transform(X))

        try:
            MSE = analysis.max_absolute_error(X[:,0], X_rec[:,0])
            self.assertTrue(MSE<10**-15)
        except Exception:
            self.assertTrue(False)

        try:
            MSE = analysis.max_absolute_error(X[:,0], X[:,0])
            self.assertTrue(MSE<10**-15)
        except Exception:
            self.assertTrue(False)

        pca_X = reduction.PCA(X, scaling='auto', n_components=4)
        X_rec = pca_X.reconstruct(pca_X.transform(X))

        try:
            MSE = analysis.max_absolute_error(X[:,0], X_rec[:,0])
            self.assertTrue(MSE>0)
        except Exception:
            self.assertTrue(False)

        try:
            MSE = analysis.max_absolute_error(X[:,0], X[:,1])
            self.assertTrue(MSE>0)
        except Exception:
            self.assertTrue(False)

# ------------------------------------------------------------------------------
