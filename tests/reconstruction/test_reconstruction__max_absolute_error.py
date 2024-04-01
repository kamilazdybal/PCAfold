import unittest
import numpy as np
from PCAfold import preprocess
from PCAfold import reduction
from PCAfold import reconstruction

class Reconstruction(unittest.TestCase):

    def test_reconstruction__max_absolute_error__allowed_calls(self):

        X = np.random.rand(100,5)
        pca_X = reduction.PCA(X, scaling='auto', n_components=2)
        X_rec = pca_X.reconstruct(pca_X.transform(X))

        try:
            MaxAE = reconstruction.max_absolute_error(X[:,0], X_rec[:,0])
        except Exception:
            self.assertTrue(False)

        try:
            MaxAE = reconstruction.max_absolute_error(X[:,0], X[:,0])
        except Exception:
            self.assertTrue(False)

# ------------------------------------------------------------------------------

    def test_reconstruction__max_absolute_error__not_allowed_calls(self):

        X = np.random.rand(100,5)

        with self.assertRaises(ValueError):
            MaxAE = reconstruction.max_absolute_error(X[:,0:2], X[:,0])

        with self.assertRaises(ValueError):
            MaxAE = reconstruction.max_absolute_error(X[:,0], X[:,0:2])

        with self.assertRaises(ValueError):
            MaxAE = reconstruction.max_absolute_error([1,2,3], X[:,0])

        with self.assertRaises(ValueError):
            MaxAE = reconstruction.max_absolute_error(X[:,0], [1,2,3])

# ------------------------------------------------------------------------------

    def test_reconstruction__max_absolute_error__computation(self):

        X = np.random.rand(100,5)
        pca_X = reduction.PCA(X, scaling='auto', n_components=5)
        X_rec = pca_X.reconstruct(pca_X.transform(X))

        try:
            MaxAE = reconstruction.max_absolute_error(X[:,0], X_rec[:,0])
            self.assertTrue(MaxAE<10**-15)
        except Exception:
            self.assertTrue(False)

        try:
            MaxAE = reconstruction.max_absolute_error(X[:,0], X[:,0])
            self.assertTrue(MaxAE<10**-15)
        except Exception:
            self.assertTrue(False)

        pca_X = reduction.PCA(X, scaling='auto', n_components=4)
        X_rec = pca_X.reconstruct(pca_X.transform(X))

        try:
            MaxAE = reconstruction.max_absolute_error(X[:,0], X_rec[:,0])
            self.assertTrue(MaxAE>0)
        except Exception:
            self.assertTrue(False)

        try:
            MaxAE = reconstruction.max_absolute_error(X[:,0], X[:,1])
            self.assertTrue(MaxAE>0)
        except Exception:
            self.assertTrue(False)

        X = np.ones((100,1))
        X_rec = np.ones((100,1))
        X_rec[40,:] = 100
        MaxAE = reconstruction.max_absolute_error(X, X_rec)
        self.assertTrue(maxae==99)

        X = np.ones((100,1))
        X_rec = np.ones((100,1))
        X_rec[40,:] = -100
        MaxAE = reconstruction.max_absolute_error(X, X_rec)
        self.assertTrue(maxae==99)

        X = np.ones((100,1))
        X_rec = np.ones((100,1))
        X_rec[40,:] = 2
        MaxAE = reconstruction.max_absolute_error(X, X_rec)
        self.assertTrue(maxae==1)

        X = np.ones((100,1))
        X_rec = np.ones((100,1))
        X_rec[40,:] = -0.1
        MaxAE = reconstruction.max_absolute_error(X, X_rec)
        self.assertTrue(maxae==0.9)

        X = np.ones((100,1))
        X_rec = np.ones((100,1))
        X_rec[40,:] = -0.1
        X_rec[0] = 10
        X_rec[81] = 20
        MaxAE = reconstruction.max_absolute_error(X, X_rec)
        self.assertTrue(maxae==19)

# ------------------------------------------------------------------------------
