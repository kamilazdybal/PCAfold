import unittest
import numpy as np
from PCAfold import preprocess
from PCAfold import reduction
from PCAfold import analysis

class Reduction(unittest.TestCase):

    def test_reduction__plot_parity__allowed_calls(self):

        X = np.random.rand(100,5)

        try:
            pca_X = reduction.PCA(X, scaling='auto', n_components=2)
            principal_components = pca_X.transform(X)
            X_rec = pca_X.reconstruct(principal_components)
            plt = reduction.plot_parity(X[:,0], X_rec[:,0], color=None, x_label=None, y_label=None, colorbar_label=None, title=None, save_filename=None)
            plt.close()
        except Exception:
            self.assertTrue(False)

        try:
            pca_X = reduction.PCA(X, scaling='auto', n_components=2)
            principal_components = pca_X.transform(X)
            X_rec = pca_X.reconstruct(principal_components)
            plt = reduction.plot_parity(X[:,0], X_rec[:,0], color=X[:,0], x_label=None, y_label=None, colorbar_label=None, title=None, save_filename=None)
            plt.close()
        except Exception:
            self.assertTrue(False)

        try:
            pca_X = reduction.PCA(X, scaling='auto', n_components=2)
            principal_components = pca_X.transform(X)
            X_rec = pca_X.reconstruct(principal_components)
            plt = reduction.plot_parity(X[:,0], X_rec[:,0], color=X[:,0:1], x_label=None, y_label=None, colorbar_label=None, title=None, save_filename=None)
            plt.close()
        except Exception:
            self.assertTrue(False)

        try:
            pca_X = reduction.PCA(X, scaling='auto', n_components=2)
            principal_components = pca_X.transform(X)
            X_rec = pca_X.reconstruct(principal_components)
            plt = reduction.plot_parity(X[:,0], X_rec[:,0], color='k', x_label=None, y_label=None, colorbar_label=None, title=None, save_filename=None)
            plt.close()
        except Exception:
            self.assertTrue(False)

        try:
            pca_X = reduction.PCA(X, scaling='auto', n_components=2)
            principal_components = pca_X.transform(X)
            X_rec = pca_X.reconstruct(principal_components)
            plt = reduction.plot_parity(X[:,0], X_rec[:,0], color='k', x_label='$x$', y_label='$y$', colorbar_label='$x_1$', title='Title', save_filename=None)
            plt.close()
        except Exception:
            self.assertTrue(False)

# ------------------------------------------------------------------------------

    def test_reduction__plot_parity__not_allowed_calls(self):

        X = np.random.rand(100,10)
        pca_X = reduction.PCA(X, scaling='auto', n_components=2)
        principal_components = pca_X.transform(X)
        X_rec = pca_X.reconstruct(principal_components)

        with self.assertRaises(ValueError):
            plt = reduction.plot_parity(X[:,0:2], X_rec[:,0], color=X[:,0])
            plt.close()

        with self.assertRaises(ValueError):
            plt = reduction.plot_parity(X[:,0], X_rec[:,0:2], color=X[:,0])
            plt.close()

        with self.assertRaises(ValueError):
            plt = reduction.plot_parity(X[:,0], X_rec[:,0], color=X[:,0:2])
            plt.close()

        with self.assertRaises(ValueError):
            plt = reduction.plot_parity([1,2,3], X_rec[:,0], color=X[:,0])
            plt.close()

        with self.assertRaises(ValueError):
            plt = reduction.plot_parity(X[:,0], [1,2,3], color=X[:,0])
            plt.close()

        with self.assertRaises(ValueError):
            plt = reduction.plot_parity(X[:,0], X_rec[:,0], color=[1,2,3])
            plt.close()

        with self.assertRaises(ValueError):
            plt = reduction.plot_parity([1,2,3], [1,2,3], color=[1,2,3])
            plt.close()

        with self.assertRaises(ValueError):
            plt = reduction.plot_parity(X[0:10,0], X_rec[:,0], color=X[:,0])
            plt.close()

        with self.assertRaises(ValueError):
            plt = reduction.plot_parity(X[:,0], X_rec[0:10,0], color=X[:,0])
            plt.close()

        with self.assertRaises(ValueError):
            plt = reduction.plot_parity(X[:,0], X_rec[:,0], color=X[0:10,0])
            plt.close()

# ------------------------------------------------------------------------------
