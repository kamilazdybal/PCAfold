import unittest
import numpy as np
from PCAfold import preprocess
from PCAfold import reduction
from PCAfold import analysis

class Reduction(unittest.TestCase):

    def test_reduction__plot_3d_manifold__allowed_calls(self):

        X = np.random.rand(100,5)

        try:
            plt = reduction.plot_3d_manifold(X[:,0], X[:,1], X[:,2], color=None, x_label=None, y_label=None, colorbar_label=None, title=None, save_filename=None)
            plt.close()
        except Exception:
            self.assertTrue(False)

        try:
            plt = reduction.plot_3d_manifold(X[:,0], X[:,1], X[:,2], color=X[:,0], x_label=None, y_label=None, colorbar_label=None, title=None, save_filename=None)
            plt.close()
        except Exception:
            self.assertTrue(False)

        try:
            plt = reduction.plot_3d_manifold(X[:,0], X[:,1], X[:,2], color=X[:,0:1], x_label=None, y_label=None, colorbar_label=None, title=None, save_filename=None)
            plt.close()
        except Exception:
            self.assertTrue(False)

        try:
            plt = reduction.plot_3d_manifold(X[:,0], X[:,1], X[:,2], color='k', x_label=None, y_label=None, colorbar_label=None, title=None, save_filename=None)
            plt.close()
        except Exception:
            self.assertTrue(False)

        try:
            plt = reduction.plot_3d_manifold(X[:,0], X[:,1], X[:,2], color='k', x_label='$x$', y_label='$y$', colorbar_label='$x_1$', title='Title', save_filename=None)
            plt.close()
        except Exception:
            self.assertTrue(False)

# ------------------------------------------------------------------------------

    def test_reduction__plot_3d_manifold__not_allowed_calls(self):

        X = np.random.rand(100,10)

        with self.assertRaises(ValueError):
            plt = reduction.plot_3d_manifold(X[:,0:2], X[:,0], X[:,0], color=X[:,0])
            plt.close()

        with self.assertRaises(ValueError):
            plt = reduction.plot_3d_manifold(X[:,0], X[:,0:2], X[:,0], color=X[:,0])
            plt.close()

        with self.assertRaises(ValueError):
            plt = reduction.plot_3d_manifold(X[:,0], X[:,0], X[:,0:2], color=X[:,0])
            plt.close()

        with self.assertRaises(ValueError):
            plt = reduction.plot_3d_manifold(X[:,0], X[:,0], X[:,0], color=X[:,0:2])
            plt.close()

        with self.assertRaises(ValueError):
            plt = reduction.plot_3d_manifold([1,2,3], X[:,0], X[:,0], color=X[:,0])
            plt.close()

        with self.assertRaises(ValueError):
            plt = reduction.plot_3d_manifold(X[:,0], [1,2,3], X[:,0], color=X[:,0])
            plt.close()

        with self.assertRaises(ValueError):
            plt = reduction.plot_3d_manifold(X[:,0], X[:,0], [1,2,3], color=X[:,0])
            plt.close()

        with self.assertRaises(ValueError):
            plt = reduction.plot_3d_manifold(X[:,0], X[:,0], X[:,0], color=[1,2,3])
            plt.close()

        with self.assertRaises(ValueError):
            plt = reduction.plot_3d_manifold([1,2,3], [1,2,3], [1,2,3], color=[1,2,3])
            plt.close()

        with self.assertRaises(ValueError):
            plt = reduction.plot_3d_manifold(X[0:10,0], X[:,0], X[:,0], color=X[:,0])
            plt.close()

        with self.assertRaises(ValueError):
            plt = reduction.plot_3d_manifold(X[:,0], X[0:10,0], X[:,0], color=X[:,0])
            plt.close()

        with self.assertRaises(ValueError):
            plt = reduction.plot_3d_manifold(X[:,0], X[:,0], X[0:10,0], color=X[:,0])
            plt.close()

        with self.assertRaises(ValueError):
            plt = reduction.plot_3d_manifold(X[:,0], X[:,0], X[:,0], color=X[0:10,0])
            plt.close()

# ------------------------------------------------------------------------------
