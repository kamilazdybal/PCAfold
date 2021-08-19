import unittest
import numpy as np
from PCAfold import preprocess
from PCAfold import reduction
from PCAfold import analysis

class Preprocess(unittest.TestCase):

    def test_preprocess__plot_3d_clustering__allowed_calls(self):

        x = np.linspace(-1,1,100)
        y = -x**2 + 1
        z = x + 10

        (idx, borders) = preprocess.variable_bins(x, 4, verbose=False)

        try:
            plt = preprocess.plot_3d_clustering(x, y, z, idx, elev=0, azim=0, x_label='$x$', y_label='$y$', z_label='$z', color_map='viridis', first_cluster_index_zero=False, figure_size=(10,6), title='x-y data set', save_filename=None)
            plt.close()
        except Exception:
            self.assertTrue(False)

        try:
            plt = preprocess.plot_3d_clustering(x, y, z, idx)
            plt.close()
        except Exception:
            self.assertTrue(False)

# ------------------------------------------------------------------------------

    def test_preprocess__plot_3d_clustering__not_allowed_calls(self):

        X = np.random.rand(100,10)
        x = np.linspace(-1,1,100)
        y = -x**2 + 1
        z = x + 10

        (idx, borders) = preprocess.variable_bins(x, 4, verbose=False)

        with self.assertRaises(ValueError):
            plt = preprocess.plot_3d_clustering(X[:,0:2], y, z, idx)
            plt.close()

        with self.assertRaises(ValueError):
            plt = preprocess.plot_3d_clustering(x, X[:,0:2], z, idx)
            plt.close()

        with self.assertRaises(ValueError):
            plt = preprocess.plot_3d_clustering(x, y, X[0:2], idx)
            plt.close()

        with self.assertRaises(ValueError):
            plt = preprocess.plot_3d_clustering(x[0:10], y, z, idx)
            plt.close()

        with self.assertRaises(ValueError):
            plt = preprocess.plot_3d_clustering(x, y[0:10], z, idx)
            plt.close()

        with self.assertRaises(ValueError):
            plt = preprocess.plot_3d_clustering(x, y, z[0:10], idx)
            plt.close()

        with self.assertRaises(ValueError):
            plt = preprocess.plot_3d_clustering(x, y, z, idx[0:10])
            plt.close()

# ------------------------------------------------------------------------------
