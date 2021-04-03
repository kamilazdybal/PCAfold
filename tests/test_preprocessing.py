import unittest
import numpy as np
from PCAfold import preprocess


class TestPreprocessing(unittest.TestCase):

################################################################################
#
# Test plotting functionalities of the `reduction` module
#
################################################################################

    def test_plot_2d_clustering_allowed_calls(self):

        x = np.linspace(-1,1,100)
        y = -x**2 + 1

        (idx, borders) = preprocess.variable_bins(x, 4, verbose=False)

        try:
            plt = preprocess.plot_2d_clustering(x, y, idx, x_label='$x$', y_label='$y$', color_map='viridis', first_cluster_index_zero=False, grid_on=True, figure_size=(10,6), title='x-y data set', save_filename=None)
            plt.close()
        except Exception:
            self.assertTrue(False)

        try:
            plt = preprocess.plot_2d_clustering(x, y, idx)
            plt.close()
        except Exception:
            self.assertTrue(False)

    def test_plot_3d_clustering_allowed_calls(self):

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

    def test_plot_3d_clustering_not_allowed_calls(self):

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

    def test_plot_2d_train_test_samples(self):

        x = np.linspace(-1,1,100)
        y = -x**2 + 1
        (idx, borders) = preprocess.variable_bins(x, 4, verbose=False)
        sample = preprocess.DataSampler(idx, idx_test=[], random_seed=None, verbose=False)
        (idx_train, idx_test) = sample.number(40, test_selection_option=1)

        try:
            plt = preprocess.plot_2d_train_test_samples(x, y, idx, idx_train, idx_test, x_label='$x$', y_label='$y$', color_map='viridis', first_cluster_index_zero=False, grid_on=True, figure_size=(12,6), title='x-y data set', save_filename=None)
            plt.close()
        except Exception:
            self.assertTrue(False)

        try:
            plt = preprocess.plot_2d_train_test_samples(x, y, idx, idx_train, idx_test)
            plt.close()
        except Exception:
            self.assertTrue(False)
