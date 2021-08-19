import unittest
import numpy as np
from PCAfold import preprocess
from PCAfold import reduction
from PCAfold import analysis

class Preprocess(unittest.TestCase):

    def test_preprocess__plot_2d_clustering__allowed_calls(self):

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

# ------------------------------------------------------------------------------

    def test_preprocess__plot_2d_clustering__not_allowed_calls(self):

        x = np.random.rand(100,1)
        y = np.random.rand(100,1)
        (idx, borders) = preprocess.variable_bins(x, 4, verbose=False)

        with self.assertRaises(ValueError):
            plt = preprocess.plot_2d_clustering([1,2,3], y, idx)

        with self.assertRaises(ValueError):
            plt = preprocess.plot_2d_clustering(x, [1,2,3], idx)

        with self.assertRaises(ValueError):
            plt = preprocess.plot_2d_clustering(x, y, [1,2,3])

        with self.assertRaises(ValueError):
            plt = preprocess.plot_2d_clustering(x, y, idx, x_label=1)

        with self.assertRaises(ValueError):
            plt = preprocess.plot_2d_clustering(x, y, idx, y_label=1)

        with self.assertRaises(ValueError):
            plt = preprocess.plot_2d_clustering(x, y, idx, color_map=1)

        with self.assertRaises(ValueError):
            plt = preprocess.plot_2d_clustering(x, y, idx, first_cluster_index_zero=1)

        with self.assertRaises(ValueError):
            plt = preprocess.plot_2d_clustering(x, y, idx, grid_on=1)

        with self.assertRaises(ValueError):
            plt = preprocess.plot_2d_clustering(x, y, idx, figure_size=1)

        with self.assertRaises(ValueError):
            plt = preprocess.plot_2d_clustering(x, y, idx, title=1)

        with self.assertRaises(ValueError):
            plt = preprocess.plot_2d_clustering(x, y, idx, save_filename=1)

        with self.assertRaises(ValueError):
            plt = preprocess.plot_2d_clustering(x[0:20,:], y, idx)

        with self.assertRaises(ValueError):
            plt = preprocess.plot_2d_clustering(x, y[0:20,:], idx)

        with self.assertRaises(ValueError):
            plt = preprocess.plot_2d_clustering(x, y, idx[0:20])

# ------------------------------------------------------------------------------
