import unittest
import numpy as np
from PCAfold import preprocess


class TestPreprocessing(unittest.TestCase):

################################################################################
#
# Test plotting functionalities of the `reduction` module
#
################################################################################

    def test_plot_2d_clustering(self):

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
