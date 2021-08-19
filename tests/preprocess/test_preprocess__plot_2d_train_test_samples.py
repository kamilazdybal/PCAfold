import unittest
import numpy as np
from PCAfold import preprocess
from PCAfold import reduction
from PCAfold import analysis

class Preprocess(unittest.TestCase):

    def test_preprocess__plot_2d_train_test_samples__allowed_calls(self):

        x = np.linspace(-1,1,100)
        y = -x**2 + 1
        (idx, borders) = preprocess.variable_bins(x, 4, verbose=False)
        sample = preprocess.DataSampler(idx, idx_test=None, random_seed=None, verbose=False)
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

# ------------------------------------------------------------------------------

    def test_preprocess__plot_2d_train_test_samples__not_allowed_calls(self):

        x = np.random.rand(100,1)
        y = np.random.rand(100,1)
        (idx, borders) = preprocess.variable_bins(x, 4, verbose=False)

        sample = preprocess.DataSampler(idx, idx_test=None, random_seed=None, verbose=False)
        (idx_train, idx_test) = sample.number(10, test_selection_option=1)

        with self.assertRaises(ValueError):
            plt = preprocess.plot_2d_train_test_samples([1,2,3], y, idx, idx_train, idx_test)

        with self.assertRaises(ValueError):
            plt = preprocess.plot_2d_train_test_samples(x, [1,2,3], idx, idx_train, idx_test)

        with self.assertRaises(ValueError):
            plt = preprocess.plot_2d_train_test_samples(x, y, [1,2,3], idx_train, idx_test)

        with self.assertRaises(ValueError):
            plt = preprocess.plot_2d_train_test_samples(x, y, idx, [1,2,3], idx_test)

        with self.assertRaises(ValueError):
            plt = preprocess.plot_2d_train_test_samples(x, y, idx, idx_train, [1,2,3])

        with self.assertRaises(ValueError):
            plt = preprocess.plot_2d_train_test_samples(x, y, idx, idx_train, idx_test, x_label=1)

        with self.assertRaises(ValueError):
            plt = preprocess.plot_2d_train_test_samples(x, y, idx, idx_train, idx_test, y_label=1)

        with self.assertRaises(ValueError):
            plt = preprocess.plot_2d_train_test_samples(x, y, idx, idx_train, idx_test, color_map=1)

        with self.assertRaises(ValueError):
            plt = preprocess.plot_2d_train_test_samples(x, y, idx, idx_train, idx_test, first_cluster_index_zero=1)

        with self.assertRaises(ValueError):
            plt = preprocess.plot_2d_train_test_samples(x, y, idx, idx_train, idx_test, grid_on=1)

        with self.assertRaises(ValueError):
            plt = preprocess.plot_2d_train_test_samples(x, y, idx, idx_train, idx_test, figure_size=1)

        with self.assertRaises(ValueError):
            plt = preprocess.plot_2d_train_test_samples(x, y, idx, idx_train, idx_test, title=1)

        with self.assertRaises(ValueError):
            plt = preprocess.plot_2d_train_test_samples(x, y, idx, idx_train, idx_test, save_filename=1)

# ------------------------------------------------------------------------------
