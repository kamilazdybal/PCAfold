import unittest
import numpy as np
from PCAfold import preprocess


class TestPreprocessing(unittest.TestCase):

################################################################################
#
# Test plotting functionalities of the `preprocess` module
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

    def test_plot_2d_clustering__not_allowed_calls(self):

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

    def test_plot_3d_clustering__allowed_calls(self):

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

    def test_plot_3d_clustering_not__allowed_calls(self):

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

    def test_plot_2d_train_test_samples__allowed_calls(self):

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

    def test_plot_2d_train_test_samples__not_allowed_calls(self):

        x = np.random.rand(100,1)
        y = np.random.rand(100,1)
        (idx, borders) = preprocess.variable_bins(x, 4, verbose=False)

        sample = preprocess.DataSampler(idx, idx_test=[], random_seed=None, verbose=False)
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

    def test_plot_conditional_statistics__allowed_calls(self):

        conditioning_variable = np.linspace(-1,1,1000)
        y = -conditioning_variable**2 + 1

        try:
            plt = preprocess.plot_conditional_statistics(y, conditioning_variable, k=10, x_label='$x$', y_label='$y$', figure_size=(10,3), title='Conditional mean')
            plt.close()
        except:
            self.assertTrue(False)

        try:
            plt = preprocess.plot_conditional_statistics(y, conditioning_variable, split_values=[-0.5,0.6], x_label='$x$', y_label='$y$', figure_size=(10,3), title='Conditional mean')
            plt.close()
        except:
            self.assertTrue(False)

        X = np.random.rand(100,)
        cond_variable = np.random.rand(100,)

        try:
            plt = preprocess.plot_conditional_statistics(X, cond_variable, k=2)
            plt.close()
        except:
            self.assertTrue(False)

        X = np.random.rand(100,1)
        cond_variable = np.random.rand(100,1)

        try:
            plt = preprocess.plot_conditional_statistics(X, cond_variable, k=2)
            plt.close()
        except:
            self.assertTrue(False)

        X = np.random.rand(100,)
        cond_variable = np.random.rand(100,)

        try:
            plt = preprocess.plot_conditional_statistics(X, cond_variable, k=2)
            plt.close()
        except:
            self.assertTrue(False)

    def test_plot_conditional_statistics__not_allowed_calls(self):

        conditioning_variable = np.linspace(-1,1,1000)
        y = -conditioning_variable**2 + 1

        with self.assertRaises(ValueError):
            plt = preprocess.plot_conditional_statistics([1,2,3], conditioning_variable)

        with self.assertRaises(ValueError):
            plt = preprocess.plot_conditional_statistics(y, [1,2,3])

        with self.assertRaises(ValueError):
            plt = preprocess.plot_conditional_statistics(y, conditioning_variable, k=0)

        with self.assertRaises(ValueError):
            plt = preprocess.plot_conditional_statistics(y, conditioning_variable, k=-1)

        with self.assertRaises(ValueError):
            plt = preprocess.plot_conditional_statistics(y, conditioning_variable, k=2.5)

        with self.assertRaises(ValueError):
            plt = preprocess.plot_conditional_statistics(y, conditioning_variable, split_values=1)

        with self.assertRaises(ValueError):
            plt = preprocess.plot_conditional_statistics(y, conditioning_variable, statistics_to_plot=['none'])

        with self.assertRaises(ValueError):
            plt = preprocess.plot_conditional_statistics(y, conditioning_variable, statistics_to_plot=['mean', 'none'])

        with self.assertRaises(ValueError):
            plt = preprocess.plot_conditional_statistics(y, conditioning_variable, color=1)

        with self.assertRaises(ValueError):
            plt = preprocess.plot_conditional_statistics(y, conditioning_variable, x_label=1)

        with self.assertRaises(ValueError):
            plt = preprocess.plot_conditional_statistics(y, conditioning_variable, y_label=1)

        with self.assertRaises(ValueError):
            plt = preprocess.plot_conditional_statistics(y, conditioning_variable, colorbar_label=1)

        with self.assertRaises(ValueError):
            plt = preprocess.plot_conditional_statistics(y, conditioning_variable, color_map=1)

        with self.assertRaises(ValueError):
            plt = preprocess.plot_conditional_statistics(y, conditioning_variable, figure_size=1)

        with self.assertRaises(ValueError):
            plt = preprocess.plot_conditional_statistics(y, conditioning_variable, title=1)

        with self.assertRaises(ValueError):
            plt = preprocess.plot_conditional_statistics(y, conditioning_variable, save_filename=1)
