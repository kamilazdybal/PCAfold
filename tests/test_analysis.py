import unittest
import numpy as np
from PCAfold import compute_normalized_variance, r2value
from PCAfold import PCA, plot_normalized_variance, plot_normalized_variance_comparison


class TestNormalizedVariance(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestNormalizedVariance, self).__init__(*args, **kwargs)
        self._indepvars = np.array([[1., 2.], [3., 5.], [6., 9.]])
        self._depvars = np.array([[2.], [4.], [1.]])
        self._names = ['depvar']
        self._default_variance_data = compute_normalized_variance(self._indepvars, self._depvars, self._names)
        self._scl_indepvars = (self._indepvars - np.min(self._indepvars, axis=0)) / (
                np.max(self._indepvars, axis=0) - np.min(self._indepvars, axis=0))

    def test_default_bandwidth_size(self):
        self.assertTrue(self._default_variance_data.bandwidth_values.size == 25)

    def test_default_min_bandwidth(self):
        mindist = np.linalg.norm(self._scl_indepvars[0, :] - self._scl_indepvars[1, :])
        self.assertTrue(self._default_variance_data.bandwidth_values[0] == mindist)

    def test_default_max_bandwidth(self):
        maxdist = np.linalg.norm(np.max(self._scl_indepvars, axis=0) - np.min(self._scl_indepvars, axis=0)) * 10.
        self.assertTrue(self._default_variance_data.bandwidth_values[-1] == maxdist)

    def test_default_varnames(self):
        self.assertTrue(self._default_variance_data.variable_names == self._names)

    def test_default_bandwidth_rise(self):
        # the 10% rise is at the minimum bandwidth because the normalized variance never reaches 0
        self.assertTrue(self._default_variance_data.bandwidth_10pct_rise[self._names[0]] ==
                        self._default_variance_data.bandwidth_values[0])

    def test_bandwidth_array(self):
        bw = np.array([0.001, 0.1])
        variance_data = compute_normalized_variance(self._indepvars, self._depvars, self._names, bandwidth_values=bw)
        self.assertTrue(np.all(variance_data.bandwidth_values == bw))
        self.assertTrue(variance_data.bandwidth_10pct_rise[self._names[0]] is None)

    def test_normalized_variance(self):
        gs_normvar = 0.69507212
        bw = np.array([1.])
        variance_data = compute_normalized_variance(self._indepvars, self._depvars, self._names, bandwidth_values=bw)
        self.assertTrue(np.abs(variance_data.normalized_variance[self._names[0]][0] - gs_normvar) < 1.e-6)

    def test_r2value(self):
        obs = np.array([0., 1., 2.])
        offset = 1.e-1
        expected = 1. - 0.5*(3 * offset**2)
        self.assertTrue(np.abs(r2value(obs, obs + offset) - expected) < 1.e-6)
        self.assertTrue(np.abs(r2value(obs, obs) - 1.0) < 1.e-6)

    def test_plot_normalized_variance(self):

        X = np.random.rand(100,5)
        pca_X = PCA(X, n_components=2)
        principal_components = pca_X.transform(X)
        variance_data = compute_normalized_variance(principal_components, X, depvar_names=['A', 'B', 'C', 'D', 'E'], bandwidth_values=np.logspace(-3, 1, 20), scale_unit_box=True)

        try:
            plt = plot_normalized_variance(variance_data, plot_variables=[0,1,2], color_map='Blues', figure_size=(10,5), title=None, save_filename=None)
            plt.close()
        except Exception:
            self.assertTrue(False)

        try:
            plt = plot_normalized_variance(variance_data, plot_variables=[], color_map='Blues', figure_size=(10,5), title='Normalized variance', save_filename=None)
            plt.close()
        except Exception:
            self.assertTrue(False)

        try:
            plt = plot_normalized_variance(variance_data, plot_variables=[2,3,4], color_map='Blues', figure_size=(15,5), title='Normalized variance', save_filename=None)
            plt.close()
        except Exception:
            self.assertTrue(False)

        try:
            plt = plot_normalized_variance(variance_data, plot_variables=[], color_map='Reds', figure_size=(10,5), title=None, save_filename=None)
            plt.close()
        except Exception:
            self.assertTrue(False)

    def test_plot_normalized_variance_comparison(self):

        X = np.random.rand(100,5)
        Y = np.random.rand(100,5)
        Z = np.random.rand(100,5)
        pca_X = PCA(X, n_components=2)
        pca_Y = PCA(Y, n_components=2)
        pca_Z = PCA(Y, n_components=2)
        principal_components_X = pca_X.transform(X)
        principal_components_Y = pca_Y.transform(Y)
        principal_components_Z = pca_Z.transform(Z)
        variance_data_X = compute_normalized_variance(principal_components_X, X, depvar_names=['A', 'B', 'C', 'D', 'E'], bandwidth_values=np.logspace(-3, 2, 20), scale_unit_box=True)
        variance_data_Y = compute_normalized_variance(principal_components_Y, Y, depvar_names=['F', 'G', 'H', 'I', 'J'], bandwidth_values=np.logspace(-3, 2, 20), scale_unit_box=True)
        variance_data_Z = compute_normalized_variance(principal_components_Z, Z, depvar_names=['K', 'L', 'M', 'N', 'O'], bandwidth_values=np.logspace(-3, 2, 20), scale_unit_box=True)

        try:
            plt = plot_normalized_variance_comparison((variance_data_X, variance_data_Y), ([0,1,2], [0,1,2]), ('Blues', 'Reds'), title=None, save_filename=None)
            plt.close()
        except Exception:
            self.assertTrue(False)

        try:
            plt = plot_normalized_variance_comparison((variance_data_X, variance_data_Y), ([0,1,2], [0,1,2]), ('Blues', 'Reds'), title='Normalized variance comparison', save_filename=None)
            plt.close()
        except Exception:
            self.assertTrue(False)

        try:
            plt = plot_normalized_variance_comparison((variance_data_X, variance_data_Y, variance_data_Z), ([0,1,2], [0,1,2], []), ('Greys', 'Blues', 'Reds'), title='Normalized variance comparison', save_filename=None)
            plt.close()
        except Exception:
            self.assertTrue(False)

        try:
            plt = plot_normalized_variance_comparison((variance_data_X, variance_data_Y, variance_data_Z), ([0], [2,3], []), ('Greys', 'Blues', 'Reds'), title=None, save_filename=None)
            plt.close()
        except Exception:
            self.assertTrue(False)
