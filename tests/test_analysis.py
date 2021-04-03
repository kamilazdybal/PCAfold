import unittest
import numpy as np
from PCAfold import compute_normalized_variance, r2value, normalized_variance_derivative, find_local_maxima, random_sampling_normalized_variance
from PCAfold import PCA, plot_normalized_variance, plot_normalized_variance_comparison
from PCAfold import plot_normalized_variance_derivative, plot_normalized_variance_derivative_comparison
from PCAfold import stratified_r2, plot_stratified_r2


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

    def test_nonzero_normalized_variance_limit(self):
        indepvars = np.array([[1., 2.], [3., 4.], [1., 2.]])
        depvars = np.array([[2.], [5.], [8.]])
        variance_data = compute_normalized_variance(indepvars, depvars, self._names)
        zerotol = 1.e-16
        self.assertTrue(variance_data.normalized_variance_limit[self._names[0]] > zerotol)
        self.assertTrue(self._default_variance_data.normalized_variance_limit[self._names[0]] <= zerotol)

    def test_normalized_variance_derivative(self):
        tol = 1.e-8

        def compute_der(normvar, sigma):
            return (normvar[2:] - normvar[:-2]) / (np.log10(sigma[2:]) - np.log10(sigma[:-2]))
        der, sig = normalized_variance_derivative(self._default_variance_data)
        self.assertTrue(self._default_variance_data.bandwidth_values[1] == sig[0])
        self.assertTrue(self._default_variance_data.bandwidth_values[-2] == sig[-1])
        d1 = compute_der(self._default_variance_data.normalized_variance[self._names[0]], self._default_variance_data.bandwidth_values)
        self.assertTrue(np.max(np.abs(der[self._names[0]] - d1/np.max(d1))) <= tol)

        # nonzero limit...
        indepvars = np.array([[1., 2.], [3., 4.], [1., 2.], [5., 7.]])
        depvars = np.array([[2.], [5.], [8.], [10.]])
        variance_data = compute_normalized_variance(indepvars, depvars, self._names)
        der, sig = normalized_variance_derivative(variance_data)
        d1 = compute_der(variance_data.normalized_variance[self._names[0]], variance_data.bandwidth_values)
        self.assertFalse(np.max(np.abs(der[self._names[0]] - d1/np.max(d1))) <= tol)
        d1 += variance_data.normalized_variance_limit[self._names[0]]
        self.assertTrue(np.max(np.abs(der[self._names[0]] - d1/np.max(d1))) <= tol)

    def test_peak_locator(self):
        tol = 1.e-4
        der, sig = normalized_variance_derivative(self._default_variance_data)
        peak_locs, peak_vals = find_local_maxima(der[self._names[0]], sig)
        self.assertTrue(peak_locs.size == 1)
        self.assertTrue(peak_vals.size == 1)
        self.assertTrue(np.abs(peak_locs[0] - 0.7225) < tol)
        self.assertTrue(np.abs(peak_vals[0] - 1.0116) < tol)
        # test with large threshold don't have peaks
        peak_locs, peak_vals = find_local_maxima(der[self._names[0]], sig, threshold=1.1)
        self.assertTrue(peak_locs.size == 0)
        self.assertTrue(peak_vals.size == 0)

    def test_normalized_variance_sampling(self):
        tol = 1.e-8
        avg_der_data, xder, _ = random_sampling_normalized_variance([1., 0.67], self._indepvars, self._depvars, self._names,
                                                                    bandwidth_values=self._default_variance_data.bandwidth_values,
                                                                    verbose=False)
        pct1 = avg_der_data[1.]
        pct2 = avg_der_data[0.67]
        der, sig = normalized_variance_derivative(self._default_variance_data)
        self.assertTrue(np.max(np.abs(der[self._names[0]] - pct1[self._names[0]])) <= tol)
        self.assertFalse(np.max(np.abs(der[self._names[0]] - pct2[self._names[0]])) <= tol)
        self.assertTrue(np.max(np.abs(sig - xder)) <= tol)

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


    def test_plot_normalized_variance_derivative(self):

        X = np.random.rand(100,5)
        pca_X = PCA(X, n_components=2)
        principal_components = pca_X.transform(X)
        variance_data = compute_normalized_variance(principal_components, X, depvar_names=['A', 'B', 'C', 'D', 'E'], bandwidth_values=np.logspace(-3, 1, 20), scale_unit_box=True)

        try:
            plt = plot_normalized_variance_derivative(variance_data, plot_variables=[0,1,2], color_map='Blues', figure_size=(10,5), title=None, save_filename=None)
            plt.close()
        except Exception:
            self.assertTrue(False)

        try:
            plt = plot_normalized_variance_derivative(variance_data, plot_variables=[], color_map='Blues', figure_size=(10,5), title='Normalized variance', save_filename=None)
            plt.close()
        except Exception:
            self.assertTrue(False)

        try:
            plt = plot_normalized_variance_derivative(variance_data, plot_variables=[2,3,4], color_map='Blues', figure_size=(15,5), title='Normalized variance', save_filename=None)
            plt.close()
        except Exception:
            self.assertTrue(False)

        try:
            plt = plot_normalized_variance_derivative(variance_data, plot_variables=[], color_map='Reds', figure_size=(10,5), title=None, save_filename=None)
            plt.close()
        except Exception:
            self.assertTrue(False)

    def test_plot_normalized_variance_derivative_comparison(self):

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
            plt = plot_normalized_variance_derivative_comparison((variance_data_X, variance_data_Y), ([0,1,2], [0,1,2]), ('Blues', 'Reds'), title=None, save_filename=None)
            plt.close()
        except Exception:
            self.assertTrue(False)

        try:
            plt = plot_normalized_variance_derivative_comparison((variance_data_X, variance_data_Y), ([0,1,2], [0,1,2]), ('Blues', 'Reds'), title='Normalized variance comparison', save_filename=None)
            plt.close()
        except Exception:
            self.assertTrue(False)

        try:
            plt = plot_normalized_variance_derivative_comparison((variance_data_X, variance_data_Y, variance_data_Z), ([0,1,2], [0,1,2], []), ('Greys', 'Blues', 'Reds'), title='Normalized variance comparison', save_filename=None)
            plt.close()
        except Exception:
            self.assertTrue(False)

        try:
            plt = plot_normalized_variance_derivative_comparison((variance_data_X, variance_data_Y, variance_data_Z), ([0], [2,3], []), ('Greys', 'Blues', 'Reds'), title=None, save_filename=None)
            plt.close()
        except Exception:
            self.assertTrue(False)


class TestStratifiedR2(unittest.TestCase):

    def test_stratified_r2_allowed_calls(self):

        X = np.random.rand(100,10)
        pca_X = PCA(X, scaling='auto', n_components=2)
        X_rec = pca_X.reconstruct(pca_X.transform(X))

        try:
            (r2_in_bins, bins_borders) = stratified_r2(X[:,0], X_rec[:,0], 10)
        except Exception:
            self.assertTrue(False)

        try:
            (r2_in_bins, bins_borders) = stratified_r2(X[:,3], X_rec[:,3], 1)
        except Exception:
            self.assertTrue(False)

        try:
            (r2_in_bins, bins_borders) = stratified_r2(X[:,0], X_rec[:,0], 10, use_global_mean=True, verbose=False)
        except Exception:
            self.assertTrue(False)

        try:
            (r2_in_bins, bins_borders) = stratified_r2(X[:,1], X_rec[:,1], 10, use_global_mean=False, verbose=False)
        except Exception:
            self.assertTrue(False)

        try:
            (r2_in_bins, bins_borders) = stratified_r2(X[:,0:1], X_rec[:,0], 10)
        except Exception:
            self.assertTrue(False)

        try:
            (r2_in_bins, bins_borders) = stratified_r2(X[:,3], X_rec[:,3:4], 1)
        except Exception:
            self.assertTrue(False)

        try:
            (r2_in_bins, bins_borders) = stratified_r2(X[:,0:1], X_rec[:,0:1], 10, use_global_mean=True, verbose=False)
        except Exception:
            self.assertTrue(False)

        try:
            (r2_in_bins, bins_borders) = stratified_r2(X[:,1:2], X_rec[:,1:2], 10, use_global_mean=False, verbose=False)
        except Exception:
            self.assertTrue(False)

    def test_stratified_r2_not_allowed_calls(self):

        X = np.random.rand(100,10)
        pca_X = PCA(X, scaling='auto', n_components=2)
        X_rec = pca_X.reconstruct(pca_X.transform(X))

        with self.assertRaises(ValueError):
            (r2_in_bins, bins_borders) = stratified_r2(X[:,1], X_rec[:,1], 0, use_global_mean=False, verbose=False)

        with self.assertRaises(ValueError):
            (r2_in_bins, bins_borders) = stratified_r2(X[:,1], X_rec[:,1], -10, use_global_mean=False, verbose=False)

        with self.assertRaises(ValueError):
            (r2_in_bins, bins_borders) = stratified_r2(X[:,1], X_rec[:,1], 5, use_global_mean=1, verbose=False)

        with self.assertRaises(ValueError):
            (r2_in_bins, bins_borders) = stratified_r2(X[:,1], X_rec[:,1], 10, use_global_mean=False, verbose=1)

        with self.assertRaises(ValueError):
            (r2_in_bins, bins_borders) = stratified_r2(X[:,1], X_rec[:,1], '10', use_global_mean=False, verbose=False)

        with self.assertRaises(ValueError):
            (r2_in_bins, bins_borders) = stratified_r2(X[:,0:2], X_rec[:,0], 10)

        with self.assertRaises(ValueError):
            (r2_in_bins, bins_borders) = stratified_r2(X[:,0], X_rec[:,0:3], 10)

        with self.assertRaises(ValueError):
            (r2_in_bins, bins_borders) = stratified_r2(X[:,3:5], X_rec[:,3:5], 1)

        with self.assertRaises(ValueError):
            (r2_in_bins, bins_borders) = stratified_r2(X[:,0:2], X_rec[:,0:4], 10, use_global_mean=True, verbose=False)

        with self.assertRaises(ValueError):
            (r2_in_bins, bins_borders) = stratified_r2(X[0:10,0], X_rec[:,0], 10)

        with self.assertRaises(ValueError):
            (r2_in_bins, bins_borders) = stratified_r2(X[:,0], X_rec[10:50,0], 10)

        with self.assertRaises(ValueError):
            (r2_in_bins, bins_borders) = stratified_r2([10,20,30], [1,2,3], 10)

    def test_plot_stratified_r2_allowed_calls(self):

        X = np.random.rand(100,5)
        pca_X = PCA(X, scaling='auto', n_components=2)
        X_rec = pca_X.reconstruct(pca_X.transform(X))
        (r2_in_bins, bins_borders) = stratified_r2(X[:,0], X_rec[:,0], 10, use_global_mean=True, verbose=False)

        try:
            plt = plot_stratified_r2(r2_in_bins, bins_borders, variable_name='$X_1$', figure_size=(10,5), title=None, save_filename=None)
            plt.close()
        except Exception:
            self.assertTrue(False)

        try:
            plt = plot_stratified_r2(r2_in_bins, bins_borders)
            plt.close()
        except Exception:
            self.assertTrue(False)

    def plot_3d_regression_allowed_calls(self):

        X = np.random.rand(100,5)
        pca_X = PCA(X, scaling='auto', n_components=2)
        X_rec = pca_X.reconstruct(pca_X.transform(X))

        try:
            plt = plot_3d_regression(X[:,0], X[:,1], X[:,0], X_rec[:,0], elev=45, azim=-45, x_label='$x$', y_label='$y$', z_label='$z$', figure_size=(7,7), title='3D regression')
            plt.close()
        except Exception:
            self.assertTrue(False)

        try:
            plt = plot_3d_regression(X[:,0:1], X[:,1:2], X[:,2:3], X_rec[:,0:1], elev=45, azim=-45, x_label='$x$', y_label='$y$', z_label='$z$', figure_size=(7,7), title='3D regression')
            plt.close()
        except Exception:
            self.assertTrue(False)

    def plot_3d_regression_not_allowed_calls(self):

        X = np.random.rand(100,5)

        with self.assertRaises(ValueError):
            plt = plot_3d_regression(X[:,0:2], X[:,1], X[:,2], X[:,0])
            plt.close()

        with self.assertRaises(ValueError):
            plt = plot_3d_regression(X[:,0], X[:,1:3], X[:,2], X[:,0])
            plt.close()

        with self.assertRaises(ValueError):
            plt = plot_3d_regression(X[:,0], X[:,1], X[:,2:4], X[:,0])
            plt.close()

        with self.assertRaises(ValueError):
            plt = plot_3d_regression(X[:,0], X[:,1], X[:,2], X[:,0:2])
            plt.close()
