import unittest
import numpy as np
from PCAfold import compute_normalized_variance, coefficient_of_determination, normalized_variance_derivative, find_local_maxima, random_sampling_normalized_variance
from PCAfold import PCA, plot_normalized_variance, plot_normalized_variance_comparison
from PCAfold import plot_normalized_variance_derivative, plot_normalized_variance_derivative_comparison
from PCAfold import coefficient_of_determination, stratified_coefficient_of_determination, plot_stratified_coefficient_of_determination
from PCAfold import mean_squared_error, root_mean_squared_error, normalized_root_mean_squared_error, good_direction_estimate
from PCAfold import plot_2d_regression, plot_3d_regression


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
        der, sig, _ = normalized_variance_derivative(self._default_variance_data)
        self.assertTrue(self._default_variance_data.bandwidth_values[1] == sig[0])
        self.assertTrue(self._default_variance_data.bandwidth_values[-2] == sig[-1])
        d1 = compute_der(self._default_variance_data.normalized_variance[self._names[0]], self._default_variance_data.bandwidth_values)
        self.assertTrue(np.max(np.abs(der[self._names[0]] - d1/np.max(d1))) <= tol)

        # nonzero limit...
        indepvars = np.array([[1., 2.], [3., 4.], [1., 2.], [5., 7.]])
        depvars = np.array([[2.], [5.], [8.], [10.]])
        variance_data = compute_normalized_variance(indepvars, depvars, self._names)
        der, sig, _ = normalized_variance_derivative(variance_data)
        d1 = compute_der(variance_data.normalized_variance[self._names[0]], variance_data.bandwidth_values)
        self.assertFalse(np.max(np.abs(der[self._names[0]] - d1/np.max(d1))) <= tol)
        d1 += variance_data.normalized_variance_limit[self._names[0]]
        self.assertTrue(np.max(np.abs(der[self._names[0]] - d1/np.max(d1))) <= tol)

    def test_peak_locator(self):
        tol = 1.e-4
        der, sig, _ = normalized_variance_derivative(self._default_variance_data)
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
        der, sig, _ = normalized_variance_derivative(self._default_variance_data)
        self.assertTrue(np.max(np.abs(der[self._names[0]] - pct1[self._names[0]])) <= tol)
        self.assertFalse(np.max(np.abs(der[self._names[0]] - pct2[self._names[0]])) <= tol)
        self.assertTrue(np.max(np.abs(sig - xder)) <= tol)
