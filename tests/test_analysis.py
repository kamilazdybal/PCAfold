import unittest
import numpy as np
from PCAfold import compute_normalized_variance, logistic_fit, assess_manifolds, r2value


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

    def test_logistic_fits(self):
        gs_sigma0 = 0.807886539
        gs_R2 = 0.995794651
        sigma0, R2 = logistic_fit(self._default_variance_data.normalized_variance[self._names[0]],
                                  self._default_variance_data.bandwidth_values)
        self.assertTrue(np.abs(R2 - gs_R2) < 1.e-6)
        self.assertTrue(np.abs(sigma0 - gs_sigma0) < 1.e-6)

        dictname = 'mydata'
        assess_manifold_dict = assess_manifolds({dictname: self._default_variance_data}, show_plot=False)
        self.assertTrue(np.abs(assess_manifold_dict[dictname]['R2'] - gs_R2) < 1.e-6)
        self.assertTrue(np.abs(assess_manifold_dict[dictname]['sigma0'] - gs_sigma0) < 1.e-6)

    def test_r2value(self):
        obs = np.array([0., 1., 2.])
        offset = 1.e-1
        expected = 1. - 0.5*(3 * offset**2)
        self.assertTrue(np.abs(r2value(obs, obs + offset) - expected) < 1.e-6)
        self.assertTrue(np.abs(r2value(obs, obs) - 1.0) < 1.e-6)
