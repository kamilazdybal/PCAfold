import unittest
import numpy as np
from PCAfold import preprocess
from PCAfold import reduction
from PCAfold import analysis

class Analysis(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(Analysis, self).__init__(*args, **kwargs)
        self._indepvars = np.array([[1., 2.], [3., 5.], [6., 9.]])
        self._depvars = np.array([[2.], [4.], [1.]])
        self._names = ['depvar']
        self._default_variance_data = analysis.compute_normalized_variance(self._indepvars, self._depvars, self._names)
        self._scl_indepvars = (self._indepvars - np.min(self._indepvars, axis=0)) / (
                np.max(self._indepvars, axis=0) - np.min(self._indepvars, axis=0))

# ------------------------------------------------------------------------------

    def test_analysis__compute_normalized_variance__allowed_calls(self):

        pass

# ------------------------------------------------------------------------------

    def test_analysis__compute_normalized_variance__not_allowed_calls(self):

        pass

# ------------------------------------------------------------------------------

    def test_analysis__compute_normalized_variance__computation(self):
        gs_normvar = 0.69507212
        bw = np.array([1.])
        variance_data = analysis.compute_normalized_variance(self._indepvars, self._depvars, self._names, bandwidth_values=bw)
        self.assertTrue(np.abs(variance_data.normalized_variance[self._names[0]][0] - gs_normvar) < 1.e-6)

# ------------------------------------------------------------------------------

    def test_analysis__compute_normalized_variance__bandwidth_array(self):
        bw = np.array([0.001, 0.1])
        variance_data = analysis.compute_normalized_variance(self._indepvars, self._depvars, self._names, bandwidth_values=bw)
        self.assertTrue(np.all(variance_data.bandwidth_values == bw))
        self.assertTrue(variance_data.bandwidth_10pct_rise[self._names[0]] is None)

# ------------------------------------------------------------------------------

    def test_analysis__compute_normalized_variance__nonzero_normalized_variance_limit(self):
        indepvars = np.array([[1., 2.], [3., 4.], [1., 2.]])
        depvars = np.array([[2.], [5.], [8.]])
        variance_data = analysis.compute_normalized_variance(indepvars, depvars, self._names)
        zerotol = 1.e-16
        self.assertTrue(variance_data.normalized_variance_limit[self._names[0]] > zerotol)
        self.assertTrue(self._default_variance_data.normalized_variance_limit[self._names[0]] <= zerotol)

# ------------------------------------------------------------------------------

    def test_analysis__compute_normalized_variance__default_bandwidth_size(self):
        self.assertTrue(self._default_variance_data.bandwidth_values.size == 25)

# ------------------------------------------------------------------------------

    def test_analysis__compute_normalized_variance__default_min_bandwidth(self):
        mindist = np.linalg.norm(self._scl_indepvars[0, :] - self._scl_indepvars[1, :])
        self.assertTrue(self._default_variance_data.bandwidth_values[0] == mindist)

# ------------------------------------------------------------------------------

    def test_analysis__compute_normalized_variance__default_max_bandwidth(self):
        maxdist = np.linalg.norm(np.max(self._scl_indepvars, axis=0) - np.min(self._scl_indepvars, axis=0)) * 10.
        self.assertTrue(self._default_variance_data.bandwidth_values[-1] == maxdist)

# ------------------------------------------------------------------------------

    def test_analysis__compute_normalized_variance__default_varnames(self):
        self.assertTrue(self._default_variance_data.variable_names == self._names)

# ------------------------------------------------------------------------------

    def test_analysis__compute_normalized_variance__default_bandwidth_rise(self):
        # the 10% rise is at the minimum bandwidth because the normalized variance never reaches 0
        self.assertTrue(self._default_variance_data.bandwidth_10pct_rise[self._names[0]] ==
                        self._default_variance_data.bandwidth_values[0])

# ------------------------------------------------------------------------------
