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

    def test_analysis__random_sampling_normalized_variance__allowed_calls(self):

        pass

# ------------------------------------------------------------------------------

    def test_analysis__random_sampling_normalized_variance__not_allowed_calls(self):

        pass

# ------------------------------------------------------------------------------

    def test_analysis__random_sampling_normalized_variance__computation(self):
        tol = 1.e-8
        avg_der_data, xder, _ = analysis.random_sampling_normalized_variance([1., 0.67], self._indepvars, self._depvars, self._names,
                                                                    bandwidth_values=self._default_variance_data.bandwidth_values,
                                                                    verbose=False)
        pct1 = avg_der_data[1.]
        pct2 = avg_der_data[0.67]
        der, sig, _ = analysis.normalized_variance_derivative(self._default_variance_data)
        self.assertTrue(np.max(np.abs(der[self._names[0]] - pct1[self._names[0]])) <= tol)
        self.assertFalse(np.max(np.abs(der[self._names[0]] - pct2[self._names[0]])) <= tol)
        self.assertTrue(np.max(np.abs(sig - xder)) <= tol)

# ------------------------------------------------------------------------------
