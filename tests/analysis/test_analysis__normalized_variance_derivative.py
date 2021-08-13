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

    def test_analysis__normalized_variance_derivative__allowed_calls(self):

        pass

# ------------------------------------------------------------------------------

    def test_analysis__normalized_variance_derivative__not_allowed_calls(self):

        pass

# ------------------------------------------------------------------------------

    def test_analysis__normalized_variance_derivative__computation(self):
        tol = 1.e-8

        def compute_der(normvar, sigma):
            return (normvar[2:] - normvar[:-2]) / (np.log10(sigma[2:]) - np.log10(sigma[:-2]))
        der, sig, _ = analysis.normalized_variance_derivative(self._default_variance_data)
        self.assertTrue(self._default_variance_data.bandwidth_values[1] == sig[0])
        self.assertTrue(self._default_variance_data.bandwidth_values[-2] == sig[-1])
        d1 = compute_der(self._default_variance_data.normalized_variance[self._names[0]], self._default_variance_data.bandwidth_values)
        self.assertTrue(np.max(np.abs(der[self._names[0]] - d1/np.max(d1))) <= tol)

        # nonzero limit...
        indepvars = np.array([[1., 2.], [3., 4.], [1., 2.], [5., 7.]])
        depvars = np.array([[2.], [5.], [8.], [10.]])
        variance_data = analysis.compute_normalized_variance(indepvars, depvars, self._names)
        der, sig, _ = analysis.normalized_variance_derivative(variance_data)
        d1 = compute_der(variance_data.normalized_variance[self._names[0]], variance_data.bandwidth_values)
        self.assertFalse(np.max(np.abs(der[self._names[0]] - d1/np.max(d1))) <= tol)
        d1 += variance_data.normalized_variance_limit[self._names[0]]
        self.assertTrue(np.max(np.abs(der[self._names[0]] - d1/np.max(d1))) <= tol)

# ------------------------------------------------------------------------------
