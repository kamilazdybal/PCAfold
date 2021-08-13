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

    def test_analysis__find_local_maxima__allowed_calls(self):

        pass

# ------------------------------------------------------------------------------

    def test_analysis__find_local_maxima__not_allowed_calls(self):

        pass

# ------------------------------------------------------------------------------

    def test_analysis__find_local_maxima__computation(self):
        tol = 1.e-4
        der, sig, _ = analysis.normalized_variance_derivative(self._default_variance_data)
        peak_locs, peak_vals = analysis.find_local_maxima(der[self._names[0]], sig)
        self.assertTrue(peak_locs.size == 1)
        self.assertTrue(peak_vals.size == 1)
        self.assertTrue(np.abs(peak_locs[0] - 0.7225) < tol)
        self.assertTrue(np.abs(peak_vals[0] - 1.0116) < tol)
        # test with large threshold don't have peaks
        peak_locs, peak_vals = analysis.find_local_maxima(der[self._names[0]], sig, threshold=1.1)
        self.assertTrue(peak_locs.size == 0)
        self.assertTrue(peak_vals.size == 0)

# ------------------------------------------------------------------------------
