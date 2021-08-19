import unittest
import numpy as np
from PCAfold import preprocess
from PCAfold import reduction
from PCAfold import analysis

class Preprocess(unittest.TestCase):

    def test_preprocess__get_populations__allowed_calls(self):

        x = np.linspace(-1,1,100)

        try:
            (idx, borders) = preprocess.variable_bins(x, 4, verbose=False)
            idx_populations = [25, 25, 25, 25]
            populations = preprocess.get_populations(idx)
            self.assertTrue(populations == idx_populations)
        except Exception:
            self.assertTrue(False)

        try:
            (idx, borders) = preprocess.variable_bins(x, 5, verbose=False)
            idx_populations = [20, 20, 20, 20, 20]
            populations = preprocess.get_populations(idx)
            self.assertTrue(populations == idx_populations)
        except Exception:
            self.assertTrue(False)

        try:
            (idx, borders) = preprocess.variable_bins(x, 2, verbose=False)
            idx_populations = [50, 50]
            populations = preprocess.get_populations(idx)
            self.assertTrue(populations == idx_populations)
        except Exception:
            self.assertTrue(False)

        try:
            (idx, borders) = preprocess.variable_bins(x, 1, verbose=False)
            idx_populations = [100]
            populations = preprocess.get_populations(idx)
            self.assertTrue(populations == idx_populations)
        except Exception:
            self.assertTrue(False)

        try:
            idx_populations = [1]
            populations = preprocess.get_populations(np.array([0]))
            self.assertTrue(populations == idx_populations)
        except Exception:
            self.assertTrue(False)

# ------------------------------------------------------------------------------

    def test_preprocess__get_populations__not_allowed_calls(self):

        pass

# ------------------------------------------------------------------------------

    def test_preprocess__get_populations__computation(self):

        pass

# ------------------------------------------------------------------------------
