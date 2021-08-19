import unittest
import numpy as np
from PCAfold import preprocess
from PCAfold import reduction
from PCAfold import analysis

class Preprocess(unittest.TestCase):

    def test_preprocess__variable_bins__allowed_calls(self):

        try:
            (idx, borders) = preprocess.variable_bins(np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]), 4, verbose=False)
            self.assertTrue(True)
        except:
            self.assertTrue(False)

        self.assertTrue(isinstance(idx, np.ndarray))
        self.assertTrue(idx.ndim == 1)

# ------------------------------------------------------------------------------

    def test_preprocess__variable_bins__not_allowed_calls(self):

        with self.assertRaises(ValueError):
            (idx, borders) = preprocess.variable_bins(np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]), 0, verbose=False)

        with self.assertRaises(ValueError):
            (idx, borders) = preprocess.variable_bins(np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]), -1, verbose=False)

        with self.assertRaises(ValueError):
            (idx, borders) = preprocess.variable_bins(np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]), 4, verbose=1)

        with self.assertRaises(ValueError):
            (idx, borders) = preprocess.variable_bins(np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]), 4, verbose='True')

# ------------------------------------------------------------------------------

    def test_preprocess__variable_bins__computation(self):

        pass

# ------------------------------------------------------------------------------
