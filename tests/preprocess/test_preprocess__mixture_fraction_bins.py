import unittest
import numpy as np
from PCAfold import preprocess
from PCAfold import reduction
from PCAfold import analysis

class Preprocess(unittest.TestCase):

    def test_preprocess__mixture_fraction_bins__allowed_calls(self):

        try:
            (idx, borders) = preprocess.mixture_fraction_bins(np.array([0.1, 0.15, 0.2, 0.25, 0.6, 0.8, 1]), 2, 0.2)
            self.assertTrue(True)
        except:
            self.assertTrue(False)

        try:
            (idx, borders) = preprocess.mixture_fraction_bins(np.array([0.1, 0.15, 0.2, 0.25, 0.6, 0.8, 1]), 1, 0.2)
            self.assertTrue(True)
        except:
            self.assertTrue(False)

        self.assertTrue(isinstance(idx, np.ndarray))
        self.assertTrue(idx.ndim == 1)

# ------------------------------------------------------------------------------

    def test_preprocess__mixture_fraction_bins__not_allowed_calls(self):

        with self.assertRaises(ValueError):
            (idx, borders) = preprocess.mixture_fraction_bins(np.array([0.1, 0.15, 0.2, 0.25, 0.6, 0.8, 1]), 0, 0.2)

        with self.assertRaises(ValueError):
            (idx, borders) = preprocess.mixture_fraction_bins(np.array([0.1, 0.15, 0.2, 0.25, 0.6, 0.8, 1]), -1, 0.2)

        with self.assertRaises(ValueError):
            (idx, borders) = preprocess.mixture_fraction_bins(np.array([0.1, 0.15, 0.2, 0.25, 0.6, 0.8, 1]), 2, 0.2, verbose=1)

        with self.assertRaises(ValueError):
            (idx, borders) = preprocess.mixture_fraction_bins(np.array([0.1, 0.15, 0.2, 0.25, 0.6, 0.8, 1]), 2, 0.2, verbose='True')

# ------------------------------------------------------------------------------

    def test_preprocess__mixture_fraction_bins__computation(self):

        pass

# ------------------------------------------------------------------------------
