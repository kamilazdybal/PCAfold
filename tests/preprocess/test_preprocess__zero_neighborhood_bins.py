import unittest
import numpy as np
from PCAfold import preprocess
from PCAfold import reduction
from PCAfold import analysis

class Preprocess(unittest.TestCase):

    def test_preprocess__zero_neighborhood_bins__allowed_calls(self):

        try:
            (idx, borders) = preprocess.zero_neighborhood_bins(np.array([-100, -20, -0.1, 0, 0.1, 1, 10, 20, 200, 300, 400]), k=4, split_at_zero=True, verbose=False)
            self.assertTrue(True)
        except:
            self.assertTrue(False)

        self.assertTrue(isinstance(idx, np.ndarray))
        self.assertTrue(idx.ndim == 1)

# ------------------------------------------------------------------------------

    def test_preprocess__zero_neighborhood_bins__not_allowed_calls(self):

        with self.assertRaises(ValueError):
            (idx, borders) = preprocess.zero_neighborhood_bins(np.array([-100, -20, -0.1, 0, 0.1, 1, 10, 20, 200, 300, 400]), k=0, split_at_zero=True, verbose=False)

        with self.assertRaises(ValueError):
            (idx, borders) = preprocess.zero_neighborhood_bins(np.array([-100, -20, -0.1, 0, 0.1, 1, 10, 20, 200, 300, 400]), k=-1, split_at_zero=True, verbose=False)

        with self.assertRaises(ValueError):
            (idx, borders) = preprocess.zero_neighborhood_bins(np.array([-100, -20, -0.1, 0, 0.1, 1, 10, 20, 200, 300, 400]), k=4, split_at_zero=True, verbose=1)

        with self.assertRaises(ValueError):
            (idx, borders) = preprocess.zero_neighborhood_bins(np.array([-100, -20, -0.1, 0, 0.1, 1, 10, 20, 200, 300, 400]), k=4, split_at_zero=True, verbose='True')

# ------------------------------------------------------------------------------

    def test_preprocess__zero_neighborhood_bins__computation(self):

        pass

# ------------------------------------------------------------------------------
