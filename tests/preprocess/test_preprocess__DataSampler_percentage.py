import unittest
import numpy as np
from PCAfold import preprocess
from PCAfold import reduction
from PCAfold import analysis

class Preprocess(unittest.TestCase):

    def test_preprocess__DataSampler_percentage__allowed_calls(self):

        idx_percentage = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,1])

        try:
            sampling = preprocess.DataSampler(idx_percentage, idx_test=None, random_seed=None, verbose=False)
            (idx_train, idx_test) = sampling.percentage(0, test_selection_option=1)
            self.assertTrue(True)
        except Exception:
            self.assertTrue(False)

        try:
            sampling = preprocess.DataSampler(idx_percentage, idx_test=None, random_seed=None, verbose=False)
            (idx_train, idx_test) = sampling.percentage(20, test_selection_option=1)
            self.assertTrue(True)
        except Exception:
            self.assertTrue(False)

        try:
            sampling = preprocess.DataSampler(idx_percentage, idx_test=None, random_seed=None, verbose=False)
            (idx_train, idx_test) = sampling.percentage(60, test_selection_option=1)
            self.assertTrue(True)
        except Exception:
            self.assertTrue(False)

        try:
            sampling = preprocess.DataSampler(idx_percentage, idx_test=None, random_seed=None, verbose=False)
            (idx_train, idx_test) = sampling.percentage(100, test_selection_option=1)
            self.assertTrue(True)
        except Exception:
            self.assertTrue(False)

        try:
            sampling = preprocess.DataSampler(idx_percentage, idx_test=None, random_seed=None, verbose=False)
            (idx_train, idx_test) = sampling.percentage(10, test_selection_option=2)
            self.assertTrue(True)
        except Exception:
            self.assertTrue(False)

        try:
            sampling = preprocess.DataSampler(idx_percentage, idx_test=None, random_seed=None, verbose=False)
            (idx_train, idx_test) = sampling.percentage(50, test_selection_option=2)
            self.assertTrue(True)
        except Exception:
            self.assertTrue(False)

        try:
            sampling = preprocess.DataSampler(idx_percentage, idx_test=None, random_seed=None, verbose=False)
            sampling.idx_test = np.array([0, 0, 1, 10, 20, 30, 31])
            (idx_train, idx_test) = sampling.percentage(40, test_selection_option=1)
            self.assertTrue(True)
        except Exception:
            self.assertTrue(False)

# ------------------------------------------------------------------------------

    def test_preprocess__DataSampler_percentage__not_allowed_calls(self):
        idx_percentage = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,1])
        sampling = preprocess.DataSampler(idx_percentage, idx_test=None, random_seed=None, verbose=False)

        with self.assertRaises(ValueError):
            (idx_train, idx_test) = sampling.percentage(100, test_selection_option=2)

        sampling.idx_test = None
        with self.assertRaises(ValueError):
            (idx_train, idx_test) = sampling.percentage(60, test_selection_option=2)

        sampling.idx_test = np.array([0, 1, 2, 3, 10, 20, 21, 22, 23, 24, 30, 31])
        with self.assertRaises(ValueError):
            (idx_train, idx_test) = sampling.percentage(80, test_selection_option=1)

# ------------------------------------------------------------------------------
