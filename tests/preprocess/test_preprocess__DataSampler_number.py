import unittest
import numpy as np
from PCAfold import preprocess
from PCAfold import reduction
from PCAfold import analysis

class Preprocess(unittest.TestCase):

    def test_preprocess__DataSampler_number__allowed_calls(self):

        idx_number = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,1])
        sampling = preprocess.DataSampler(idx_number, idx_test=None, random_seed=None, verbose=False)

        try:
            sampling = preprocess.DataSampler(idx_number, idx_test=None, random_seed=None, verbose=False)
            (idx_train, idx_test) = sampling.number(40, test_selection_option=1)
            self.assertTrue(True)
        except Exception:
            self.assertTrue(False)

        try:
            sampling = preprocess.DataSampler(idx_number, idx_test=None, random_seed=None, verbose=False)
            (idx_train, idx_test) = sampling.number(70, test_selection_option=1)
            self.assertTrue(True)
        except Exception:
            self.assertTrue(False)

        try:
            sampling = preprocess.DataSampler(idx_number, idx_test=None, random_seed=None, verbose=False)
            (idx_train, idx_test) = sampling.number(40, test_selection_option=2)
            self.assertTrue(True)
        except Exception:
            self.assertTrue(False)

        try:
            sampling = preprocess.DataSampler(idx_number, idx_test=None, random_seed=None, verbose=False)
            (idx_train, idx_test) = sampling.number(70, test_selection_option=2)
            self.assertTrue(True)
        except Exception:
            self.assertTrue(False)

        try:
            sampling = preprocess.DataSampler(idx_number, idx_test=None, random_seed=None, verbose=False)
            (idx_train, idx_test) = sampling.number(0, test_selection_option=2)
            self.assertTrue(True)
        except Exception:
            self.assertTrue(False)

        try:
            sampling = preprocess.DataSampler(idx_number, idx_test=None, random_seed=None, verbose=False)
            sampling.idx_test = np.array([0, 0, 1, 10, 20, 30, 31])
            (idx_train, idx_test) = sampling.number(40, test_selection_option=1)
            self.assertTrue(True)
        except Exception:
            self.assertTrue(False)

# ------------------------------------------------------------------------------

    def test_preprocess__DataSampler_number__not_allowed_calls(self):

        idx_number = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,1])
        sampling = preprocess.DataSampler(idx_number, idx_test=None, random_seed=None, verbose=False)

        with self.assertRaises(ValueError):
            (idx_train, idx_test) = sampling.number(80, test_selection_option=1)

        with self.assertRaises(ValueError):
            (idx_train, idx_test) = sampling.number(80, test_selection_option=2)

        with self.assertRaises(ValueError):
            (idx_train, idx_test) = sampling.number(-2, test_selection_option=2)

        with self.assertRaises(ValueError):
            (idx_train, idx_test) = sampling.number(102, test_selection_option=2)

        sampling.idx_test = np.array([0, 1, 2, 3, 10, 20, 21, 22, 23, 24, 30, 31])
        with self.assertRaises(ValueError):
            (idx_train, idx_test) = sampling.number(46, test_selection_option=1)

# ------------------------------------------------------------------------------
