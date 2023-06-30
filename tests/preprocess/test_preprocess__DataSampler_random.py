import unittest
import numpy as np
from PCAfold import preprocess
from PCAfold import reduction
from PCAfold import analysis

class Preprocess(unittest.TestCase):

    def test_preprocess__DataSampler_random__allowed_calls(self):

        idx_random = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,1])

        try:
            sampling = preprocess.DataSampler(idx_random, idx_test=None, random_seed=None, verbose=False)
            (idx_train, idx_test) = sampling.random(40, test_selection_option=1)
            self.assertTrue(True)
        except Exception:
            self.assertTrue(False)

        try:
            sampling = preprocess.DataSampler(idx_random, idx_test=None, random_seed=None, verbose=False)
            (idx_train, idx_test) = sampling.random(0, test_selection_option=1)
            self.assertTrue(True)
        except Exception:
            self.assertTrue(False)

        try:
            sampling = preprocess.DataSampler(idx_random, idx_test=None, random_seed=None, verbose=False)
            (idx_train, idx_test) = sampling.random(51, test_selection_option=1)
            self.assertTrue(True)
        except Exception:
            self.assertTrue(False)

        try:
            sampling = preprocess.DataSampler(idx_random, idx_test=None, random_seed=None, verbose=False)
            (idx_train, idx_test) = sampling.random(100, test_selection_option=1)
            self.assertTrue(True)
        except Exception:
            self.assertTrue(False)

        try:
            sampling = preprocess.DataSampler(idx_random, idx_test=None, random_seed=None, verbose=False)
            (idx_train, idx_test) = sampling.random(0, test_selection_option=2)
            self.assertTrue(True)
        except Exception:
            self.assertTrue(False)

        try:
            idx_random = np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0])
            idx_test = np.array([1, 2, 3, 4, 5, 6])
            sampling = preprocess.DataSampler(idx_random, idx_test=idx_test, random_seed=None, verbose=False)
            (idx_train, idx_test) = sampling.random(70, test_selection_option=1)
            self.assertTrue(True)
        except Exception:
            self.assertTrue(False)

        try:
            idx_random = np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0])
            idx_test = np.array([1, 2, 3, 4, 5, 6])
            sampling = preprocess.DataSampler(idx_random, idx_test=idx_test, random_seed=None, verbose=False)
            (idx_train, idx_test) = sampling.random(10, test_selection_option=2)
            self.assertTrue(True)
        except Exception:
            self.assertTrue(False)

        try:
            idx_random = np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0])
            idx_test = np.array([1, 2, 3, 4, 5, 6])
            sampling = preprocess.DataSampler(idx_random, idx_test=idx_test, random_seed=None, verbose=False)
            (idx_train, idx_test) = sampling.random(70, test_selection_option=2)
            self.assertTrue(True)
        except Exception:
            self.assertTrue(False)

# ------------------------------------------------------------------------------

    def test_preprocess__DataSampler_random__not_allowed_calls(self):

        idx_random = np.zeros((100,1)).astype(int)
        idx_random[50:80,:] = 1
        idx_random[95:99,:] = 1

        sampling = preprocess.DataSampler(idx_random, idx_test=None, random_seed=None, verbose=False)

        with self.assertRaises(ValueError):
            (idx_train, idx_test) = sampling.random(101, test_selection_option=1)

        with self.assertRaises(ValueError):
            (idx_train, idx_test) = sampling.random(-1, test_selection_option=1)

        with self.assertRaises(ValueError):
            (idx_train, idx_test) = sampling.random(90, test_selection_option=2)

        with self.assertRaises(ValueError):
            (idx_train, idx_test) = sampling.random(51, test_selection_option=2)

        with self.assertRaises(ValueError):
            (idx_train, idx_test) = sampling.random(101, test_selection_option=2)

        idx_random = np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0])
        idx_test = np.array([1, 2, 3, 4, 5, 6])
        sampling = preprocess.DataSampler(idx_random, idx_test=idx_test, random_seed=None, verbose=False)
        with self.assertRaises(ValueError):
            (idx_train, idx_test) = sampling.random(80, test_selection_option=1)

        idx_random = np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0])
        idx_test = np.array([1, 2, 3, 4, 5, 6])
        sampling = preprocess.DataSampler(idx_random, idx_test=idx_test, random_seed=None, verbose=False)
        with self.assertRaises(ValueError):
            (idx_train, idx_test) = sampling.random(80, test_selection_option=2)

# ------------------------------------------------------------------------------
