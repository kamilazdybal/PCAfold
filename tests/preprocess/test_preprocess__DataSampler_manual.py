import unittest
import numpy as np
from PCAfold import preprocess
from PCAfold import reduction
from PCAfold import analysis

class Preprocess(unittest.TestCase):

    def test__DataSampler_manual__allowed_calls(self):

        idx_manual = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,1])

        # Calls that should work:
        try:
            sampling = preprocess.DataSampler(idx_manual, idx_test=None, random_seed=None, verbose=False)
            (idx_train, idx_test) = sampling.manual({0: 1, 1: 1})
            self.assertTrue(True)
        except Exception:
            self.assertTrue(False)

        try:
            sampling = preprocess.DataSampler(idx_manual, idx_test=None, random_seed=None, verbose=False)
            (idx_train, idx_test) = sampling.manual({0: 10, 1: 10}, sampling_type='percentage', test_selection_option=1)
            self.assertTrue(True)
        except Exception:
            self.assertTrue(False)

        try:
            sampling = preprocess.DataSampler(idx_manual, idx_test=None, random_seed=None, verbose=False)
            (idx_train, idx_test) = sampling.manual({0: 50, 1: 50}, sampling_type='percentage', test_selection_option=1)
            self.assertTrue(True)
        except Exception:
            self.assertTrue(False)

        try:
            sampling = preprocess.DataSampler(idx_manual, idx_test=None, random_seed=None, verbose=False)
            (idx_train, idx_test) = sampling.manual({0: 60, 1: 60}, sampling_type='percentage', test_selection_option=1)
            self.assertTrue(True)
        except Exception:
            self.assertTrue(False)

        try:
            sampling = preprocess.DataSampler(idx_manual, idx_test=None, random_seed=None, verbose=False)
            (idx_train, idx_test) = sampling.manual({0: 5, 1: 6}, sampling_type='number', test_selection_option=1)
            self.assertTrue(True)
        except Exception:
            self.assertTrue(False)

        try:
            sampling = preprocess.DataSampler(idx_manual, idx_test=None, random_seed=None, verbose=False)
            (idx_train, idx_test) = sampling.manual({0: 2, 1: 2}, sampling_type='number', test_selection_option=1)
            self.assertTrue(True)
        except Exception:
            self.assertTrue(False)

        try:
            sampling = preprocess.DataSampler(idx_manual, idx_test=None, random_seed=None, verbose=False)
            (idx_train, idx_test) = sampling.manual({0: 5, 1: 5}, sampling_type='number', test_selection_option=1)
            self.assertTrue(True)
        except Exception:
            self.assertTrue(False)

        try:
            sampling = preprocess.DataSampler(idx_manual, idx_test=None, random_seed=None, verbose=False)
            (idx_train, idx_test) = sampling.manual({0: 10, 1: 10}, sampling_type='percentage', test_selection_option=2)
            self.assertTrue(True)
        except Exception:
            self.assertTrue(False)

        try:
            sampling = preprocess.DataSampler(idx_manual, idx_test=None, random_seed=None, verbose=False)
            (idx_train, idx_test) = sampling.manual({0: 50, 1: 50}, sampling_type='percentage', test_selection_option=2)
            self.assertTrue(True)
        except Exception:
            self.assertTrue(False)

        try:
            sampling = preprocess.DataSampler(idx_manual, idx_test=None, random_seed=None, verbose=False)
            (idx_train, idx_test) = sampling.manual({0: 1, 1: 0}, sampling_type='number', test_selection_option=2)
            self.assertTrue(True)
        except Exception:
            self.assertTrue(False)

        try:
            sampling = preprocess.DataSampler(idx_manual, idx_test=None, random_seed=None, verbose=False)
            sampling.idx_test = np.array([0, 0, 1, 10, 20, 30, 31])
            (idx_train, idx_test) = sampling.manual({0: 2, 1: 2}, sampling_type='number', test_selection_option=1)
            self.assertTrue(True)
        except Exception:
            self.assertTrue(False)

        try:
            sampling = preprocess.DataSampler(idx_manual, idx_test=None, random_seed=None, verbose=False)
            sampling.idx_test = np.array([0, 1, 2, 3, 10, 20, 21, 22, 23, 24, 30, 31])
            (idx_train, idx_test) = sampling.manual({0: 70, 1: 10}, sampling_type='percentage', test_selection_option=1)
            self.assertTrue(True)
        except Exception:
            self.assertTrue(False)

# ------------------------------------------------------------------------------

    def test__DataSampler_manual__not_allowed_calls(self):

        idx_manual = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,1])
        sampling = preprocess.DataSampler(idx_manual, idx_test=None, random_seed=None, verbose=False)

        # Calls that should not work:
        with self.assertRaises(ValueError):
            (idx_train, idx_test) = sampling.manual({0: 20, 1: 20}, sampling_type='number', test_selection_option=1)

        with self.assertRaises(ValueError):
            (idx_train, idx_test) = sampling.manual({0: 2.2, 1: 1}, sampling_type='number', test_selection_option=1)

        with self.assertRaises(ValueError):
            (idx_train, idx_test) = sampling.manual({0: 1, 1: 1}, sampling_type='perc')

        with self.assertRaises(ValueError):
            (idx_train, idx_test) = sampling.manual({0: 20, 1: -20}, sampling_type='percentage', test_selection_option=1)

        with self.assertRaises(ValueError):
            (idx_train, idx_test) = sampling.manual({0: 51, 1: 10}, sampling_type='percentage', test_selection_option=2)

        with self.assertRaises(ValueError):
            (idx_train, idx_test) = sampling.manual({0: 15, 1: 2}, sampling_type='number', test_selection_option=2)

        with self.assertRaises(ValueError):
            (idx_train, idx_test) = sampling.manual({1: 1, 2: 1})

        sampling.idx_test = np.array([0, 1, 2, 3, 10, 20, 21, 22, 23, 24, 30, 31])
        with self.assertRaises(ValueError):
            (idx_train, idx_test) = sampling.manual({0: 2, 1: 8}, sampling_type='number', test_selection_option=1)

        sampling.idx_test = np.array([0, 1, 2, 3, 10, 20, 21, 22, 23, 24, 30, 31])
        with self.assertRaises(ValueError):
            (idx_train, idx_test) = sampling.manual({0: 75, 1: 10}, sampling_type='number', test_selection_option=1)

# ------------------------------------------------------------------------------
