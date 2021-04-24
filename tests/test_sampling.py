import unittest
import numpy as np
from PCAfold import DataSampler


class TestSampling(unittest.TestCase):

    def test_test_selection_option_1(self):

        idx = np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1])
        sam = DataSampler(idx)

        (idx_train, idx_test) = sam.number(10)
        n_observations = len(idx)
        self.assertTrue(np.size(idx_test) + np.size(idx_train) == n_observations)

        (idx_train, idx_test) = sam.percentage(10)
        self.assertTrue(np.size(idx_test) + np.size(idx_train) == n_observations)

        (idx_train, idx_test) = sam.manual({0: 1, 1: 4}, sampling_type='number')
        self.assertTrue(np.size(idx_test) + np.size(idx_train) == n_observations)

        (idx_train, idx_test) = sam.random(10)
        self.assertTrue(np.size(idx_test) + np.size(idx_train) == n_observations)

    def test_test_selection_option_2(self):

        idx = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        sam = DataSampler(idx)

        (idx_train, idx_test) = sam.number(10, test_selection_option=2)
        n_observations = len(idx)
        self.assertTrue(np.size(idx_test) + np.size(idx_train) < n_observations)

        (idx_train, idx_test) = sam.percentage(10, test_selection_option=2)
        self.assertTrue(np.size(idx_test) + np.size(idx_train) < n_observations)

        (idx_train, idx_test) = sam.manual({0: 1, 1: 4}, sampling_type='number', test_selection_option=2)
        self.assertTrue(np.size(idx_test) + np.size(idx_train) < n_observations)

        (idx_train, idx_test) = sam.random(10, test_selection_option=2)
        self.assertTrue(np.size(idx_test) + np.size(idx_train) < n_observations)

    def test_disjoint_train_test_samples(self):

        idx = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 5, 5, 5, 5, 10, 10, 10, 10, 10])
        sam = DataSampler(idx)

        (idx_train, idx_test) = sam.number(40, test_selection_option=1)
        self.assertTrue(len(np.setdiff1d(idx_test, idx_train)) != 0)
        self.assertTrue(not np.any(np.in1d(idx_train, idx_test)))

        (idx_train, idx_test) = sam.percentage(40, test_selection_option=1)
        self.assertTrue(len(np.setdiff1d(idx_test, idx_train)) != 0)
        self.assertTrue(not np.any(np.in1d(idx_train, idx_test)))

        (idx_train, idx_test) = sam.manual({0: 1, 1: 4, 2: 1, 3: 1, 4: 1}, sampling_type='number',
                                           test_selection_option=1)
        self.assertTrue(len(np.setdiff1d(idx_test, idx_train)) != 0)
        self.assertTrue(not np.any(np.in1d(idx_train, idx_test)))

        (idx_train, idx_test) = sam.random(40, test_selection_option=1)
        self.assertTrue(len(np.setdiff1d(idx_test, idx_train)) != 0)
        self.assertTrue(not np.any(np.in1d(idx_train, idx_test)))

        (idx_train, idx_test) = sam.number(40, test_selection_option=2)
        self.assertTrue(len(np.setdiff1d(idx_test, idx_train)) != 0)
        self.assertTrue(not np.any(np.in1d(idx_train, idx_test)))

        (idx_train, idx_test) = sam.percentage(40, test_selection_option=2)
        self.assertTrue(len(np.setdiff1d(idx_test, idx_train)) != 0)
        self.assertTrue(not np.any(np.in1d(idx_train, idx_test)))

        (idx_train, idx_test) = sam.manual({0: 1, 1: 1, 2: 1, 3: 1, 4: 1}, sampling_type='number',
                                           test_selection_option=2)
        self.assertTrue(len(np.setdiff1d(idx_test, idx_train)) != 0)
        self.assertTrue(not np.any(np.in1d(idx_train, idx_test)))

        (idx_train, idx_test) = sam.random(10, test_selection_option=2)
        self.assertTrue(len(np.setdiff1d(idx_test, idx_train)) != 0)
        self.assertTrue(not np.any(np.in1d(idx_train, idx_test)))

        (idx_train, idx_test) = sam.manual({0: 10, 1: 10, 2: 60, 3: 10, 4: 80}, sampling_type='percentage',
                                           test_selection_option=1)
        self.assertTrue(len(np.setdiff1d(idx_test, idx_train)) != 0)
        self.assertTrue(not np.any(np.in1d(idx_train, idx_test)))

        (idx_train, idx_test) = sam.manual({0: 40, 1: 40, 2: 40, 3: 40, 4: 40}, sampling_type='percentage',
                                           test_selection_option=2)
        self.assertTrue(len(np.setdiff1d(idx_test, idx_train)) != 0)
        self.assertTrue(not np.any(np.in1d(idx_train, idx_test)))

    def test_predefined_idx_test(self):
        idx = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 5, 5, 5, 5, 10, 10, 10, 10, 10])
        idx_test_predefined = [0, 10, 20, 21]
        sam = DataSampler(idx, idx_test=idx_test_predefined)

        (idx_train, idx_test) = sam.number(40, test_selection_option=1)
        self.assertTrue(len(np.setdiff1d(idx_test, idx_train)) != 0)
        self.assertTrue(not np.any(np.in1d(idx_train, idx_test)))
        self.assertTrue(np.array_equal(idx_test_predefined, idx_test))

        (idx_train, idx_test) = sam.percentage(40, test_selection_option=1)
        self.assertTrue(len(np.setdiff1d(idx_test, idx_train)) != 0)
        self.assertTrue(not np.any(np.in1d(idx_train, idx_test)))
        self.assertTrue(np.array_equal(idx_test_predefined, idx_test))

        (idx_train, idx_test) = sam.manual({0: 1, 1: 1, 2: 1, 3: 1, 4: 1}, sampling_type='number',
                                           test_selection_option=1)
        self.assertTrue(len(np.setdiff1d(idx_test, idx_train)) != 0)
        self.assertTrue(not np.any(np.in1d(idx_train, idx_test)))
        self.assertTrue(np.array_equal(idx_test_predefined, idx_test))

        (idx_train, idx_test) = sam.random(40, test_selection_option=1)
        self.assertTrue(len(np.setdiff1d(idx_test, idx_train)) != 0)
        self.assertTrue(not np.any(np.in1d(idx_train, idx_test)))
        self.assertTrue(np.array_equal(idx_test_predefined, idx_test))

        (idx_train, idx_test) = sam.number(40, test_selection_option=2)
        self.assertTrue(len(np.setdiff1d(idx_test, idx_train)) != 0)
        self.assertTrue(not np.any(np.in1d(idx_train, idx_test)))
        self.assertTrue(np.array_equal(idx_test_predefined, idx_test))

        (idx_train, idx_test) = sam.percentage(40, test_selection_option=2)
        self.assertTrue(len(np.setdiff1d(idx_test, idx_train)) != 0)
        self.assertTrue(not np.any(np.in1d(idx_train, idx_test)))
        self.assertTrue(np.array_equal(idx_test_predefined, idx_test))

        (idx_train, idx_test) = sam.manual({0: 1, 1: 1, 2: 1, 3: 1, 4: 1}, sampling_type='number',
                                           test_selection_option=2)
        self.assertTrue(len(np.setdiff1d(idx_test, idx_train)) != 0)
        self.assertTrue(not np.any(np.in1d(idx_train, idx_test)))
        self.assertTrue(np.array_equal(idx_test_predefined, idx_test))

        (idx_train, idx_test) = sam.random(10, test_selection_option=2)
        self.assertTrue(len(np.setdiff1d(idx_test, idx_train)) != 0)
        self.assertTrue(not np.any(np.in1d(idx_train, idx_test)))
        self.assertTrue(np.array_equal(idx_test_predefined, idx_test))

        (idx_train, idx_test) = sam.manual({0: 20, 1: 80, 2: 60, 3: 80, 4: 80}, sampling_type='percentage',
                                           test_selection_option=1)
        self.assertTrue(len(np.setdiff1d(idx_test, idx_train)) != 0)
        self.assertTrue(not np.any(np.in1d(idx_train, idx_test)))
        self.assertTrue(np.array_equal(idx_test_predefined, idx_test))

        (idx_train, idx_test) = sam.manual({0: 40, 1: 40, 2: 40, 3: 40, 4: 40}, sampling_type='percentage',
                                           test_selection_option=2)
        self.assertTrue(len(np.setdiff1d(idx_test, idx_train)) != 0)
        self.assertTrue(not np.any(np.in1d(idx_train, idx_test)))
        self.assertTrue(np.array_equal(idx_test_predefined, idx_test))


class TestDataSampler(unittest.TestCase):

    def test_DataSampler_allowed_initializations(self):

        try:
            DataSampler(np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]))
            self.assertTrue(True)
        except Exception:
            self.assertTrue(False)

        try:
            DataSampler(np.array([1, 1, 1, 1, 2, 2, 2, 2]))
            self.assertTrue(True)
        except Exception:
            self.assertTrue(False)

        try:
            DataSampler(np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]),
                        idx_test=np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 2, 2]))
            self.assertTrue(True)
        except Exception:
            self.assertTrue(False)

        try:
            sam = DataSampler(np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]))
            sam.idx = np.array([1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2])
            sam.idx_test = np.arange(1, 10, 1)
            sam.random_seed = 100
            sam.random_seed = None
            sam.random_seed = -1
            sam.verbose = False
            sam.verbose = True
            sam.idx_test = []
            self.assertTrue(True)
        except Exception:
            self.assertTrue(False)

    def test_DataSampler_not_allowed_initializations(self):

        with self.assertRaises(ValueError):
            DataSampler(np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]), idx_test=[], random_seed=0.4, verbose=False)

        with self.assertRaises(ValueError):
            DataSampler(np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]), idx_test=[], random_seed=100, verbose=2)

        with self.assertRaises(ValueError):
            DataSampler(np.array([0, 0, 0, 1, 1]), idx_test=np.array([1, 2, 3, 4, 5, 6, 7, 8]), random_seed=100,
                        verbose=False)

        with self.assertRaises(ValueError):
            DataSampler(np.array([]), idx_test=[], random_seed=None, verbose=False)

        with self.assertRaises(ValueError):
            sam = DataSampler(np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]))
            sam.idx_test = np.arange(1, 100, 1)

        with self.assertRaises(ValueError):
            sam = DataSampler(np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]))
            sam.random_seed = 10.1

        with self.assertRaises(ValueError):
            sam = DataSampler(np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]))
            sam.random_seed = False

        with self.assertRaises(ValueError):
            sam = DataSampler(np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]))
            sam.verbose = 10

        with self.assertRaises(ValueError):
            sam = DataSampler(np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]))
            sam.idx = []

        with self.assertRaises(ValueError):
            sam = DataSampler(np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]))
            sam.idx_test = [0, 1, 2, 3, 4, 5, 6]
            sam.idx = np.array([0, 1])

        with self.assertRaises(ValueError):
            sam = DataSampler(np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]))
            sam.idx_test = 'hello'

        with self.assertRaises(ValueError):
            sam = DataSampler(np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]))
            sam.idx = 'hello'

    def test_number_allowed_calls(self):

        idx_number = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,1])
        sampling = DataSampler(idx_number, idx_test=[], random_seed=None, verbose=False)

        try:
            sampling = DataSampler(idx_number, idx_test=[], random_seed=None, verbose=False)
            (idx_train, idx_test) = sampling.number(40, test_selection_option=1)
            self.assertTrue(True)
        except Exception:
            self.assertTrue(False)

        try:
            sampling = DataSampler(idx_number, idx_test=[], random_seed=None, verbose=False)
            (idx_train, idx_test) = sampling.number(70, test_selection_option=1)
            self.assertTrue(True)
        except Exception:
            self.assertTrue(False)

        try:
            sampling = DataSampler(idx_number, idx_test=[], random_seed=None, verbose=False)
            (idx_train, idx_test) = sampling.number(40, test_selection_option=2)
            self.assertTrue(True)
        except Exception:
            self.assertTrue(False)

        try:
            sampling = DataSampler(idx_number, idx_test=[], random_seed=None, verbose=False)
            (idx_train, idx_test) = sampling.number(70, test_selection_option=2)
            self.assertTrue(True)
        except Exception:
            self.assertTrue(False)

        try:
            sampling = DataSampler(idx_number, idx_test=[], random_seed=None, verbose=False)
            (idx_train, idx_test) = sampling.number(0, test_selection_option=2)
            self.assertTrue(True)
        except Exception:
            self.assertTrue(False)

        try:
            sampling = DataSampler(idx_number, idx_test=[], random_seed=None, verbose=False)
            sampling.idx_test = np.array([0, 0, 1, 10, 20, 30, 31])
            (idx_train, idx_test) = sampling.number(40, test_selection_option=1)
            self.assertTrue(True)
        except Exception:
            self.assertTrue(False)

    def test_number_not_allowed_calls(self):

        idx_number = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,1])
        sampling = DataSampler(idx_number, idx_test=[], random_seed=None, verbose=False)

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

    def test_percentage_allowed_calls(self):

        idx_percentage = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,1])

        try:
            sampling = DataSampler(idx_percentage, idx_test=[], random_seed=None, verbose=False)
            (idx_train, idx_test) = sampling.percentage(0, test_selection_option=1)
            self.assertTrue(True)
        except Exception:
            self.assertTrue(False)

        try:
            sampling = DataSampler(idx_percentage, idx_test=[], random_seed=None, verbose=False)
            (idx_train, idx_test) = sampling.percentage(20, test_selection_option=1)
            self.assertTrue(True)
        except Exception:
            self.assertTrue(False)

        try:
            sampling = DataSampler(idx_percentage, idx_test=[], random_seed=None, verbose=False)
            (idx_train, idx_test) = sampling.percentage(60, test_selection_option=1)
            self.assertTrue(True)
        except Exception:
            self.assertTrue(False)

        try:
            sampling = DataSampler(idx_percentage, idx_test=[], random_seed=None, verbose=False)
            (idx_train, idx_test) = sampling.percentage(100, test_selection_option=1)
            self.assertTrue(True)
        except Exception:
            self.assertTrue(False)

        try:
            sampling = DataSampler(idx_percentage, idx_test=[], random_seed=None, verbose=False)
            (idx_train, idx_test) = sampling.percentage(10, test_selection_option=2)
            self.assertTrue(True)
        except Exception:
            self.assertTrue(False)

        try:
            sampling = DataSampler(idx_percentage, idx_test=[], random_seed=None, verbose=False)
            (idx_train, idx_test) = sampling.percentage(50, test_selection_option=2)
            self.assertTrue(True)
        except Exception:
            self.assertTrue(False)

        try:
            sampling = DataSampler(idx_percentage, idx_test=[], random_seed=None, verbose=False)
            sampling.idx_test = np.array([0, 0, 1, 10, 20, 30, 31])
            (idx_train, idx_test) = sampling.percentage(40, test_selection_option=1)
            self.assertTrue(True)
        except Exception:
            self.assertTrue(False)

    def test_percentage_not_allowed_calls(self):
        idx_percentage = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,1])
        sampling = DataSampler(idx_percentage, idx_test=[], random_seed=None, verbose=False)

        with self.assertRaises(ValueError):
            (idx_train, idx_test) = sampling.percentage(100, test_selection_option=2)

        sampling.idx_test = []
        with self.assertRaises(ValueError):
            (idx_train, idx_test) = sampling.percentage(60, test_selection_option=2)

        sampling.idx_test = np.array([0, 1, 2, 3, 10, 20, 21, 22, 23, 24, 30, 31])
        with self.assertRaises(ValueError):
            (idx_train, idx_test) = sampling.percentage(80, test_selection_option=1)

    def test_manual_allowed_calls(self):

        idx_manual = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,1])

        # Calls that should work:
        try:
            sampling = DataSampler(idx_manual, idx_test=[], random_seed=None, verbose=False)
            (idx_train, idx_test) = sampling.manual({0: 1, 1: 1})
            self.assertTrue(True)
        except Exception:
            self.assertTrue(False)

        try:
            sampling = DataSampler(idx_manual, idx_test=[], random_seed=None, verbose=False)
            (idx_train, idx_test) = sampling.manual({0: 10, 1: 10}, sampling_type='percentage', test_selection_option=1)
            self.assertTrue(True)
        except Exception:
            self.assertTrue(False)

        try:
            sampling = DataSampler(idx_manual, idx_test=[], random_seed=None, verbose=False)
            (idx_train, idx_test) = sampling.manual({0: 50, 1: 50}, sampling_type='percentage', test_selection_option=1)
            self.assertTrue(True)
        except Exception:
            self.assertTrue(False)

        try:
            sampling = DataSampler(idx_manual, idx_test=[], random_seed=None, verbose=False)
            (idx_train, idx_test) = sampling.manual({0: 60, 1: 60}, sampling_type='percentage', test_selection_option=1)
            self.assertTrue(True)
        except Exception:
            self.assertTrue(False)

        try:
            sampling = DataSampler(idx_manual, idx_test=[], random_seed=None, verbose=False)
            (idx_train, idx_test) = sampling.manual({0: 5, 1: 6}, sampling_type='number', test_selection_option=1)
            self.assertTrue(True)
        except Exception:
            self.assertTrue(False)

        try:
            sampling = DataSampler(idx_manual, idx_test=[], random_seed=None, verbose=False)
            (idx_train, idx_test) = sampling.manual({0: 2, 1: 2}, sampling_type='number', test_selection_option=1)
            self.assertTrue(True)
        except Exception:
            self.assertTrue(False)

        try:
            sampling = DataSampler(idx_manual, idx_test=[], random_seed=None, verbose=False)
            (idx_train, idx_test) = sampling.manual({0: 5, 1: 5}, sampling_type='number', test_selection_option=1)
            self.assertTrue(True)
        except Exception:
            self.assertTrue(False)

        try:
            sampling = DataSampler(idx_manual, idx_test=[], random_seed=None, verbose=False)
            (idx_train, idx_test) = sampling.manual({0: 10, 1: 10}, sampling_type='percentage', test_selection_option=2)
            self.assertTrue(True)
        except Exception:
            self.assertTrue(False)

        try:
            sampling = DataSampler(idx_manual, idx_test=[], random_seed=None, verbose=False)
            (idx_train, idx_test) = sampling.manual({0: 50, 1: 50}, sampling_type='percentage', test_selection_option=2)
            self.assertTrue(True)
        except Exception:
            self.assertTrue(False)

        try:
            sampling = DataSampler(idx_manual, idx_test=[], random_seed=None, verbose=False)
            (idx_train, idx_test) = sampling.manual({0: 1, 1: 0}, sampling_type='number', test_selection_option=2)
            self.assertTrue(True)
        except Exception:
            self.assertTrue(False)

        try:
            sampling = DataSampler(idx_manual, idx_test=[], random_seed=None, verbose=False)
            sampling.idx_test = np.array([0, 0, 1, 10, 20, 30, 31])
            (idx_train, idx_test) = sampling.manual({0: 2, 1: 2}, sampling_type='number', test_selection_option=1)
            self.assertTrue(True)
        except Exception:
            self.assertTrue(False)

        try:
            sampling = DataSampler(idx_manual, idx_test=[], random_seed=None, verbose=False)
            sampling.idx_test = np.array([0, 1, 2, 3, 10, 20, 21, 22, 23, 24, 30, 31])
            (idx_train, idx_test) = sampling.manual({0: 70, 1: 10}, sampling_type='percentage', test_selection_option=1)
            self.assertTrue(True)
        except Exception:
            self.assertTrue(False)

    def test_manual_not_allowed_calls(self):

        idx_manual = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,1])
        sampling = DataSampler(idx_manual, idx_test=[], random_seed=None, verbose=False)

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

    def test_random_allowed_calls(self):

        idx_random = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,1])

        try:
            sampling = DataSampler(idx_random, idx_test=[], random_seed=None, verbose=False)
            (idx_train, idx_test) = sampling.random(40, test_selection_option=1)
            self.assertTrue(True)
        except Exception:
            self.assertTrue(False)

        try:
            sampling = DataSampler(idx_random, idx_test=[], random_seed=None, verbose=False)
            (idx_train, idx_test) = sampling.random(0, test_selection_option=1)
            self.assertTrue(True)
        except Exception:
            self.assertTrue(False)

        try:
            sampling = DataSampler(idx_random, idx_test=[], random_seed=None, verbose=False)
            (idx_train, idx_test) = sampling.random(51, test_selection_option=1)
            self.assertTrue(True)
        except Exception:
            self.assertTrue(False)

        try:
            sampling = DataSampler(idx_random, idx_test=[], random_seed=None, verbose=False)
            (idx_train, idx_test) = sampling.random(100, test_selection_option=1)
            self.assertTrue(True)
        except Exception:
            self.assertTrue(False)

        try:
            sampling = DataSampler(idx_random, idx_test=[], random_seed=None, verbose=False)
            (idx_train, idx_test) = sampling.random(0, test_selection_option=2)
            self.assertTrue(True)
        except Exception:
            self.assertTrue(False)

        try:
            idx_random = np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0])
            idx_test = [1, 2, 3, 4, 5, 6]
            sampling = DataSampler(idx_random, idx_test=idx_test, random_seed=None, verbose=False)
            (idx_train, idx_test) = sampling.random(70, test_selection_option=1)
            self.assertTrue(True)
        except Exception:
            self.assertTrue(False)

        try:
            sampling = DataSampler(idx_random, idx_test=idx_test, random_seed=None, verbose=False)
            (idx_train, idx_test) = sampling.random(10, test_selection_option=2)
            self.assertTrue(True)
        except Exception:
            self.assertTrue(False)

        try:
            idx_random = np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0])
            idx_test = [1, 2, 3, 4, 5, 6]
            sampling = DataSampler(idx_random, idx_test=idx_test, random_seed=None, verbose=False)
            (idx_train, idx_test) = sampling.random(70, test_selection_option=2)
            self.assertTrue(True)
        except Exception:
            self.assertTrue(False)

    def test_random_not_allowed_calls(self):

        idx_random = np.zeros((100,1))
        idx_random[50:80,:] = 1
        idx_random[95:99,:] = 1

        sampling = DataSampler(idx_random, idx_test=[], random_seed=None, verbose=False)

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
        idx_test = [1, 2, 3, 4, 5, 6]
        sampling = DataSampler(idx_random, idx_test=idx_test, random_seed=None, verbose=False)
        with self.assertRaises(ValueError):
            (idx_train, idx_test) = sampling.random(80, test_selection_option=1)

        idx_random = np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0])
        idx_test = [1, 2, 3, 4, 5, 6]
        sampling = DataSampler(idx_random, idx_test=idx_test, random_seed=None, verbose=False)
        with self.assertRaises(ValueError):
            (idx_train, idx_test) = sampling.random(80, test_selection_option=2)

class TestRandomSeed(unittest.TestCase):

    def test_random_seed_number(self):

        idx = np.zeros((1000,))
        idx[500:800] = 1

        try:
            sampling = DataSampler(idx, idx_test=[], random_seed=123, verbose=False)
            (idx_train, idx_test) = sampling.number(40, test_selection_option=1)

            for i in range(0,10):
                sampling = DataSampler(idx, idx_test=[], random_seed=123, verbose=False)
                (idx_train_i, idx_test_i) = sampling.number(40, test_selection_option=1)
                self.assertTrue((idx_train_i == idx_train).all())
        except Exception:
            self.assertTrue(False)

        try:
            sampling = DataSampler(idx, idx_test=[], random_seed=1234, verbose=False)
            (idx_train, idx_test) = sampling.number(40, test_selection_option=1)

            for i in range(0,10):
                sampling = DataSampler(idx, idx_test=[], random_seed=1234, verbose=False)
                (idx_train_i, idx_test_i) = sampling.number(40, test_selection_option=1)
                self.assertTrue((idx_train_i == idx_train).all())
        except Exception:
            self.assertTrue(False)

        try:
            sampling = DataSampler(idx, idx_test=[], random_seed=123, verbose=False)
            (idx_train, idx_test) = sampling.number(40, test_selection_option=1)

            for i in range(0,10):
                sampling = DataSampler(idx, idx_test=[], random_seed=123, verbose=False)
                (idx_train_i, idx_test_i) = sampling.number(40, test_selection_option=1)
                self.assertTrue((idx_train_i == idx_train).all())
        except Exception:
            self.assertTrue(False)

        try:
            sampling = DataSampler(idx, idx_test=[], random_seed=1234, verbose=False)
            (idx_train, idx_test) = sampling.number(40, test_selection_option=2)

            for i in range(0,10):
                sampling = DataSampler(idx, idx_test=[], random_seed=1234, verbose=False)
                (idx_train_i, idx_test_i) = sampling.number(40, test_selection_option=2)
                self.assertTrue((idx_train_i == idx_train).all())
        except Exception:
            self.assertTrue(False)

            sampling = DataSampler(idx, idx_test=[1,100,200], random_seed=123, verbose=False)
            (idx_train, idx_test) = sampling.number(40, test_selection_option=1)

            for i in range(0,10):
                sampling = DataSampler(idx, idx_test=[1,100,200], random_seed=123, verbose=False)
                (idx_train_i, idx_test_i) = sampling.number(40, test_selection_option=1)
                self.assertTrue((idx_train_i == idx_train).all())
        except Exception:
            self.assertTrue(False)

        try:
            sampling = DataSampler(idx, idx_test=[1,100,200], random_seed=1234, verbose=False)
            (idx_train, idx_test) = sampling.number(40, test_selection_option=2)

            for i in range(0,10):
                sampling = DataSampler(idx, idx_test=[1,100,200], random_seed=1234, verbose=False)
                (idx_train_i, idx_test_i) = sampling.number(40, test_selection_option=2)
                self.assertTrue((idx_train_i == idx_train).all())
        except Exception:
            self.assertTrue(False)

    def test_random_seed_percentage(self):

        idx = np.zeros((1000,))
        idx[500:800] = 1

        idx = idx.astype(int)

        try:
            sampling = DataSampler(idx, idx_test=[], random_seed=123, verbose=False)
            (idx_train, idx_test) = sampling.percentage(40, test_selection_option=1)

            for i in range(0,10):
                sampling = DataSampler(idx, idx_test=[], random_seed=123, verbose=False)
                (idx_train_i, idx_test_i) = sampling.percentage(40, test_selection_option=1)
                self.assertTrue((idx_train_i == idx_train).all())
        except Exception:
            self.assertTrue(False)

        try:
            sampling = DataSampler(idx, idx_test=[], random_seed=1234, verbose=False)
            (idx_train, idx_test) = sampling.percentage(40, test_selection_option=1)

            for i in range(0,10):
                sampling = DataSampler(idx, idx_test=[], random_seed=1234, verbose=False)
                (idx_train_i, idx_test_i) = sampling.percentage(40, test_selection_option=1)
                self.assertTrue((idx_train_i == idx_train).all())
        except Exception:
            self.assertTrue(False)

        try:
            sampling = DataSampler(idx, idx_test=[], random_seed=123, verbose=False)
            (idx_train, idx_test) = sampling.percentage(40, test_selection_option=1)

            for i in range(0,10):
                sampling = DataSampler(idx, idx_test=[], random_seed=123, verbose=False)
                (idx_train_i, idx_test_i) = sampling.percentage(40, test_selection_option=1)
                self.assertTrue((idx_train_i == idx_train).all())
        except Exception:
            self.assertTrue(False)

        try:
            sampling = DataSampler(idx, idx_test=[], random_seed=1234, verbose=False)
            (idx_train, idx_test) = sampling.percentage(40, test_selection_option=2)

            for i in range(0,10):
                sampling = DataSampler(idx, idx_test=[], random_seed=1234, verbose=False)
                (idx_train_i, idx_test_i) = sampling.percentage(40, test_selection_option=2)
                self.assertTrue((idx_train_i == idx_train).all())
        except Exception:
            self.assertTrue(False)

            sampling = DataSampler(idx, idx_test=[1,100,200], random_seed=123, verbose=False)
            (idx_train, idx_test) = sampling.percentage(40, test_selection_option=1)

            for i in range(0,10):
                sampling = DataSampler(idx, idx_test=[1,100,200], random_seed=123, verbose=False)
                (idx_train_i, idx_test_i) = sampling.percentage(40, test_selection_option=1)
                self.assertTrue((idx_train_i == idx_train).all())
        except Exception:
            self.assertTrue(False)

        try:
            sampling = DataSampler(idx, idx_test=[1,100,200], random_seed=1234, verbose=False)
            (idx_train, idx_test) = sampling.percentage(40, test_selection_option=2)

            for i in range(0,10):
                sampling = DataSampler(idx, idx_test=[1,100,200], random_seed=1234, verbose=False)
                (idx_train_i, idx_test_i) = sampling.percentage(40, test_selection_option=2)
                self.assertTrue((idx_train_i == idx_train).all())
        except Exception:
            self.assertTrue(False)

    def test_random_seed_manual_by_number(self):

        idx = np.zeros((1000,))
        idx[500:800] = 1

        idx = idx.astype(int)

        sampling_dictionary_number = {0:300, 1:150}

        try:
            sampling = DataSampler(idx, idx_test=[], random_seed=123, verbose=False)
            (idx_train, idx_test) = sampling.manual(sampling_dictionary_number, sampling_type='number', test_selection_option=1)

            for i in range(0,10):
                sampling = DataSampler(idx, idx_test=[], random_seed=123, verbose=False)
                (idx_train_i, idx_test_i) = sampling.manual(sampling_dictionary_number, sampling_type='number', test_selection_option=1)
                self.assertTrue((idx_train_i == idx_train).all())
        except Exception:
            self.assertTrue(False)

        try:
            sampling = DataSampler(idx, idx_test=[], random_seed=1234, verbose=False)
            (idx_train, idx_test) = sampling.manual(sampling_dictionary_number, sampling_type='number', test_selection_option=1)

            for i in range(0,10):
                sampling = DataSampler(idx, idx_test=[], random_seed=1234, verbose=False)
                (idx_train_i, idx_test_i) = sampling.manual(sampling_dictionary_number, sampling_type='number', test_selection_option=1)
                self.assertTrue((idx_train_i == idx_train).all())
        except Exception:
            self.assertTrue(False)

        try:
            sampling = DataSampler(idx, idx_test=[], random_seed=123, verbose=False)
            (idx_train, idx_test) = sampling.manual(sampling_dictionary_number, sampling_type='number', test_selection_option=1)

            for i in range(0,10):
                sampling = DataSampler(idx, idx_test=[], random_seed=123, verbose=False)
                (idx_train_i, idx_test_i) = sampling.manual(sampling_dictionary_number, sampling_type='number', test_selection_option=1)
                self.assertTrue((idx_train_i == idx_train).all())
        except Exception:
            self.assertTrue(False)

        try:
            sampling = DataSampler(idx, idx_test=[], random_seed=1234, verbose=False)
            (idx_train, idx_test) = sampling.manual(sampling_dictionary_number, sampling_type='number', test_selection_option=2)

            for i in range(0,10):
                sampling = DataSampler(idx, idx_test=[], random_seed=1234, verbose=False)
                (idx_train_i, idx_test_i) = sampling.manual(sampling_dictionary_number, sampling_type='number', test_selection_option=2)
                self.assertTrue((idx_train_i == idx_train).all())
        except Exception:
            self.assertTrue(False)

            sampling = DataSampler(idx, idx_test=[1,100,200], random_seed=123, verbose=False)
            (idx_train, idx_test) = sampling.manual(sampling_dictionary_number, sampling_type='number', test_selection_option=1)

            for i in range(0,10):
                sampling = DataSampler(idx, idx_test=[1,100,200], random_seed=123, verbose=False)
                (idx_train_i, idx_test_i) = sampling.manual(sampling_dictionary_number, sampling_type='number', test_selection_option=1)
                self.assertTrue((idx_train_i == idx_train).all())
        except Exception:
            self.assertTrue(False)

        try:
            sampling = DataSampler(idx, idx_test=[1,100,200], random_seed=1234, verbose=False)
            (idx_train, idx_test) = sampling.manual(sampling_dictionary_number, sampling_type='number', test_selection_option=2)

            for i in range(0,10):
                sampling = DataSampler(idx, idx_test=[1,100,200], random_seed=1234, verbose=False)
                (idx_train_i, idx_test_i) = sampling.manual(sampling_dictionary_number, sampling_type='number', test_selection_option=2)
                self.assertTrue((idx_train_i == idx_train).all())
        except Exception:
            self.assertTrue(False)

    def test_random_seed_manual_by_percentage(self):

        idx = np.zeros((1000,))
        idx[500:800] = 1

        idx = idx.astype(int)

        sampling_dictionary_number = {0:30, 1:15}

        try:
            sampling = DataSampler(idx, idx_test=[], random_seed=123, verbose=False)
            (idx_train, idx_test) = sampling.manual(sampling_dictionary_number, sampling_type='percentage', test_selection_option=1)

            for i in range(0,10):
                sampling = DataSampler(idx, idx_test=[], random_seed=123, verbose=False)
                (idx_train_i, idx_test_i) = sampling.manual(sampling_dictionary_number, sampling_type='percentage', test_selection_option=1)
                self.assertTrue((idx_train_i == idx_train).all())
        except Exception:
            self.assertTrue(False)

        try:
            sampling = DataSampler(idx, idx_test=[], random_seed=1234, verbose=False)
            (idx_train, idx_test) = sampling.manual(sampling_dictionary_number, sampling_type='percentage', test_selection_option=1)

            for i in range(0,10):
                sampling = DataSampler(idx, idx_test=[], random_seed=1234, verbose=False)
                (idx_train_i, idx_test_i) = sampling.manual(sampling_dictionary_number, sampling_type='percentage', test_selection_option=1)
                self.assertTrue((idx_train_i == idx_train).all())
        except Exception:
            self.assertTrue(False)

        try:
            sampling = DataSampler(idx, idx_test=[], random_seed=123, verbose=False)
            (idx_train, idx_test) = sampling.manual(sampling_dictionary_number, sampling_type='percentage', test_selection_option=1)

            for i in range(0,10):
                sampling = DataSampler(idx, idx_test=[], random_seed=123, verbose=False)
                (idx_train_i, idx_test_i) = sampling.manual(sampling_dictionary_number, sampling_type='percentage', test_selection_option=1)
                self.assertTrue((idx_train_i == idx_train).all())
        except Exception:
            self.assertTrue(False)

        try:
            sampling = DataSampler(idx, idx_test=[], random_seed=1234, verbose=False)
            (idx_train, idx_test) = sampling.manual(sampling_dictionary_number, sampling_type='percentage', test_selection_option=2)

            for i in range(0,10):
                sampling = DataSampler(idx, idx_test=[], random_seed=1234, verbose=False)
                (idx_train_i, idx_test_i) = sampling.manual(sampling_dictionary_number, sampling_type='percentage', test_selection_option=2)
                self.assertTrue((idx_train_i == idx_train).all())
        except Exception:
            self.assertTrue(False)

            sampling = DataSampler(idx, idx_test=[1,100,200], random_seed=123, verbose=False)
            (idx_train, idx_test) = sampling.manual(sampling_dictionary_number, sampling_type='percentage', test_selection_option=1)

            for i in range(0,10):
                sampling = DataSampler(idx, idx_test=[1,100,200], random_seed=123, verbose=False)
                (idx_train_i, idx_test_i) = sampling.manual(sampling_dictionary_number, sampling_type='percentage', test_selection_option=1)
                self.assertTrue((idx_train_i == idx_train).all())
        except Exception:
            self.assertTrue(False)

        try:
            sampling = DataSampler(idx, idx_test=[1,100,200], random_seed=1234, verbose=False)
            (idx_train, idx_test) = sampling.manual(sampling_dictionary_number, sampling_type='percentage', test_selection_option=2)

            for i in range(0,10):
                sampling = DataSampler(idx, idx_test=[1,100,200], random_seed=1234, verbose=False)
                (idx_train_i, idx_test_i) = sampling.manual(sampling_dictionary_number, sampling_type='percentage', test_selection_option=2)
                self.assertTrue((idx_train_i == idx_train).all())
        except Exception:
            self.assertTrue(False)

    def test_random_seed_random(self):

        idx = np.zeros((1000,))
        idx[500:800] = 1

        idx = idx.astype(int)

        try:
            sampling = DataSampler(idx, idx_test=[], random_seed=123, verbose=False)
            (idx_train, idx_test) = sampling.random(40, test_selection_option=1)

            for i in range(0,10):
                sampling = DataSampler(idx, idx_test=[], random_seed=123, verbose=False)
                (idx_train_i, idx_test_i) = sampling.random(40, test_selection_option=1)
                self.assertTrue((idx_train_i == idx_train).all())
        except Exception:
            self.assertTrue(False)

        try:
            sampling = DataSampler(idx, idx_test=[], random_seed=1234, verbose=False)
            (idx_train, idx_test) = sampling.random(40, test_selection_option=1)

            for i in range(0,10):
                sampling = DataSampler(idx, idx_test=[], random_seed=1234, verbose=False)
                (idx_train_i, idx_test_i) = sampling.random(40, test_selection_option=1)
                self.assertTrue((idx_train_i == idx_train).all())
        except Exception:
            self.assertTrue(False)

        try:
            sampling = DataSampler(idx, idx_test=[], random_seed=123, verbose=False)
            (idx_train, idx_test) = sampling.random(40, test_selection_option=1)

            for i in range(0,10):
                sampling = DataSampler(idx, idx_test=[], random_seed=123, verbose=False)
                (idx_train_i, idx_test_i) = sampling.random(40, test_selection_option=1)
                self.assertTrue((idx_train_i == idx_train).all())
        except Exception:
            self.assertTrue(False)

        try:
            sampling = DataSampler(idx, idx_test=[], random_seed=1234, verbose=False)
            (idx_train, idx_test) = sampling.random(40, test_selection_option=2)

            for i in range(0,10):
                sampling = DataSampler(idx, idx_test=[], random_seed=1234, verbose=False)
                (idx_train_i, idx_test_i) = sampling.random(40, test_selection_option=2)
                self.assertTrue((idx_train_i == idx_train).all())
        except Exception:
            self.assertTrue(False)

            sampling = DataSampler(idx, idx_test=[1,100,200], random_seed=123, verbose=False)
            (idx_train, idx_test) = sampling.random(40, test_selection_option=1)

            for i in range(0,10):
                sampling = DataSampler(idx, idx_test=[1,100,200], random_seed=123, verbose=False)
                (idx_train_i, idx_test_i) = sampling.random(40, test_selection_option=1)
                self.assertTrue((idx_train_i == idx_train).all())
        except Exception:
            self.assertTrue(False)

        try:
            sampling = DataSampler(idx, idx_test=[1,100,200], random_seed=1234, verbose=False)
            (idx_train, idx_test) = sampling.random(40, test_selection_option=2)

            for i in range(0,10):
                sampling = DataSampler(idx, idx_test=[1,100,200], random_seed=1234, verbose=False)
                (idx_train_i, idx_test_i) = sampling.random(40, test_selection_option=2)
                self.assertTrue((idx_train_i == idx_train).all())
        except Exception:
            self.assertTrue(False)
