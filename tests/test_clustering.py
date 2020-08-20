import unittest
import numpy as np
from PCAfold import preprocess


class TestClustering(unittest.TestCase):
    def test_variable_bins(self):
        try:
            idx = preprocess.variable_bins(np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]), 4, verbose=False)
            self.assertTrue(True)
        except:
            self.assertTrue(False)

        self.assertTrue(isinstance(idx, np.ndarray))
        self.assertTrue(idx.ndim == 1)

    def test_predefined_variable_bins(self):
        try:
            idx = preprocess.predefined_variable_bins(np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]), [3.5, 8.5],
                                                      verbose=False)
            self.assertTrue(True)
        except:
            self.assertTrue(False)

        self.assertTrue(isinstance(idx, np.ndarray))
        self.assertTrue(idx.ndim == 1)

    def test_mixture_fraction_bins(self):
        try:
            idx = preprocess.mixture_fraction_bins(np.array([0.1, 0.15, 0.2, 0.25, 0.6, 0.8, 1]), 2, 0.2)
            self.assertTrue(True)
        except:
            self.assertTrue(False)

        self.assertTrue(isinstance(idx, np.ndarray))
        self.assertTrue(idx.ndim == 1)

    def test_source_bins(self):
        try:
            idx = preprocess.source_bins(np.array([-100, -20, -0.1, 0, 0.1, 1, 10, 20, 200, 300, 400]), k=4,
                                         split_at_zero=True, verbose=False)

            self.assertTrue(True)
        except:
            self.assertTrue(False)

        self.assertTrue(isinstance(idx, np.ndarray))
        self.assertTrue(idx.ndim == 1)

    def test_degrade_clusters(self):

        (idx, k) = preprocess.degrade_clusters([1, 1, 2, 2, 3, 3], verbose=False)

        self.assertTrue(np.min(idx) == 0)
        self.assertTrue(k == 3)
