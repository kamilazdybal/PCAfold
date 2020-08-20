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

    def test_degrade_clusters_allowed_calls(self):

        try:
            idx_undegraded = [1, 1, 2, 2, 3, 3]
            idx_degraded = [0, 0, 1, 1, 2, 2]
            (idx, k) = preprocess.degrade_clusters(idx_undegraded, verbose=False)
            self.assertTrue(np.min(idx) == 0)
            self.assertTrue(k == 3)
            self.assertTrue(list(idx) == idx_degraded)
        except:
            self.assertTrue(False)

        try:
            idx_undegraded = [-1, -1, 1, 1, 2, 2, 3, 3]
            idx_degraded = [0, 0, 1, 1, 2, 2, 3, 3]
            (idx, k) = preprocess.degrade_clusters(idx_undegraded, verbose=False)
            self.assertTrue(np.min(idx) == 0)
            self.assertTrue(k == 4)
            self.assertTrue(list(idx) == idx_degraded)
        except:
            self.assertTrue(False)

        try:
            idx_undegraded = [-1, 1, 3, -1, 1, 1, 2, 2, 3, 3]
            idx_degraded = [0, 1, 3, 0, 1, 1, 2, 2, 3, 3]
            (idx, k) = preprocess.degrade_clusters(idx_undegraded, verbose=False)
            self.assertTrue(np.min(idx) == 0)
            self.assertTrue(k == 4)
            self.assertTrue(list(idx) == idx_degraded)
        except:
            self.assertTrue(False)

        try:
            idx = np.array([-1,-1,0,0,0,0,1,1,1,1,5])
            (idx, k) = preprocess.degrade_clusters(idx, verbose=False)
            self.assertTrue(np.min(idx) == 0)
            self.assertTrue(k == 4)
        except:
            self.assertTrue(False)

    def test_degrade_clusters_not_allowed_calls(self):

        idx_test = [0,0,0,1,1,1,True,2,2,2]
        with self.assertRaises(ValueError):
            (idx, k) = preprocess.degrade_clusters(idx_test, verbose=False)

        idx_test = [0,0,0,1,1,1,5.1,2,2,2]
        with self.assertRaises(ValueError):
            (idx, k) = preprocess.degrade_clusters(idx_test, verbose=False)

        idx_test = np.array([0,0,0,1.1,1,1,2,2,2])
        with self.assertRaises(ValueError):
            (idx, k) = preprocess.degrade_clusters(idx_test, verbose=False)

        idx_test = np.array([-1.2,0,0,0,1,1,1,2,2,2])
        with self.assertRaises(ValueError):
            (idx, k) = preprocess.degrade_clusters(idx_test, verbose=False)

        with self.assertRaises(ValueError):
            (idx, k) = preprocess.degrade_clusters(1, verbose=False)

        with self.assertRaises(ValueError):
            (idx, k) = preprocess.degrade_clusters('list', verbose=False)

    def test_flip_clusters_allowed_calls(self):

        try:
            idx_unflipped = np.array([0,0,0,1,1,1,2,2,2])
            idx_flipped = np.array([0,0,0,2,2,2,1,1,1])
            idx = preprocess.flip_clusters(idx_unflipped, dictionary={1:2, 2:1})
            self.assertTrue(idx_flipped.all() == idx.all())
        except:
            self.assertTrue(False)

        try:
            idx_unflipped = np.array([0,0,0,1,1,1,2,2,2])
            idx_flipped = np.array([0,0,0,10,10,10,20,20,20])
            idx = preprocess.flip_clusters(idx_unflipped, dictionary={1:10, 2:20})
            self.assertTrue(idx_flipped.all() == idx.all())
        except:
            self.assertTrue(False)

    def test_flip_clusters_not_allowed_calls(self):

        idx_unflipped = np.array([0,0,0,1,1,1,2,2,2])
        with self.assertRaises(ValueError):
            idx = preprocess.flip_clusters(idx_unflipped, dictionary={3:2,2:3})

        with self.assertRaises(ValueError):
            idx = preprocess.flip_clusters(idx_unflipped, dictionary={0:1,1:1.5})
