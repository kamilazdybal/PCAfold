import unittest
import numpy as np
from PCAfold import preprocess
from PCAfold import reduction
from PCAfold import analysis

class Preprocess(unittest.TestCase):

    def test_preprocess__degrade_clusters__allowed_calls(self):

        pass

# ------------------------------------------------------------------------------

    def test_preprocess__degrade_clusters__not_allowed_calls(self):

        idx_test = np.array([0,0,0,1,1,1,-0.1,2,2,2])
        with self.assertRaises(ValueError):
            (idx, k) = preprocess.degrade_clusters(idx_test, verbose=False)

        idx_test = np.array([0,0,0,1,1,1,5.1,2,2,2])
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

# ------------------------------------------------------------------------------

    def test_preprocess__degrade_clusters__computation(self):

        try:
            idx_undegraded = np.array([1, 1, 2, 2, 3, 3])
            idx_degraded = np.array([0, 0, 1, 1, 2, 2])
            (idx, k) = preprocess.degrade_clusters(idx_undegraded, verbose=False)
            self.assertTrue(np.min(idx) == 0)
            self.assertTrue(k == 3)
            self.assertTrue(np.array_equal(idx, idx_degraded))
        except:
            self.assertTrue(False)

        try:
            idx_undegraded = np.array([-1, -1, 1, 1, 2, 2, 3, 3])
            idx_degraded = np.array([0, 0, 1, 1, 2, 2, 3, 3])
            (idx, k) = preprocess.degrade_clusters(idx_undegraded, verbose=False)
            self.assertTrue(np.min(idx) == 0)
            self.assertTrue(k == 4)
            self.assertTrue(np.array_equal(idx, idx_degraded))
        except:
            self.assertTrue(False)

        try:
            idx_undegraded = np.array([-1, 1, 3, -1, 1, 1, 2, 2, 3, 3])
            idx_degraded = np.array([0, 1, 3, 0, 1, 1, 2, 2, 3, 3])
            (idx, k) = preprocess.degrade_clusters(idx_undegraded, verbose=False)
            self.assertTrue(np.min(idx) == 0)
            self.assertTrue(k == 4)
            self.assertTrue(np.array_equal(idx, idx_degraded))
        except:
            self.assertTrue(False)

        try:
            idx = np.array([-1,-1,0,0,0,0,1,1,1,1,5])
            (idx, k) = preprocess.degrade_clusters(idx, verbose=False)
            self.assertTrue(np.min(idx) == 0)
            self.assertTrue(k == 4)
        except:
            self.assertTrue(False)

# ------------------------------------------------------------------------------
