import unittest
import numpy as np
from PCAfold import preprocess
from PCAfold import reduction
from PCAfold import analysis

class Preprocess(unittest.TestCase):

    def test_preprocess__get_partition__allowed_calls(self):

        pass

# ------------------------------------------------------------------------------

    def test_preprocess__get_parition__not_allowed_calls(self):

        X = np.random.rand(100,10)
        idx = np.zeros((90,))

        with self.assertRaises(ValueError):
            (x_in_clusters, idx_in_clusters) = preprocess.get_partition(X, idx)

        X = np.random.rand(100,10)
        idx = np.zeros((110,))

        with self.assertRaises(ValueError):
            (x_in_clusters, idx_in_clusters) = preprocess.get_partition(X, idx)

# ------------------------------------------------------------------------------

    def test_preprocess__get_partition__computation(self):

        try:
            x = np.array([[1,2,10],[1,2,10],[1,2,10]])
            idx = np.array([0,0,0])
            pre_x_in_clusters = [np.array([[1,2,10],[1,2,10],[1,2,10]])]
            pre_idx_in_clusters = [np.array([0,1,2])]
            (x_in_clusters, idx_in_clusters) = preprocess.get_partition(x, idx)
            comparison_1 = (pre_x_in_clusters[0] == x_in_clusters[0])
            self.assertTrue(comparison_1.all())
            comparison_2 = (pre_idx_in_clusters[0] == idx_in_clusters[0])
            self.assertTrue(comparison_2.all())
        except Exception:
            self.assertTrue(False)

        try:
            x = np.array([[1,2,10],[1,2,10],[30,40,50]])
            idx = np.array([0,0,1])
            pre_x_in_clusters = [np.array([[1,2,10],[1,2,10]]), np.array([[30,40,50]])]
            pre_idx_in_clusters = [np.array([0,1]), np.array([2])]
            (x_in_clusters, idx_in_clusters) = preprocess.get_partition(x, idx)
            comparison_1 = (pre_x_in_clusters[0] == x_in_clusters[0])
            comparison_2 = (pre_x_in_clusters[1] == x_in_clusters[1])
            self.assertTrue(comparison_1.all())
            self.assertTrue(comparison_2.all())
            comparison_3 = (pre_idx_in_clusters[0] == idx_in_clusters[0])
            comparison_4 = (pre_idx_in_clusters[1] == idx_in_clusters[1])
            self.assertTrue(comparison_3.all())
            self.assertTrue(comparison_4.all())
        except Exception:
            self.assertTrue(False)

# ------------------------------------------------------------------------------
