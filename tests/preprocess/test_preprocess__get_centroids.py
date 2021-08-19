import unittest
import numpy as np
from PCAfold import preprocess
from PCAfold import reduction
from PCAfold import analysis

class Preprocess(unittest.TestCase):

    def test_preprocess__get_centroids__allowed_calls(self):

        pass

# ------------------------------------------------------------------------------

    def test_preprocess__get_centroidss__not_allowed_calls(self):

        X = np.random.rand(100,10)
        idx = np.zeros((90,))

        with self.assertRaises(ValueError):
            centroids = preprocess.get_centroids(X, idx)

        X = np.random.rand(100,10)
        idx = np.zeros((110,))

        with self.assertRaises(ValueError):
            centroids = preprocess.get_centroids(X, idx)

# ------------------------------------------------------------------------------

    def test_preprocess__get_centroids__allowed_calls(self):

        try:
            x = np.array([[1,2,10],[1,2,10],[1,2,10]])
            idx = np.array([0,0,0])
            idx_centroids = np.array([[1, 2, 10]])
            centroids = preprocess.get_centroids(x, idx)
            comparison = (idx_centroids == centroids)
            self.assertTrue(comparison.all())
        except Exception:
            self.assertTrue(False)

        try:
            x = np.array([[1,2,10],[1,2,10],[20,30,40]])
            idx = np.array([0,0,1])
            idx_centroids = np.array([[1, 2, 10], [20,30,40]])
            centroids = preprocess.get_centroids(x, idx)
            comparison = (idx_centroids == centroids)
            self.assertTrue(comparison.all())
        except Exception:
            self.assertTrue(False)

# ------------------------------------------------------------------------------
