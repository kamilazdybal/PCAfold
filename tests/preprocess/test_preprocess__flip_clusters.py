import unittest
import numpy as np
from PCAfold import preprocess
from PCAfold import reduction
from PCAfold import analysis

class Preprocess(unittest.TestCase):

    def test_preprocess__flip_clusters__allowed_calls(self):

        pass

# ------------------------------------------------------------------------------

    def test_preprocess__flip_clusters__not_allowed_calls(self):

        idx_unflipped = np.array([0,0,0,1,1,1,2,2,2])
        with self.assertRaises(ValueError):
            idx = preprocess.flip_clusters(idx_unflipped, dictionary={3:2,2:3})

        with self.assertRaises(ValueError):
            idx = preprocess.flip_clusters(idx_unflipped, dictionary={0:1,1:1.5})

# ------------------------------------------------------------------------------

    def test_preprocess__flip_clusters__computation(self):

        try:
            idx_unflipped = np.array([0,0,0,1,1,1,2,2,2])
            idx_flipped = np.array([0,0,0,2,2,2,1,1,1])
            idx = preprocess.flip_clusters(idx_unflipped, dictionary={1:2, 2:1})
            comparison = idx_flipped == idx
            self.assertTrue(comparison.all())
        except:
            self.assertTrue(False)

        try:
            idx_unflipped = np.array([0,0,0,1,1,1,2,2,2])
            idx_flipped = np.array([0,0,0,10,10,10,20,20,20])
            idx = preprocess.flip_clusters(idx_unflipped, dictionary={1:10, 2:20})
            comparison = idx_flipped == idx
            self.assertTrue(comparison.all())
        except:
            self.assertTrue(False)

# ------------------------------------------------------------------------------
