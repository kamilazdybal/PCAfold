import unittest
import numpy as np
from PCAfold import preprocess
from PCAfold import reduction
from PCAfold import analysis

class Preprocess(unittest.TestCase):

    def test_preprocess__PreProcessing(self):

        test_data_set = np.random.rand(100,20)

        # Inject two constant columns:
        test_data_set_constant = np.hstack((test_data_set[:,0:3], 2.4*np.ones((100,1)), test_data_set[:,3:15], -8.1*np.ones((100,1)), test_data_set[:,15::]))
        idx_removed_check = [3,16]
        idx_retained_check = [0,1,2,4,5,6,7,8,9,10,11,12,13,14,15,17,18,19,20,21]

        try:
            preprocessed = preprocess.PreProcessing(test_data_set_constant, scaling='none', nocenter=True)
            comparison = preprocessed.X_removed == test_data_set
            self.assertTrue(comparison.all())
            self.assertTrue(preprocessed.idx_removed == idx_removed_check)
            self.assertTrue(preprocessed.idx_retained == idx_retained_check)
            self.assertTrue(np.shape(preprocessed.X_cs) == (100,20))
        except Exception:
            self.assertTrue(False)

# ------------------------------------------------------------------------------

    def test_preprocess__PreProcessing__not_allowed_attribute_setting(self):

        test_data_set = np.random.rand(100,20)
        pp = preprocess.PreProcessing(test_data_set, scaling='auto')

        with self.assertRaises(AttributeError):
            pp.X_cs = 1
        with self.assertRaises(AttributeError):
            pp.X_center = 1
        with self.assertRaises(AttributeError):
            pp.X_scale = 1
        with self.assertRaises(AttributeError):
            pp.X_removed = 1
        with self.assertRaises(AttributeError):
            pp.idx_removed = 1
        with self.assertRaises(AttributeError):
            pp.idx_retained = 1

# ------------------------------------------------------------------------------
