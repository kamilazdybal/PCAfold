import unittest
import numpy as np
from PCAfold import preprocess
from PCAfold import reduction
from PCAfold import analysis

class Preprocess(unittest.TestCase):

    def test_preprocess__remove_constant_vars__allowed_calls(self):

        X = np.random.rand(100,20)

        try:
            (X_removed, idx_removed, idx_retained) = preprocess.remove_constant_vars(X)
        except Exception:
            self.assertTrue(False)

        X = np.random.rand(100,1)

        try:
            (X_removed, idx_removed, idx_retained) = preprocess.remove_constant_vars(X)
        except Exception:
            self.assertTrue(False)

# ------------------------------------------------------------------------------

    def test_preprocess__remove_constant_vars__not_allowed_calls(self):

        X = np.random.rand(100,)

        with self.assertRaises(ValueError):
            (X_removed, idx_removed, idx_retained) = preprocess.remove_constant_vars(X)

        with self.assertRaises(ValueError):
            (X_removed, idx_removed, idx_retained) = preprocess.remove_constant_vars([1,2,3,4,5])

        X = np.random.rand(100,20)

        with self.assertRaises(ValueError):
            (X_removed, idx_removed, idx_retained) = preprocess.remove_constant_vars(X, maxtol='none')

        with self.assertRaises(ValueError):
            (X_removed, idx_removed, idx_retained) = preprocess.remove_constant_vars(X, rangetol='none')

# ------------------------------------------------------------------------------

    def test_preprocess__remove_constant_vars__computation(self):

        test_data_set = np.random.rand(100,20)

        try:
            # Inject two constant columns:
            test_data_set_constant = np.hstack((test_data_set[:,0:3], 2.4*np.ones((100,1)), test_data_set[:,3:15], -8.1*np.ones((100,1)), test_data_set[:,15::]))
            idx_removed_check = [3,16]
            idx_retained_check = [0,1,2,4,5,6,7,8,9,10,11,12,13,14,15,17,18,19,20,21]
            (X_removed, idx_removed, idx_retained) = preprocess.remove_constant_vars(test_data_set_constant)
            comparison = X_removed == test_data_set
            self.assertTrue(comparison.all())
            self.assertTrue(idx_removed == idx_removed_check)
            self.assertTrue(idx_retained == idx_retained_check)
        except Exception:
            self.assertTrue(False)

        try:
            # Inject a constant column that has values close to zero:
            close_to_zero_column = -10**(-14)*np.ones((100,1))
            close_to_zero_column[20:30,:] = -10**(-13)
            close_to_zero_column[80:85,:] = -10**(-14)
            test_data_set_constant = np.hstack((test_data_set[:,0:3], close_to_zero_column, test_data_set[:,3::]))
            idx_removed_check = [3]
            idx_retained_check = [0,1,2,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
            (X_removed, idx_removed, idx_retained) = preprocess.remove_constant_vars(test_data_set_constant)
            comparison = X_removed == test_data_set
            self.assertTrue(comparison.all())
            self.assertTrue(idx_removed == idx_removed_check)
            self.assertTrue(idx_retained == idx_retained_check)
        except Exception:
            self.assertTrue(False)

        try:
            # Inject a constant column that has values close to zero:
            close_to_zero_column = -10**(-14)*np.ones((100,1))
            close_to_zero_column[20:30,:] = 10**(-13)
            close_to_zero_column[80:85,:] = 10**(-14)
            test_data_set_constant = np.hstack((test_data_set[:,0:3], close_to_zero_column, test_data_set[:,3::]))
            idx_removed_check = [3]
            idx_retained_check = [0,1,2,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
            (X_removed, idx_removed, idx_retained) = preprocess.remove_constant_vars(test_data_set_constant)
            comparison = X_removed == test_data_set
            self.assertTrue(comparison.all())
            self.assertTrue(idx_removed == idx_removed_check)
            self.assertTrue(idx_retained == idx_retained_check)
        except Exception:
            self.assertTrue(False)

# ------------------------------------------------------------------------------
