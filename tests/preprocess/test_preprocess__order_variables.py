import unittest
import numpy as np
from PCAfold import preprocess
from PCAfold import reduction
from PCAfold import analysis

class Preprocess(unittest.TestCase):

    def test_preprocess__order_variables__allowed_calls(self):

        X = np.array([[100, 1, 10, 0.1],
                      [200, 2, 20, 0.2],
                      [300, 3, 30, 0.3]])

        X_descending = np.array([[100, 10, 1, 0.1],
                                 [200, 20, 2, 0.2],
                                 [300, 30, 3, 0.3]])

        idx_descending = [0, 2, 1, 3]

        X_ascending = np.array([[0.1, 1, 10, 100],
                                [0.2, 2, 20, 200],
                                [0.3, 3, 30, 300]])

        idx_ascending = [3, 1, 2, 0]

        try:
            (X_ordered, idx) = preprocess.order_variables(X, method='mean', descending=True)
            self.assertTrue(np.array_equal(X_ordered, X_descending))
            self.assertTrue(np.array_equal(idx, idx_descending))
        except Exception:
            self.assertTrue(False)

        try:
            (X_ordered, idx) = preprocess.order_variables(X, method='mean', descending=False)
            self.assertTrue(np.array_equal(X_ordered, X_ascending))
            self.assertTrue(np.array_equal(idx, idx_ascending))
        except Exception:
            self.assertTrue(False)

        try:
            (X_ordered, idx) = preprocess.order_variables(X, method='min', descending=False)
            (X_ordered, idx) = preprocess.order_variables(X, method='max', descending=False)
            (X_ordered, idx) = preprocess.order_variables(X, method='std', descending=False)
            (X_ordered, idx) = preprocess.order_variables(X, method='var', descending=False)
        except Exception:
            self.assertTrue(False)

        X = np.ones((10,1))

        try:
            (X_ordered, idx) = preprocess.order_variables(X, method='mean', descending=False)
            (X_ordered, idx) = preprocess.order_variables(X, method='min', descending=False)
            (X_ordered, idx) = preprocess.order_variables(X, method='max', descending=False)
            (X_ordered, idx) = preprocess.order_variables(X, method='std', descending=False)
            (X_ordered, idx) = preprocess.order_variables(X, method='var', descending=False)
        except Exception:
            self.assertTrue(False)

# ------------------------------------------------------------------------------

    def test_preprocess__order_variables__not_allowed_calls(self):

        X = np.array([[100, 1, 10, 0.1],
                      [200, 2, 20, 0.2],
                      [300, 3, 30, 0.3]])

        with self.assertRaises(ValueError):
            (X_ordered, idx) = preprocess.order_variables(X, method='mean', descending=1)

        with self.assertRaises(ValueError):
            (X_ordered, idx) = preprocess.order_variables(X, method='none', descending=True)

        with self.assertRaises(ValueError):
            (X_ordered, idx) = preprocess.order_variables(X, method=1, descending=True)

        with self.assertRaises(ValueError):
            (X_ordered, idx) = preprocess.order_variables([1,2,3], method='mean', descending=True)

        X = np.array([1,10,100])

        with self.assertRaises(ValueError):
            (X_ordered, idx) = preprocess.order_variables(X, method='mean', descending=True)

# ------------------------------------------------------------------------------
