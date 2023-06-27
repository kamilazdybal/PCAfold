import unittest
import numpy as np
from PCAfold import preprocess

class Preprocess(unittest.TestCase):

    def test_preprocess__invert_zero_pivot_transform__allowed_calls(self):

        X = np.array([[1,-10],
                      [2,0],
                      [3,-30],
                      [2,0],
                      [1,-10]])

        X_transformed, maximum_positive, minimum_negative = preprocess.zero_pivot_transform(X)

        try:
            X_back = preprocess.invert_zero_pivot_transform(X_transformed, maximum_positive, minimum_negative)
        except Exception:
            self.assertTrue(False)

        try:
            X_back = preprocess.invert_zero_pivot_transform(X_transformed, np.array([1,10]), minimum_negative)
            X_back = preprocess.invert_zero_pivot_transform(X_transformed, maximum_positive, np.array([1,10]))
        except Exception:
            self.assertTrue(False)

# ------------------------------------------------------------------------------

    def test_preprocess__invert_zero_pivot_transform__not_allowed_calls(self):

        X = np.array([[1,-10],
                      [2,0],
                      [3,-30],
                      [2,0],
                      [1,-10]])

        X_transformed, maximum_positive, minimum_negative = preprocess.zero_pivot_transform(X)

        with self.assertRaises(ValueError):
            X_back = preprocess.invert_zero_pivot_transform([], maximum_positive, minimum_negative)

        with self.assertRaises(ValueError):
            X_back = preprocess.invert_zero_pivot_transform(X, [], minimum_negative)

        with self.assertRaises(ValueError):
            X_back = preprocess.invert_zero_pivot_transform(X, maximum_positive, [])

        with self.assertRaises(ValueError):
            X_back = preprocess.invert_zero_pivot_transform(X, np.array([1,2,3]), minimum_negative)

        with self.assertRaises(ValueError):
            X_back = preprocess.invert_zero_pivot_transform(X, maximum_positive, np.array([1,2,3]))

# ------------------------------------------------------------------------------

    def test_preprocess__invert_zero_pivot_transform__computation(self):

        X = np.array([[1,10],
                      [-2,0],
                      [3,-30],
                      [0,0],
                      [1,-10],
                      [-1,5]])

        X_transformed, maximum_positive, minimum_negative = preprocess.zero_pivot_transform(X)
        X_back = preprocess.invert_zero_pivot_transform(X_transformed, maximum_positive, minimum_negative)
        self.assertTrue(np.array_equal(X, X_back))

        # Variables that contain negative and positive values and do not contain zeros:

        X = np.array([[1,10],
                      [-2,4.5],
                      [3,-30],
                      [-0.5,-1],
                      [1,-10],
                      [-1,5]])

        X_transformed, maximum_positive, minimum_negative = preprocess.zero_pivot_transform(X)
        X_back = preprocess.invert_zero_pivot_transform(X_transformed, maximum_positive, minimum_negative)
        self.assertTrue(np.array_equal(X, X_back))

        # Variables that only contain non-negative or non-positive values:

        X = np.array([[1,-20],
                      [2,0],
                      [3,-30],
                      [0,0],
                      [1,-10]])

        X_transformed, maximum_positive, minimum_negative = preprocess.zero_pivot_transform(X)
        X_back = preprocess.invert_zero_pivot_transform(X_transformed, maximum_positive, minimum_negative)
        self.assertTrue(np.array_equal(X, X_back))
