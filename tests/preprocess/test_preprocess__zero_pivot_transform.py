import unittest
import numpy as np
from PCAfold import preprocess

class Preprocess(unittest.TestCase):

    def test_preprocess__zero_pivot_transform__allowed_calls(self):

        X = np.array([[1,-10],
                      [2,0],
                      [3,-30],
                      [2,0],
                      [1,-10]])

        try:
            X_transformed, maximum_positive, minimum_negative = preprocess.zero_pivot_transform(X)
        except Exception:
            self.assertTrue(False)

# ------------------------------------------------------------------------------

    def test_preprocess__zero_pivot_transform__not_allowed_calls(self):

        X = np.array([[1,-10],
                      [2,0],
                      [3,-30],
                      [2,0],
                      [1,-10]])

        with self.assertRaises(ValueError):
            X_transformed, maximum_positive, minimum_negative = preprocess.zero_pivot_transform([])

        with self.assertRaises(ValueError):
            X_transformed, maximum_positive, minimum_negative = preprocess.zero_pivot_transform(X, method='method')

        with self.assertRaises(ValueError):
            X_transformed, maximum_positive, minimum_negative = preprocess.zero_pivot_transform(X, verbose=[])

# ------------------------------------------------------------------------------

    def test_preprocess__zero_pivot_transform__computation(self):

        # Variables that contain negative and positive values:

        X = np.array([[1,10],
                      [-2,0],
                      [3,-30],
                      [0,0],
                      [1,-10],
                      [-1,5]])

        X_transformed, maximum_positive, minimum_negative = preprocess.zero_pivot_transform(X)

        self.assertTrue(maximum_positive[0]==3)
        self.assertTrue(maximum_positive[1]==10)
        self.assertTrue(minimum_negative[0]==-2)
        self.assertTrue(minimum_negative[1]==-30)

        X_transformed_manually = np.array([[1/3,10/10],
                                            [-2/2,0],
                                            [3/3,-30/30],
                                            [0,0],
                                            [1/3,-10/30],
                                            [-1/2,5/10]])

        self.assertTrue(np.array_equal(X_transformed[:,0], X_transformed_manually[:,0]))
        self.assertTrue(np.array_equal(X_transformed[:,1], X_transformed_manually[:,1]))

        # Variables that contain negative and positive values and do not contain zeros:

        X = np.array([[1,10],
                      [-2,4.5],
                      [3,-30],
                      [-0.5,-1],
                      [1,-10],
                      [-1,5]])

        X_transformed, maximum_positive, minimum_negative = preprocess.zero_pivot_transform(X)

        self.assertTrue(maximum_positive[0]==3)
        self.assertTrue(maximum_positive[1]==10)
        self.assertTrue(minimum_negative[0]==-2)
        self.assertTrue(minimum_negative[1]==-30)

        X_transformed_manually = np.array([[1/3,10/10],
                                            [-2/2,4.5/10],
                                            [3/3,-30/30],
                                            [-0.5/2,-1/30],
                                            [1/3,-10/30],
                                            [-1/2,5/10]])

        self.assertTrue(np.array_equal(X_transformed[:,0], X_transformed_manually[:,0]))
        self.assertTrue(np.array_equal(X_transformed[:,1], X_transformed_manually[:,1]))

        # Variables that only contain non-negative or non-positive values:

        X = np.array([[1,-20],
                      [2,0],
                      [3,-30],
                      [0,0],
                      [1,-10]])

        X_transformed, maximum_positive, minimum_negative = preprocess.zero_pivot_transform(X)

        self.assertTrue(maximum_positive[0]==3)
        self.assertTrue(maximum_positive[1]==0)
        self.assertTrue(minimum_negative[0]==0)
        self.assertTrue(minimum_negative[1]==-30)

        self.assertTrue(np.array_equal(X_transformed[:,0], X[:,0]/3.0))
        self.assertTrue(np.array_equal(X_transformed[:,1], X[:,1]/30.0))
