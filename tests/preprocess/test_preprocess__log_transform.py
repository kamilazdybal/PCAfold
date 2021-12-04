import unittest
import numpy as np
from PCAfold import preprocess
from PCAfold import reduction
from PCAfold import analysis

class Preprocess(unittest.TestCase):

    def test_preprocess__log_transform__allowed_calls(self):

        X = np.random.rand(100,20) + 1

        try:
            X_log = preprocess.log_transform(X)
            X_symlog = preprocess.log_transform(X, method='ln')
            X_symlog = preprocess.log_transform(X, method='symlog', threshold=1)
            X_symlog = preprocess.log_transform(X, method='symlog', threshold=1.e-2)
            X_symlog = preprocess.log_transform(X, method='symlog', threshold=1.e-4)
        except Exception:
            self.assertTrue(False)

# ------------------------------------------------------------------------------

    def test_preprocess__log_transform__not_allowed_calls(self):

        X = np.random.rand(100,20) + 1

        with self.assertRaises(ValueError):
            X_log = preprocess.log_transform(X, method='l')
            X_log = preprocess.log_transform(X, threshold=[])
            X_log = preprocess.log_transform([1,2,3])
# ------------------------------------------------------------------------------

    def test_preprocess__log_transform__computation(self):

        pass
