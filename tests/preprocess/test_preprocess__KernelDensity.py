import unittest
import numpy as np
from PCAfold import preprocess
from PCAfold import reduction
from PCAfold import analysis

class Preprocess(unittest.TestCase):

    def test_preprocess__KernelDensity__allowed_calls(self):

        X = np.random.rand(100,20)

        try:
            kerneld = preprocess.KernelDensity(X, X[:,1])
        except Exception:
            self.assertTrue(False)

        try:
            kerneld = preprocess.KernelDensity(X, X[:,4:9])
        except Exception:
            self.assertTrue(False)

        try:
            kerneld = preprocess.KernelDensity(X, X[:,0])
        except Exception:
            self.assertTrue(False)

        try:
            kerneld = preprocess.KernelDensity(X, X)
        except Exception:
            self.assertTrue(False)

        try:
            kerneld.X_weighted
            kerneld.weights
        except Exception:
            self.assertTrue(False)

# ------------------------------------------------------------------------------

    def test_preprocess__KernelDensity__not_allowed_calls(self):

        X = np.random.rand(100,20)
        kerneld = preprocess.KernelDensity(X, X[:,1])

        with self.assertRaises(AttributeError):
            kerneld.X_weighted = 1

        with self.assertRaises(AttributeError):
            kerneld.weights = 1

        with self.assertRaises(ValueError):
            kerneld = preprocess.KernelDensity(X, X[20:30,1])

        with self.assertRaises(ValueError):
            kerneld = preprocess.KernelDensity(X, X[20:30,:])

# ------------------------------------------------------------------------------
