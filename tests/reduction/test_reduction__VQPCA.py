import unittest
import numpy as np
from PCAfold import preprocess
from PCAfold import reduction
from PCAfold import analysis

class Reduction(unittest.TestCase):

    def test_reduction__VQPCA__allowed_calls(self):

        X = np.random.rand(400,10)

        try:
            vqpca = reduction.VQPCA(X, 3, 2, random_state=100)
            vqpca = reduction.VQPCA(X, 3, 2, init='uniform', random_state=100)
            vqpca = reduction.VQPCA(X, 3, 2, init='uniform', max_iter=20, random_state=100)
        except Exception:
            self.assertTrue(False)

        try:
            idx0 = np.ones((400,1))
            idx0[20:40,:] = 2
            idx0[200:300,:] = 3
            idx0 = idx0.astype(int)
            vqpca = reduction.VQPCA(X, 3, 2, idx0=idx0)

            idx0 = np.ones((400,))
            idx0[20:40] = 2
            idx0[200:300] = 3
            idx0 = idx0.astype(int)
            vqpca = reduction.VQPCA(X, 3, 2, idx0=idx0)
        except Exception:
            self.assertTrue(False)

# ------------------------------------------------------------------------------

    def test_reduction__VQPCA__allowed_attribute_call(self):

        X = np.random.rand(400,10)

        try:
            vqpca = reduction.VQPCA(X, 3, 2, random_state=100)
            vqpca.idx
            vqpca.collected_idx
            vqpca.converged
            vqpca.A
            vqpca.principal_components
            vqpca.reconstruction_errors_in_clusters
        except Exception:
            self.assertTrue(False)

# ------------------------------------------------------------------------------

    def test_reduction__VQPCA__not_allowed_attribute_setting(self):

        X = np.random.rand(400,10)
        vqpca = reduction.VQPCA(X, 3, 2, random_state=100)

        with self.assertRaises(AttributeError):
            vqpca.idx = 1
        with self.assertRaises(AttributeError):
            vqpca.collected_idx = 1
        with self.assertRaises(AttributeError):
            vqpca.converged = 1
        with self.assertRaises(AttributeError):
            vqpca.A = 1
        with self.assertRaises(AttributeError):
            vqpca.principal_components = 1
        with self.assertRaises(AttributeError):
            vqpca.reconstruction_errors_in_clusters = 1

# ------------------------------------------------------------------------------

    def test_reduction__VQPCA__not_allowed_calls(self):

        X = np.random.rand(400,10)

        with self.assertRaises(ValueError):
            vqpca = reduction.VQPCA(X, 0, 2)

        with self.assertRaises(ValueError):
            vqpca = reduction.VQPCA(X, 2, 0)

        with self.assertRaises(ValueError):
            vqpca = reduction.VQPCA(X, -1, 2)

        with self.assertRaises(ValueError):
            vqpca = reduction.VQPCA(X, 2, -1)

        with self.assertRaises(ValueError):
            vqpca = reduction.VQPCA(X, 2, 2, scaling='hello')

        with self.assertRaises(ValueError):
            vqpca = reduction.VQPCA(X, 2, 2, init='hello')

        with self.assertRaises(ValueError):
            vqpca = reduction.VQPCA(X, 2, 2, max_iter=0)

        with self.assertRaises(ValueError):
            vqpca = reduction.VQPCA(X, 2, 2, max_iter=-1)

        with self.assertRaises(ValueError):
            vqpca = reduction.VQPCA(X, 2, 2, random_state='hello')

# ------------------------------------------------------------------------------

    def test_reduction__VQPCA__computation(self):

        X = np.random.rand(400,10)

        # Test that random seed is working properly:
        try:
            vqpca_1 = reduction.VQPCA(X, 3, 2, init='random', random_state=100)
            idx_1 = vqpca_1.idx
            vqpca_2 = reduction.VQPCA(X, 3, 2, init='random', random_state=100)
            idx_2 = vqpca_2.idx
            self.assertTrue(np.array_equal(idx_1, idx_2))
        except Exception:
            self.assertTrue(False)

        # Test that random seed is working properly:
        try:
            vqpca_1 = reduction.VQPCA(X, 3, 2, init='random', random_state=100)
            idx_1 = vqpca_1.idx
            vqpca_2 = reduction.VQPCA(X, 3, 2, init='random', random_state=200)
            idx_2 = vqpca_2.idx
            self.assertTrue(not np.array_equal(idx_1, idx_2))
        except Exception:
            self.assertTrue(False)

        # Test that uniform cluster initialization is working properly:
        try:
            vqpca_1 = reduction.VQPCA(X, 3, 2, init='uniform')
            idx_1 = vqpca_1.idx
            vqpca_2 = reduction.VQPCA(X, 3, 2, init='uniform')
            idx_2 = vqpca_2.idx
            self.assertTrue(np.array_equal(idx_1, idx_2))
        except Exception:
            self.assertTrue(False)

# ------------------------------------------------------------------------------
