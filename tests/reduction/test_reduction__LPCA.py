import unittest
import numpy as np
from PCAfold import preprocess
from PCAfold import reduction
from PCAfold import analysis

class Reduction(unittest.TestCase):

    def test_reduction__LPCA__allowed_calls(self):

        X = np.random.rand(100,10)
        idx = np.zeros((100,))
        idx[50:80] = 1
        idx = idx.astype(int)

        try:
            lpca_X = reduction.LPCA(X, idx, scaling='none', n_components=2)

            S = lpca_X.S
            A = lpca_X.A
            L = lpca_X.L
            Z = lpca_X.principal_components
            l = lpca_X.loadings
            tq = lpca_X.tq
            tqj = lpca_X.tqj

            S_k1 = lpca_X.S[0]
            A_k1 = lpca_X.A[0]
            L_k1 = lpca_X.L[0]
            Z_k1 = lpca_X.principal_components[0]
            l_k1 = lpca_X.loadings[0]
            tq_k1 = lpca_X.tq[0]
            tqj_k1 = lpca_X.tqj[0]

            S1_k1 = lpca_X.S[0][0,0]
            A1_k1 = lpca_X.A[0][:,0]
            L1_k1 = lpca_X.L[0][0]
            Z1_k1 = lpca_X.principal_components[0][:,0]
            l1_k1 = lpca_X.loadings[0][:,0]
            tq1_k1 = lpca_X.tq[0][0]
            tqj1_k1 = lpca_X.tqj[0][:,0]

        except Exception:
            self.assertTrue(False)

        idx = np.zeros((100,1))
        idx[50:80] = 1
        idx = idx.astype(int)

        try:
            lpca_X = reduction.LPCA(X, idx, scaling='none', n_components=2)

            S = lpca_X.S
            A = lpca_X.A
            L = lpca_X.L
            Z = lpca_X.principal_components
            l = lpca_X.loadings
            tq = lpca_X.tq
            tqj = lpca_X.tqj

            S_k1 = lpca_X.S[0]
            A_k1 = lpca_X.A[0]
            L_k1 = lpca_X.L[0]
            Z_k1 = lpca_X.principal_components[0]
            l_k1 = lpca_X.loadings[0]
            tq_k1 = lpca_X.tq[0]
            tqj_k1 = lpca_X.tqj[0]

            S1_k1 = lpca_X.S[0][0,0]
            A1_k1 = lpca_X.A[0][:,0]
            L1_k1 = lpca_X.L[0][0]
            Z1_k1 = lpca_X.principal_components[0][:,0]
            l1_k1 = lpca_X.loadings[0][:,0]
            tq1_k1 = lpca_X.tq[0][0]
            tqj_k1 = lpca_X.tqj[0][:,0]

        except Exception:
            self.assertTrue(False)

# ------------------------------------------------------------------------------

    def test_reduction__LPCA_equivalent_to_PCA_for_one_cluster(self):

        X = np.random.rand(100,10)
        idx = np.zeros((100,))
        idx = idx.astype(int)

        try:
            lpca_X = reduction.LPCA(X, idx, scaling='none', n_components=2)
            pca_X = reduction.PCA(X, scaling='none', n_components=2)

            lpca_S = lpca_X.S[0]
            pca_S = pca_X.S
            self.assertTrue(np.array_equal(lpca_S, pca_S))

            lpca_A = lpca_X.A[0]
            pca_A = pca_X.A[:,0:2]
            self.assertTrue(np.array_equal(lpca_A, pca_A))

            lpca_L = lpca_X.L[0]
            pca_L = pca_X.L[0:2]
            self.assertTrue(np.array_equal(lpca_L, pca_L))

            lpca_Z = lpca_X.principal_components[0]
            pca_Z = pca_X.transform(X)
            self.assertTrue(np.array_equal(lpca_Z, pca_Z))

            lpca_loadings = lpca_X.loadings[0]
            pca_loadings = pca_X.loadings
            self.assertTrue(np.array_equal(lpca_loadings, pca_loadings))

            lpca_tq = lpca_X.tq[0]
            pca_tq = pca_X.tq
            self.assertTrue(np.array_equal(lpca_tq, pca_tq))

            lpca_tqj = lpca_X.tqj[0]
            pca_tqj = pca_X.tqj
            self.assertTrue(np.array_equal(lpca_tqj, pca_tqj))

        except:
            self.assertTrue(False)

        try:
            lpca_X = reduction.LPCA(X, idx, scaling='range', n_components=2)
            pca_X = reduction.PCA(X, scaling='range', n_components=2)

            lpca_S = lpca_X.S[0]
            pca_S = pca_X.S
            self.assertTrue(np.array_equal(lpca_S, pca_S))

            lpca_A = lpca_X.A[0]
            pca_A = pca_X.A[:,0:2]
            self.assertTrue(np.array_equal(lpca_A, pca_A))

            lpca_L = lpca_X.L[0]
            pca_L = pca_X.L[0:2]
            self.assertTrue(np.array_equal(lpca_L, pca_L))

            lpca_Z = lpca_X.principal_components[0]
            pca_Z = pca_X.transform(X)
            self.assertTrue(np.array_equal(lpca_Z, pca_Z))

            lpca_loadings = lpca_X.loadings[0]
            pca_loadings = pca_X.loadings
            self.assertTrue(np.array_equal(lpca_loadings, pca_loadings))

            lpca_tq = lpca_X.tq[0]
            pca_tq = pca_X.tq
            self.assertTrue(np.array_equal(lpca_tq, pca_tq))

            lpca_tqj = lpca_X.tqj[0]
            pca_tqj = pca_X.tqj
            self.assertTrue(np.array_equal(lpca_tqj, pca_tqj))

        except:
            self.assertTrue(False)

        try:
            lpca_X = reduction.LPCA(X, idx, scaling='auto', n_components=0)
            pca_X = reduction.PCA(X, scaling='auto', n_components=0)

            lpca_S = lpca_X.S[0]
            pca_S = pca_X.S
            self.assertTrue(np.array_equal(lpca_S, pca_S))

            lpca_A = lpca_X.A[0]
            pca_A = pca_X.A
            self.assertTrue(np.array_equal(lpca_A, pca_A))

            lpca_L = lpca_X.L[0]
            pca_L = pca_X.L
            self.assertTrue(np.array_equal(lpca_L, pca_L))

            lpca_Z = lpca_X.principal_components[0]
            pca_Z = pca_X.transform(X)
            self.assertTrue(np.array_equal(lpca_Z, pca_Z))

            lpca_loadings = lpca_X.loadings[0]
            pca_loadings = pca_X.loadings
            self.assertTrue(np.array_equal(lpca_loadings, pca_loadings))

            lpca_tq = lpca_X.tq[0]
            pca_tq = pca_X.tq
            self.assertTrue(np.array_equal(lpca_tq, pca_tq))

            lpca_tqj = lpca_X.tqj[0]
            pca_tqj = pca_X.tqj
            self.assertTrue(np.array_equal(lpca_tqj, pca_tqj))

        except:
            self.assertTrue(False)

        X = np.random.rand(100,10)
        idx = np.zeros((100,1))
        idx = idx.astype(int)

        try:
            lpca_X = reduction.LPCA(X, idx, scaling='none', n_components=2)
            pca_X = reduction.PCA(X, scaling='none', n_components=2)

            lpca_S = lpca_X.S[0]
            pca_S = pca_X.S
            self.assertTrue(np.array_equal(lpca_S, pca_S))

            lpca_A = lpca_X.A[0]
            pca_A = pca_X.A[:,0:2]
            self.assertTrue(np.array_equal(lpca_A, pca_A))

            lpca_L = lpca_X.L[0]
            pca_L = pca_X.L[0:2]
            self.assertTrue(np.array_equal(lpca_L, pca_L))

            lpca_Z = lpca_X.principal_components[0]
            pca_Z = pca_X.transform(X)
            self.assertTrue(np.array_equal(lpca_Z, pca_Z))

            lpca_loadings = lpca_X.loadings[0]
            pca_loadings = pca_X.loadings
            self.assertTrue(np.array_equal(lpca_loadings, pca_loadings))

            lpca_tq = lpca_X.tq[0]
            pca_tq = pca_X.tq
            self.assertTrue(np.array_equal(lpca_tq, pca_tq))

            lpca_tqj = lpca_X.tqj[0]
            pca_tqj = pca_X.tqj
            self.assertTrue(np.array_equal(lpca_tqj, pca_tqj))

        except:
            self.assertTrue(False)

        try:
            lpca_X = reduction.LPCA(X, idx, scaling='range', n_components=2)
            pca_X = reduction.PCA(X, scaling='range', n_components=2)

            lpca_S = lpca_X.S[0]
            pca_S = pca_X.S
            self.assertTrue(np.array_equal(lpca_S, pca_S))

            lpca_A = lpca_X.A[0]
            pca_A = pca_X.A[:,0:2]
            self.assertTrue(np.array_equal(lpca_A, pca_A))

            lpca_L = lpca_X.L[0]
            pca_L = pca_X.L[0:2]
            self.assertTrue(np.array_equal(lpca_L, pca_L))

            lpca_Z = lpca_X.principal_components[0]
            pca_Z = pca_X.transform(X)
            self.assertTrue(np.array_equal(lpca_Z, pca_Z))

            lpca_loadings = lpca_X.loadings[0]
            pca_loadings = pca_X.loadings
            self.assertTrue(np.array_equal(lpca_loadings, pca_loadings))

            lpca_tq = lpca_X.tq[0]
            pca_tq = pca_X.tq
            self.assertTrue(np.array_equal(lpca_tq, pca_tq))

            lpca_tqj = lpca_X.tqj[0]
            pca_tqj = pca_X.tqj
            self.assertTrue(np.array_equal(lpca_tqj, pca_tqj))

        except:
            self.assertTrue(False)

        try:
            lpca_X = reduction.LPCA(X, idx, scaling='auto', n_components=0)
            pca_X = reduction.PCA(X, scaling='auto', n_components=0)

            lpca_S = lpca_X.S[0]
            pca_S = pca_X.S
            self.assertTrue(np.array_equal(lpca_S, pca_S))

            lpca_A = lpca_X.A[0]
            pca_A = pca_X.A
            self.assertTrue(np.array_equal(lpca_A, pca_A))

            lpca_L = lpca_X.L[0]
            pca_L = pca_X.L
            self.assertTrue(np.array_equal(lpca_L, pca_L))

            lpca_Z = lpca_X.principal_components[0]
            pca_Z = pca_X.transform(X)
            self.assertTrue(np.array_equal(lpca_Z, pca_Z))

            lpca_loadings = lpca_X.loadings[0]
            pca_loadings = pca_X.loadings
            self.assertTrue(np.array_equal(lpca_loadings, pca_loadings))

            lpca_tq = lpca_X.tq[0]
            pca_tq = pca_X.tq
            self.assertTrue(np.array_equal(lpca_tq, pca_tq))

            lpca_tqj = lpca_X.tqj[0]
            pca_tqj = pca_X.tqj
            self.assertTrue(np.array_equal(lpca_tqj, pca_tqj))

        except:
            self.assertTrue(False)

# ------------------------------------------------------------------------------

    def test_reduction__LPCA__not_allowed_calls(self):

        X = np.random.rand(100,10)
        idx = np.zeros((100,))
        idx[50:80] = 1
        idx = idx.astype(int)
        lpca_X = reduction.LPCA(X, idx, scaling='none', n_components=5)

        with self.assertRaises(IndexError):
            S = lpca_X.S[2]

        with self.assertRaises(IndexError):
            A = lpca_X.A[2]

        with self.assertRaises(IndexError):
            L = lpca_X.L[2]

        with self.assertRaises(IndexError):
            Z = lpca_X.principal_components[2]

        with self.assertRaises(IndexError):
            loadings = lpca_X.loadings[2]

        with self.assertRaises(IndexError):
            tq = lpca_X.tq[2]

        with self.assertRaises(IndexError):
            tqj = lpca_X.tqj[2]

        with self.assertRaises(IndexError):
            A = lpca_X.A[0][:,8]

        with self.assertRaises(IndexError):
            L = lpca_X.L[0][8]

        with self.assertRaises(IndexError):
            Z = lpca_X.principal_components[0][:,8]

        with self.assertRaises(IndexError):
            loadings = lpca_X.loadings[0][:,8]

        with self.assertRaises(IndexError):
            tqj = lpca_X.tqj[0][:,8]

        with self.assertRaises(ValueError):
            lpca_X = reduction.LPCA([1,2,3], idx, scaling='none', n_components=5)

        with self.assertRaises(ValueError):
            lpca_X = reduction.LPCA(X, [1,2,3], scaling='none', n_components=5)

        with self.assertRaises(ValueError):
            lpca_X = reduction.LPCA(X, idx, scaling=1, n_components=5)

        with self.assertRaises(ValueError):
            lpca_X = reduction.LPCA(X, idx, scaling='none', n_components=20)

        X = np.random.rand(100,10)
        idx = np.zeros((200,))
        idx[50:80] = 1
        idx = idx.astype(int)

        with self.assertRaises(ValueError):
            lpca_X = reduction.LPCA(X, idx)

        X = np.random.rand(100,10)
        idx = np.zeros((100,2))
        idx = idx.astype(int)

        with self.assertRaises(ValueError):
            lpca_X = reduction.LPCA(X, idx)

# ------------------------------------------------------------------------------

    def test_reduction__LPCA__not_allowed_attribute_set(self):

        X = np.random.rand(100,5)
        idx = np.zeros((100,))
        idx[50:80] = 1
        idx = idx.astype(int)
        lpca_X = reduction.LPCA(X, idx)

        with self.assertRaises(AttributeError):
            lpca_X.S = 1

        with self.assertRaises(AttributeError):
            lpca_X.A = 1

        with self.assertRaises(AttributeError):
            lpca_X.L = 1

        with self.assertRaises(AttributeError):
            lpca_X.principal_components = 1

        with self.assertRaises(AttributeError):
            lpca_X.loadings = 1

        with self.assertRaises(AttributeError):
            lpca_X.tq = 1

        with self.assertRaises(AttributeError):
            lpca_X.tqj = 1

        with self.assertRaises(AttributeError):
            lpca_X.X_reconstructed = 1

# ------------------------------------------------------------------------------

    def test_reduction__LPCA__data_reconstruction(self):

        X = np.random.rand(100,10)
        idx = np.zeros((100,))
        idx[50:80] = 1
        idx = idx.astype(int)

        try:
            lpca_X = reduction.LPCA(X, idx, scaling='none', n_components=2)
            X_reconstructed = lpca_X.X_reconstructed
            (n_observations, n_variables) = np.shape(X_reconstructed)
            self.assertTrue(n_observations==100)
            self.assertTrue(n_variables==10)
        except:
            self.assertTrue(False)

        X = np.random.rand(100,10)
        idx = np.zeros((100,))
        idx[50:80] = 1
        idx = idx.astype(int)
        X[50:80,2] = np.ones_like(idx[50:80])

        try:
            lpca_X = reduction.LPCA(X, idx, scaling='none', n_components=2)
            X_reconstructed = lpca_X.X_reconstructed
            (n_observations, n_variables) = np.shape(X_reconstructed)
            self.assertTrue(n_observations==100)
            self.assertTrue(n_variables==10)
        except:
            self.assertTrue(False)

        X = np.random.rand(100,10)
        idx = np.zeros((100,))
        idx[50:80] = 1
        idx = idx.astype(int)
        X[:,2] = np.ones_like(idx)

        try:
            lpca_X = reduction.LPCA(X, idx, scaling='none', n_components=2)
            X_reconstructed = lpca_X.X_reconstructed
            (n_observations, n_variables) = np.shape(X_reconstructed)
            self.assertTrue(n_observations==100)
            self.assertTrue(n_variables==10)
        except:
            self.assertTrue(False)

# ------------------------------------------------------------------------------
