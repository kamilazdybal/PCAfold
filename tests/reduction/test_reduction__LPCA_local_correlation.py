import unittest
import numpy as np
from PCAfold import preprocess
from PCAfold import reduction
from PCAfold import analysis

class Reduction(unittest.TestCase):

    def test_reduction__LPCA_local_correlation__allowed_calls(self):

        X = np.random.rand(100,5)
        idx = np.zeros((100,))
        idx[50:80] = 1
        idx = idx.astype(int)

        try:
            lpca = reduction.LPCA(X, idx)
            (local_correlations, weighted, unweighted) = lpca.local_correlation(X[:,0])
        except:
            self.assertTrue(False)

        try:
            lpca = reduction.LPCA(X, idx)
            (local_correlations, weighted, unweighted) = lpca.local_correlation(X[:,0:1])
        except:
            self.assertTrue(False)

        try:
            lpca = reduction.LPCA(X, idx)
            (local_correlations, weighted, unweighted) = lpca.local_correlation(X[:,0], index=1, metric='pearson')
        except:
            self.assertTrue(False)

        try:
            lpca = reduction.LPCA(X, idx, n_components=2)
            (local_correlations, weighted, unweighted) = lpca.local_correlation(X[:,0], index=1, metric='pearson')
        except:
            self.assertTrue(False)

        try:
            lpca = reduction.LPCA(X, idx, n_components=0)
            (local_correlations, weighted, unweighted) = lpca.local_correlation(X[:,0], index=4, metric='pearson')
        except:
            self.assertTrue(False)

        try:
            lpca = reduction.LPCA(X, idx)
            (local_correlations, weighted, unweighted) = lpca.local_correlation(X[:,0], index=1, metric='spearman')
        except:
            self.assertTrue(False)

        try:
            lpca = reduction.LPCA(X, idx, n_components=2)
            (local_correlations, weighted, unweighted) = lpca.local_correlation(X[:,0], index=1, metric='spearman')
        except:
            self.assertTrue(False)

        try:
            lpca = reduction.LPCA(X, idx, n_components=0)
            (local_correlations, weighted, unweighted) = lpca.local_correlation(X[:,0], index=4, metric='spearman')
        except:
            self.assertTrue(False)

# ------------------------------------------------------------------------------

    def test_reduction__LPCA_local_correlation__not_allowed_calls(self):

        X = np.random.rand(100,5)
        idx = np.zeros((100,))
        idx[50:80] = 1
        idx = idx.astype(int)
        lpca = reduction.LPCA(X, idx, n_components=2)

        with self.assertRaises(ValueError):
            (local_correlations, weighted, unweighted) = lpca.local_correlation(X[:,0:2])

        with self.assertRaises(ValueError):
            (local_correlations, weighted, unweighted) = lpca.local_correlation(X[0:50,0])

        with self.assertRaises(ValueError):
            (local_correlations, weighted, unweighted) = lpca.local_correlation(X[:,0], index=-1)

        with self.assertRaises(ValueError):
            (local_correlations, weighted, unweighted) = lpca.local_correlation(X[:,0], index=2)

        with self.assertRaises(ValueError):
            (local_correlations, weighted, unweighted) = lpca.local_correlation(X[:,0], metric='none')

        with self.assertRaises(ValueError):
            (local_correlations, weighted, unweighted) = lpca.local_correlation(X[:,0], verbose=1)

# ------------------------------------------------------------------------------
