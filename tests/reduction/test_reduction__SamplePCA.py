import unittest
import numpy as np
from PCAfold import preprocess
from PCAfold import reduction
from PCAfold import analysis

class Reduction(unittest.TestCase):

    def test_reduction__SamplePCA__allowed_calls(self):

        X = np.random.rand(200,20)
        idx_X_r = np.arange(91,151,1)

        try:
            sample_pca = reduction.SamplePCA(X, idx_X_r, 'auto', 2, 1, X_source=None)

            eigenvalues = sample_pca.eigenvalues
            eigenvectors = sample_pca.eigenvectors
            pc_scores = sample_pca.pc_scores
            pc_sources = sample_pca.pc_sources
            C = sample_pca.C
            D = sample_pca.D
            C_r = sample_pca.C_r
            D_r = sample_pca.D_r

            sample_pca = reduction.SamplePCA(X, idx_X_r, 'auto', 2, 2, X_source=None)

            eigenvalues = sample_pca.eigenvalues
            eigenvectors = sample_pca.eigenvectors
            pc_scores = sample_pca.pc_scores
            pc_sources = sample_pca.pc_sources
            C = sample_pca.C
            D = sample_pca.D
            C_r = sample_pca.C_r
            D_r = sample_pca.D_r

            sample_pca = reduction.SamplePCA(X, idx_X_r, 'auto', 2, 3, X_source=None)

            eigenvalues = sample_pca.eigenvalues
            eigenvectors = sample_pca.eigenvectors
            pc_scores = sample_pca.pc_scores
            pc_sources = sample_pca.pc_sources
            C = sample_pca.C
            D = sample_pca.D
            C_r = sample_pca.C_r
            D_r = sample_pca.D_r

            sample_pca = reduction.SamplePCA(X, idx_X_r, 'auto', 2, 4, X_source=None)

            eigenvalues = sample_pca.eigenvalues
            eigenvectors = sample_pca.eigenvectors
            pc_scores = sample_pca.pc_scores
            pc_sources = sample_pca.pc_sources
            C = sample_pca.C
            D = sample_pca.D
            C_r = sample_pca.C_r
            D_r = sample_pca.D_r

        except Exception:
            self.assertTrue(False)

        try:
            sample_pca = reduction.SamplePCA(X, idx_X_r, 'range', 2, 1, X_source=None)

            eigenvalues = sample_pca.eigenvalues
            eigenvectors = sample_pca.eigenvectors
            pc_scores = sample_pca.pc_scores
            pc_sources = sample_pca.pc_sources
            C = sample_pca.C
            D = sample_pca.D
            C_r = sample_pca.C_r
            D_r = sample_pca.D_r

            sample_pca = reduction.SamplePCA(X, idx_X_r, 'range', 2, 2, X_source=None)

            eigenvalues = sample_pca.eigenvalues
            eigenvectors = sample_pca.eigenvectors
            pc_scores = sample_pca.pc_scores
            pc_sources = sample_pca.pc_sources
            C = sample_pca.C
            D = sample_pca.D
            C_r = sample_pca.C_r
            D_r = sample_pca.D_r

            sample_pca = reduction.SamplePCA(X, idx_X_r, 'range', 2, 3, X_source=None)

            eigenvalues = sample_pca.eigenvalues
            eigenvectors = sample_pca.eigenvectors
            pc_scores = sample_pca.pc_scores
            pc_sources = sample_pca.pc_sources
            C = sample_pca.C
            D = sample_pca.D
            C_r = sample_pca.C_r
            D_r = sample_pca.D_r

            sample_pca = reduction.SamplePCA(X, idx_X_r, 'range', 2, 4, X_source=None)

            eigenvalues = sample_pca.eigenvalues
            eigenvectors = sample_pca.eigenvectors
            pc_scores = sample_pca.pc_scores
            pc_sources = sample_pca.pc_sources
            C = sample_pca.C
            D = sample_pca.D
            C_r = sample_pca.C_r
            D_r = sample_pca.D_r

        except Exception:
            self.assertTrue(False)

        try:
            sample_pca = reduction.SamplePCA(X, idx_X_r, 'range', 1, 1, X_source=None)

            eigenvalues = sample_pca.eigenvalues
            eigenvectors = sample_pca.eigenvectors
            pc_scores = sample_pca.pc_scores
            pc_sources = sample_pca.pc_sources
            C = sample_pca.C
            D = sample_pca.D
            C_r = sample_pca.C_r
            D_r = sample_pca.D_r

            sample_pca = reduction.SamplePCA(X, idx_X_r, 'range', 1, 2, X_source=None)

            eigenvalues = sample_pca.eigenvalues
            eigenvectors = sample_pca.eigenvectors
            pc_scores = sample_pca.pc_scores
            pc_sources = sample_pca.pc_sources
            C = sample_pca.C
            D = sample_pca.D
            C_r = sample_pca.C_r
            D_r = sample_pca.D_r

            sample_pca = reduction.SamplePCA(X, idx_X_r, 'range', 1, 3, X_source=None)

            eigenvalues = sample_pca.eigenvalues
            eigenvectors = sample_pca.eigenvectors
            pc_scores = sample_pca.pc_scores
            pc_sources = sample_pca.pc_sources
            C = sample_pca.C
            D = sample_pca.D
            C_r = sample_pca.C_r
            D_r = sample_pca.D_r

            sample_pca = reduction.SamplePCA(X, idx_X_r, 'range', 1, 4, X_source=None)

            eigenvalues = sample_pca.eigenvalues
            eigenvectors = sample_pca.eigenvectors
            pc_scores = sample_pca.pc_scores
            pc_sources = sample_pca.pc_sources
            C = sample_pca.C
            D = sample_pca.D
            C_r = sample_pca.C_r
            D_r = sample_pca.D_r

        except Exception:
            self.assertTrue(False)

        X_source = np.random.rand(200,20)
        try:
            sample_pca = reduction.SamplePCA(X, idx_X_r, 'range', 2, 1, X_source=X_source)

            eigenvalues = sample_pca.eigenvalues
            eigenvectors = sample_pca.eigenvectors
            pc_scores = sample_pca.pc_scores
            pc_sources = sample_pca.pc_sources
            C = sample_pca.C
            D = sample_pca.D
            C_r = sample_pca.C_r
            D_r = sample_pca.D_r

            sample_pca = reduction.SamplePCA(X, idx_X_r, 'range', 2, 2, X_source=X_source)

            eigenvalues = sample_pca.eigenvalues
            eigenvectors = sample_pca.eigenvectors
            pc_scores = sample_pca.pc_scores
            pc_sources = sample_pca.pc_sources
            C = sample_pca.C
            D = sample_pca.D
            C_r = sample_pca.C_r
            D_r = sample_pca.D_r

            sample_pca = reduction.SamplePCA(X, idx_X_r, 'range', 2, 3, X_source=X_source)

            eigenvalues = sample_pca.eigenvalues
            eigenvectors = sample_pca.eigenvectors
            pc_scores = sample_pca.pc_scores
            pc_sources = sample_pca.pc_sources
            C = sample_pca.C
            D = sample_pca.D
            C_r = sample_pca.C_r
            D_r = sample_pca.D_r

            sample_pca = reduction.SamplePCA(X, idx_X_r, 'range', 2, 4, X_source=X_source)

            eigenvalues = sample_pca.eigenvalues
            eigenvectors = sample_pca.eigenvectors
            pc_scores = sample_pca.pc_scores
            pc_sources = sample_pca.pc_sources
            C = sample_pca.C
            D = sample_pca.D
            C_r = sample_pca.C_r
            D_r = sample_pca.D_r

        except Exception:
            self.assertTrue(False)

        try:
            sample_pca = reduction.SamplePCA(X, idx_X_r, 'pareto', 10, 1, X_source=X_source)

            eigenvalues = sample_pca.eigenvalues
            eigenvectors = sample_pca.eigenvectors
            pc_scores = sample_pca.pc_scores
            pc_sources = sample_pca.pc_sources
            C = sample_pca.C
            D = sample_pca.D
            C_r = sample_pca.C_r
            D_r = sample_pca.D_r

            sample_pca = reduction.SamplePCA(X, idx_X_r, 'pareto', 10, 2, X_source=X_source)

            eigenvalues = sample_pca.eigenvalues
            eigenvectors = sample_pca.eigenvectors
            pc_scores = sample_pca.pc_scores
            pc_sources = sample_pca.pc_sources
            C = sample_pca.C
            D = sample_pca.D
            C_r = sample_pca.C_r
            D_r = sample_pca.D_r

            sample_pca = reduction.SamplePCA(X, idx_X_r, 'pareto', 10, 3, X_source=X_source)

            eigenvalues = sample_pca.eigenvalues
            eigenvectors = sample_pca.eigenvectors
            pc_scores = sample_pca.pc_scores
            pc_sources = sample_pca.pc_sources
            C = sample_pca.C
            D = sample_pca.D
            C_r = sample_pca.C_r
            D_r = sample_pca.D_r

            sample_pca = reduction.SamplePCA(X, idx_X_r, 'pareto', 10, 4, X_source=X_source)

            eigenvalues = sample_pca.eigenvalues
            eigenvectors = sample_pca.eigenvectors
            pc_scores = sample_pca.pc_scores
            pc_sources = sample_pca.pc_sources
            C = sample_pca.C
            D = sample_pca.D
            C_r = sample_pca.C_r
            D_r = sample_pca.D_r

        except Exception:
            self.assertTrue(False)

        idx_X_r = np.arange(91,151,1)[:,None]

        try:
            sample_pca = reduction.SamplePCA(X, idx_X_r, 'auto', 2, 1, X_source=None)

            eigenvalues = sample_pca.eigenvalues
            eigenvectors = sample_pca.eigenvectors
            pc_scores = sample_pca.pc_scores
            pc_sources = sample_pca.pc_sources
            C = sample_pca.C
            D = sample_pca.D
            C_r = sample_pca.C_r
            D_r = sample_pca.D_r

            sample_pca = reduction.SamplePCA(X, idx_X_r, 'auto', 2, 2, X_source=None)

            eigenvalues = sample_pca.eigenvalues
            eigenvectors = sample_pca.eigenvectors
            pc_scores = sample_pca.pc_scores
            pc_sources = sample_pca.pc_sources
            C = sample_pca.C
            D = sample_pca.D
            C_r = sample_pca.C_r
            D_r = sample_pca.D_r

            sample_pca = reduction.SamplePCA(X, idx_X_r, 'auto', 2, 3, X_source=None)

            eigenvalues = sample_pca.eigenvalues
            eigenvectors = sample_pca.eigenvectors
            pc_scores = sample_pca.pc_scores
            pc_sources = sample_pca.pc_sources
            C = sample_pca.C
            D = sample_pca.D
            C_r = sample_pca.C_r
            D_r = sample_pca.D_r

            sample_pca = reduction.SamplePCA(X, idx_X_r, 'auto', 2, 4, X_source=None)

            eigenvalues = sample_pca.eigenvalues
            eigenvectors = sample_pca.eigenvectors
            pc_scores = sample_pca.pc_scores
            pc_sources = sample_pca.pc_sources
            C = sample_pca.C
            D = sample_pca.D
            C_r = sample_pca.C_r
            D_r = sample_pca.D_r

        except Exception:
            self.assertTrue(False)

    def test_reduction__SamplePCA__not_allowed_calls(self):

        X = np.random.rand(200,20)
        idx_X_r = np.arange(91,151,1)

        with self.assertRaises(ValueError):
            sample_pca = reduction.SamplePCA([], idx_X_r, 'auto', 2, 2, X_source=None)

        with self.assertRaises(ValueError):
            sample_pca = reduction.SamplePCA(X, [], 'auto', 2, 2, X_source=None)

        with self.assertRaises(ValueError):
            sample_pca = reduction.SamplePCA(X, idx_X_r, [], 2, 2, X_source=None)

        with self.assertRaises(ValueError):
            sample_pca = reduction.SamplePCA(X, idx_X_r, 'auto', [], 2, X_source=None)

        with self.assertRaises(ValueError):
            sample_pca = reduction.SamplePCA(X, idx_X_r, 'auto', 2, [], X_source=None)

        with self.assertRaises(ValueError):
            sample_pca = reduction.SamplePCA(X, idx_X_r, 'auto', 2, 2, X_source=[])

        with self.assertRaises(ValueError):
            sample_pca = reduction.SamplePCA(X, idx_X_r, 'auto', 2, 5, X_source=None)

        with self.assertRaises(ValueError):
            sample_pca = reduction.SamplePCA(X, idx_X_r, 'auto', 25, 2, X_source=None)

# ------------------------------------------------------------------------------
