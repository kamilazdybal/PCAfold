import unittest
import numpy as np
from PCAfold import preprocess
from PCAfold import reduction
from PCAfold import analysis

class Reduction(unittest.TestCase):

    def test_reduction__EquilibratedSamplePCA__allowed_calls(self):

        X = np.random.rand(200,20)
        X_source = np.random.rand(200,20)

        idx = np.zeros((200,))
        idx[20:60,] = 1
        idx[150:190] = 2
        idx = idx.astype(int)

        try:

            equilibrate_pca = reduction.EquilibratedSamplePCA(X, idx, 'auto', 2, 1, X_source=None, n_iterations=10, stop_iter=0, random_seed=None, verbose=False)

            eigenvalues = equilibrate_pca.eigenvalues
            eigenvectors = equilibrate_pca.eigenvectors
            pc_scores = equilibrate_pca.pc_scores
            pc_sources = equilibrate_pca.pc_sources
            idx_train = equilibrate_pca.idx_train
            C_r = equilibrate_pca.C_r
            D_r = equilibrate_pca.D_r


            equilibrate_pca = reduction.EquilibratedSamplePCA(X, idx, 'auto', 2, 2, X_source=None, n_iterations=10, stop_iter=0, random_seed=None, verbose=False)

            eigenvalues = equilibrate_pca.eigenvalues
            eigenvectors = equilibrate_pca.eigenvectors
            pc_scores = equilibrate_pca.pc_scores
            pc_sources = equilibrate_pca.pc_sources
            idx_train = equilibrate_pca.idx_train
            C_r = equilibrate_pca.C_r
            D_r = equilibrate_pca.D_r

            equilibrate_pca = reduction.EquilibratedSamplePCA(X, idx, 'auto', 2, 3, X_source=None, n_iterations=10, stop_iter=0, random_seed=None, verbose=False)

            eigenvalues = equilibrate_pca.eigenvalues
            eigenvectors = equilibrate_pca.eigenvectors
            pc_scores = equilibrate_pca.pc_scores
            pc_sources = equilibrate_pca.pc_sources
            idx_train = equilibrate_pca.idx_train
            C_r = equilibrate_pca.C_r
            D_r = equilibrate_pca.D_r

            equilibrate_pca = reduction.EquilibratedSamplePCA(X, idx, 'auto', 2, 4, X_source=None, n_iterations=10, stop_iter=0, random_seed=None, verbose=False)

            eigenvalues = equilibrate_pca.eigenvalues
            eigenvectors = equilibrate_pca.eigenvectors
            pc_scores = equilibrate_pca.pc_scores
            pc_sources = equilibrate_pca.pc_sources
            idx_train = equilibrate_pca.idx_train
            C_r = equilibrate_pca.C_r
            D_r = equilibrate_pca.D_r

        except Exception:
            self.assertTrue(False)

        try:
            equilibrate_pca = reduction.EquilibratedSamplePCA(X, idx, 'auto', 2, 1, X_source=None, n_iterations=1, stop_iter=0, random_seed=None, verbose=False)

            eigenvalues = equilibrate_pca.eigenvalues
            eigenvectors = equilibrate_pca.eigenvectors
            pc_scores = equilibrate_pca.pc_scores
            pc_sources = equilibrate_pca.pc_sources
            idx_train = equilibrate_pca.idx_train
            C_r = equilibrate_pca.C_r
            D_r = equilibrate_pca.D_r

            equilibrate_pca = reduction.EquilibratedSamplePCA(X, idx, 'auto', 2, 2, X_source=None, n_iterations=1, stop_iter=0, random_seed=None, verbose=False)

            eigenvalues = equilibrate_pca.eigenvalues
            eigenvectors = equilibrate_pca.eigenvectors
            pc_scores = equilibrate_pca.pc_scores
            pc_sources = equilibrate_pca.pc_sources
            idx_train = equilibrate_pca.idx_train
            C_r = equilibrate_pca.C_r
            D_r = equilibrate_pca.D_r

            equilibrate_pca = reduction.EquilibratedSamplePCA(X, idx, 'auto', 2, 3, X_source=None, n_iterations=1, stop_iter=0, random_seed=None, verbose=False)

            eigenvalues = equilibrate_pca.eigenvalues
            eigenvectors = equilibrate_pca.eigenvectors
            pc_scores = equilibrate_pca.pc_scores
            pc_sources = equilibrate_pca.pc_sources
            idx_train = equilibrate_pca.idx_train
            C_r = equilibrate_pca.C_r
            D_r = equilibrate_pca.D_r

            equilibrate_pca = reduction.EquilibratedSamplePCA(X, idx, 'auto', 2, 4, X_source=None, n_iterations=1, stop_iter=0, random_seed=None, verbose=False)

            eigenvalues = equilibrate_pca.eigenvalues
            eigenvectors = equilibrate_pca.eigenvectors
            pc_scores = equilibrate_pca.pc_scores
            pc_sources = equilibrate_pca.pc_sources
            idx_train = equilibrate_pca.idx_train
            C_r = equilibrate_pca.C_r
            D_r = equilibrate_pca.D_r

        except Exception:
            self.assertTrue(False)

        try:
            equilibrate_pca = reduction.EquilibratedSamplePCA(X, idx, 'range', 2, 1, X_source=None, n_iterations=1, stop_iter=0, random_seed=100, verbose=False)

            eigenvalues = equilibrate_pca.eigenvalues
            eigenvectors = equilibrate_pca.eigenvectors
            pc_scores = equilibrate_pca.pc_scores
            pc_sources = equilibrate_pca.pc_sources
            idx_train = equilibrate_pca.idx_train
            C_r = equilibrate_pca.C_r
            D_r = equilibrate_pca.D_r

            equilibrate_pca = reduction.EquilibratedSamplePCA(X, idx, 'range', 2, 2, X_source=None, n_iterations=1, stop_iter=0, random_seed=100, verbose=False)

            eigenvalues = equilibrate_pca.eigenvalues
            eigenvectors = equilibrate_pca.eigenvectors
            pc_scores = equilibrate_pca.pc_scores
            pc_sources = equilibrate_pca.pc_sources
            idx_train = equilibrate_pca.idx_train
            C_r = equilibrate_pca.C_r
            D_r = equilibrate_pca.D_r

            equilibrate_pca = reduction.EquilibratedSamplePCA(X, idx, 'range', 2, 3, X_source=None, n_iterations=1, stop_iter=0, random_seed=100, verbose=False)

            eigenvalues = equilibrate_pca.eigenvalues
            eigenvectors = equilibrate_pca.eigenvectors
            pc_scores = equilibrate_pca.pc_scores
            pc_sources = equilibrate_pca.pc_sources
            idx_train = equilibrate_pca.idx_train
            C_r = equilibrate_pca.C_r
            D_r = equilibrate_pca.D_r

            equilibrate_pca = reduction.EquilibratedSamplePCA(X, idx, 'range', 2, 4, X_source=None, n_iterations=1, stop_iter=0, random_seed=100, verbose=False)

            eigenvalues = equilibrate_pca.eigenvalues
            eigenvectors = equilibrate_pca.eigenvectors
            pc_scores = equilibrate_pca.pc_scores
            pc_sources = equilibrate_pca.pc_sources
            idx_train = equilibrate_pca.idx_train
            C_r = equilibrate_pca.C_r
            D_r = equilibrate_pca.D_r

        except Exception:
            self.assertTrue(False)

        try:

            equilibrate_pca = reduction.EquilibratedSamplePCA(X, idx, 'auto', 2, 1, X_source=X_source, n_iterations=10, stop_iter=0, random_seed=None, verbose=False)

            eigenvalues = equilibrate_pca.eigenvalues
            eigenvectors = equilibrate_pca.eigenvectors
            pc_scores = equilibrate_pca.pc_scores
            pc_sources = equilibrate_pca.pc_sources
            idx_train = equilibrate_pca.idx_train
            C_r = equilibrate_pca.C_r
            D_r = equilibrate_pca.D_r


            equilibrate_pca = reduction.EquilibratedSamplePCA(X, idx, 'auto', 2, 2, X_source=X_source, n_iterations=10, stop_iter=0, random_seed=None, verbose=False)

            eigenvalues = equilibrate_pca.eigenvalues
            eigenvectors = equilibrate_pca.eigenvectors
            pc_scores = equilibrate_pca.pc_scores
            pc_sources = equilibrate_pca.pc_sources
            idx_train = equilibrate_pca.idx_train
            C_r = equilibrate_pca.C_r
            D_r = equilibrate_pca.D_r

            equilibrate_pca = reduction.EquilibratedSamplePCA(X, idx, 'auto', 2, 3, X_source=X_source, n_iterations=10, stop_iter=0, random_seed=None, verbose=False)

            eigenvalues = equilibrate_pca.eigenvalues
            eigenvectors = equilibrate_pca.eigenvectors
            pc_scores = equilibrate_pca.pc_scores
            pc_sources = equilibrate_pca.pc_sources
            idx_train = equilibrate_pca.idx_train
            C_r = equilibrate_pca.C_r
            D_r = equilibrate_pca.D_r

            equilibrate_pca = reduction.EquilibratedSamplePCA(X, idx, 'auto', 2, 4, X_source=X_source, n_iterations=10, stop_iter=0, random_seed=None, verbose=False)

            eigenvalues = equilibrate_pca.eigenvalues
            eigenvectors = equilibrate_pca.eigenvectors
            pc_scores = equilibrate_pca.pc_scores
            pc_sources = equilibrate_pca.pc_sources
            idx_train = equilibrate_pca.idx_train
            C_r = equilibrate_pca.C_r
            D_r = equilibrate_pca.D_r

        except Exception:
            self.assertTrue(False)

        try:
            equilibrate_pca = reduction.EquilibratedSamplePCA(X, idx, 'auto', 2, 1, X_source=X_source, n_iterations=1, stop_iter=0, random_seed=None, verbose=False)

            eigenvalues = equilibrate_pca.eigenvalues
            eigenvectors = equilibrate_pca.eigenvectors
            pc_scores = equilibrate_pca.pc_scores
            pc_sources = equilibrate_pca.pc_sources
            idx_train = equilibrate_pca.idx_train
            C_r = equilibrate_pca.C_r
            D_r = equilibrate_pca.D_r

            equilibrate_pca = reduction.EquilibratedSamplePCA(X, idx, 'auto', 2, 2, X_source=None, n_iterations=1, stop_iter=0, random_seed=None, verbose=False)

            eigenvalues = equilibrate_pca.eigenvalues
            eigenvectors = equilibrate_pca.eigenvectors
            pc_scores = equilibrate_pca.pc_scores
            pc_sources = equilibrate_pca.pc_sources
            idx_train = equilibrate_pca.idx_train
            C_r = equilibrate_pca.C_r
            D_r = equilibrate_pca.D_r

            equilibrate_pca = reduction.EquilibratedSamplePCA(X, idx, 'auto', 2, 3, X_source=X_source, n_iterations=1, stop_iter=0, random_seed=None, verbose=False)

            eigenvalues = equilibrate_pca.eigenvalues
            eigenvectors = equilibrate_pca.eigenvectors
            pc_scores = equilibrate_pca.pc_scores
            pc_sources = equilibrate_pca.pc_sources
            idx_train = equilibrate_pca.idx_train
            C_r = equilibrate_pca.C_r
            D_r = equilibrate_pca.D_r

            equilibrate_pca = reduction.EquilibratedSamplePCA(X, idx, 'auto', 2, 4, X_source=X_source, n_iterations=1, stop_iter=0, random_seed=None, verbose=False)

            eigenvalues = equilibrate_pca.eigenvalues
            eigenvectors = equilibrate_pca.eigenvectors
            pc_scores = equilibrate_pca.pc_scores
            pc_sources = equilibrate_pca.pc_sources
            idx_train = equilibrate_pca.idx_train
            C_r = equilibrate_pca.C_r
            D_r = equilibrate_pca.D_r

        except Exception:
            self.assertTrue(False)

        try:
            equilibrate_pca = reduction.EquilibratedSamplePCA(X, idx, 'range', 2, 1, X_source=X_source, n_iterations=1, stop_iter=0, random_seed=100, verbose=False)

            eigenvalues = equilibrate_pca.eigenvalues
            eigenvectors = equilibrate_pca.eigenvectors
            pc_scores = equilibrate_pca.pc_scores
            pc_sources = equilibrate_pca.pc_sources
            idx_train = equilibrate_pca.idx_train
            C_r = equilibrate_pca.C_r
            D_r = equilibrate_pca.D_r

            equilibrate_pca = reduction.EquilibratedSamplePCA(X, idx, 'range', 2, 2, X_source=X_source, n_iterations=1, stop_iter=0, random_seed=100, verbose=False)

            eigenvalues = equilibrate_pca.eigenvalues
            eigenvectors = equilibrate_pca.eigenvectors
            pc_scores = equilibrate_pca.pc_scores
            pc_sources = equilibrate_pca.pc_sources
            idx_train = equilibrate_pca.idx_train
            C_r = equilibrate_pca.C_r
            D_r = equilibrate_pca.D_r

            equilibrate_pca = reduction.EquilibratedSamplePCA(X, idx, 'range', 2, 3, X_source=X_source, n_iterations=1, stop_iter=0, random_seed=100, verbose=False)

            eigenvalues = equilibrate_pca.eigenvalues
            eigenvectors = equilibrate_pca.eigenvectors
            pc_scores = equilibrate_pca.pc_scores
            pc_sources = equilibrate_pca.pc_sources
            idx_train = equilibrate_pca.idx_train
            C_r = equilibrate_pca.C_r
            D_r = equilibrate_pca.D_r

            equilibrate_pca = reduction.EquilibratedSamplePCA(X, idx, 'range', 2, 4, X_source=X_source, n_iterations=1, stop_iter=0, random_seed=100, verbose=False)

            eigenvalues = equilibrate_pca.eigenvalues
            eigenvectors = equilibrate_pca.eigenvectors
            pc_scores = equilibrate_pca.pc_scores
            pc_sources = equilibrate_pca.pc_sources
            idx_train = equilibrate_pca.idx_train
            C_r = equilibrate_pca.C_r
            D_r = equilibrate_pca.D_r

        except Exception:
            self.assertTrue(False)

    def test_reduction__EquilibratedSamplePCA__not_allowed_calls(self):

        X = np.random.rand(200,20)
        X_source = np.random.rand(200,20)

        idx = np.zeros((200,))
        idx[20:60,] = 1
        idx[150:190] = 2
        idx = idx.astype(int)

        with self.assertRaises(ValueError):
            equilibrate_pca = reduction.EquilibratedSamplePCA([], idx, 'range', 2, 4, X_source=X_source, n_iterations=1, stop_iter=0, random_seed=100, verbose=False)

        with self.assertRaises(ValueError):
            equilibrate_pca = reduction.EquilibratedSamplePCA(X, [], 'range', 2, 4, X_source=X_source, n_iterations=1, stop_iter=0, random_seed=100, verbose=False)

        with self.assertRaises(ValueError):
            equilibrate_pca = reduction.EquilibratedSamplePCA(X, idx, [], 2, 4, X_source=X_source, n_iterations=1, stop_iter=0, random_seed=100, verbose=False)

        with self.assertRaises(ValueError):
            equilibrate_pca = reduction.EquilibratedSamplePCA(X, idx, 'range', [], 4, X_source=X_source, n_iterations=1, stop_iter=0, random_seed=100, verbose=False)

        with self.assertRaises(ValueError):
            equilibrate_pca = reduction.EquilibratedSamplePCA(X, idx, 'range', 2, [], X_source=X_source, n_iterations=1, stop_iter=0, random_seed=100, verbose=False)

        with self.assertRaises(ValueError):
            equilibrate_pca = reduction.EquilibratedSamplePCA(X, idx, 'range', 2, 4, X_source=[], n_iterations=1, stop_iter=0, random_seed=100, verbose=False)

        with self.assertRaises(ValueError):
            equilibrate_pca = reduction.EquilibratedSamplePCA(X, idx, 'range', 2, 4, X_source=X_source, n_iterations=[], stop_iter=0, random_seed=100, verbose=False)

        with self.assertRaises(ValueError):
            equilibrate_pca = reduction.EquilibratedSamplePCA(X, idx, 'range', 2, 4, X_source=X_source, n_iterations=1, stop_iter=[], random_seed=100, verbose=False)

        with self.assertRaises(ValueError):
            equilibrate_pca = reduction.EquilibratedSamplePCA(X, idx, 'range', 2, 4, X_source=X_source, n_iterations=1, stop_iter=0, random_seed=[], verbose=False)

        with self.assertRaises(ValueError):
            equilibrate_pca = reduction.EquilibratedSamplePCA(X, idx, 'range', 2, 4, X_source=X_source, n_iterations=1, stop_iter=0, random_seed=100, verbose=[])

# ------------------------------------------------------------------------------
