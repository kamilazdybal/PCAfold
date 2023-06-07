import unittest
import numpy as np
from PCAfold import preprocess
from PCAfold import reduction
from PCAfold import reconstruction

class Reconstruction(unittest.TestCase):

    def test_reconstruction__stratified_coefficient_of_determination__allowed_calls(self):

        X = np.random.rand(100,10)
        pca_X = reduction.PCA(X, scaling='auto', n_components=2)
        X_rec = pca_X.reconstruct(pca_X.transform(X))

        try:
            (idx, bins_borders) = preprocess.variable_bins(X[:,0], k=10, verbose=False)
            r2_in_bins = reconstruction.stratified_coefficient_of_determination(X[:,0], X_rec[:,0], idx=idx)
        except Exception:
            self.assertTrue(False)

        try:
            (idx, bins_borders) = preprocess.variable_bins(X[:,0], k=1, verbose=False)
            r2_in_bins = reconstruction.stratified_coefficient_of_determination(X[:,3], X_rec[:,3], idx=idx)
        except Exception:
            self.assertTrue(False)

        try:
            (idx, bins_borders) = preprocess.variable_bins(X[:,0], k=10, verbose=False)
            r2_in_bins = reconstruction.stratified_coefficient_of_determination(X[:,0], X_rec[:,0], idx=idx, use_global_mean=True, verbose=False)
        except Exception:
            self.assertTrue(False)

        try:
            (idx, bins_borders) = preprocess.variable_bins(X[:,0], k=10, verbose=False)
            r2_in_bins = reconstruction.stratified_coefficient_of_determination(X[:,1], X_rec[:,1], idx=idx, use_global_mean=False, verbose=False)
        except Exception:
            self.assertTrue(False)

        try:
            (idx, bins_borders) = preprocess.variable_bins(X[:,0], k=10, verbose=False)
            r2_in_bins = reconstruction.stratified_coefficient_of_determination(X[:,0:1], X_rec[:,0], idx=idx)
        except Exception:
            self.assertTrue(False)

        try:
            (idx, bins_borders) = preprocess.variable_bins(X[:,0], k=10, verbose=False)
            r2_in_bins = reconstruction.stratified_coefficient_of_determination(X[:,3], X_rec[:,3:4], idx=idx)
        except Exception:
            self.assertTrue(False)

        try:
            (idx, bins_borders) = preprocess.variable_bins(X[:,0], k=10, verbose=False)
            r2_in_bins = reconstruction.stratified_coefficient_of_determination(X[:,0:1], X_rec[:,0:1], idx=idx, use_global_mean=True, verbose=False)
        except Exception:
            self.assertTrue(False)

        try:
            (idx, bins_borders) = preprocess.variable_bins(X[:,0], k=10, verbose=False)
            r2_in_bins = reconstruction.stratified_coefficient_of_determination(X[:,1:2], X_rec[:,1:2], idx=idx, use_global_mean=False, verbose=False)
        except Exception:
            self.assertTrue(False)

# ------------------------------------------------------------------------------

    def test_reconstruction__stratified_coefficient_of_determination__not_allowed_calls(self):

        X = np.random.rand(100,10)
        pca_X = reduction.PCA(X, scaling='auto', n_components=2)
        X_rec = pca_X.reconstruct(pca_X.transform(X))
        (idx, bins_borders) = preprocess.variable_bins(X[:,0], k=10, verbose=False)

        with self.assertRaises(ValueError):
            r2_in_bins = reconstruction.stratified_coefficient_of_determination(X[:,1], X_rec[:,1], -10, use_global_mean=False, verbose=False)

        with self.assertRaises(ValueError):
            r2_in_bins = reconstruction.stratified_coefficient_of_determination(X[:,1], X_rec[:,1], idx=idx, use_global_mean=1, verbose=False)

        with self.assertRaises(ValueError):
            r2_in_bins = reconstruction.stratified_coefficient_of_determination(X[:,1], X_rec[:,1], idx=idx, use_global_mean=False, verbose=1)

        with self.assertRaises(ValueError):
            r2_in_bins = reconstruction.stratified_coefficient_of_determination(X[:,1], X_rec[:,1], '10', use_global_mean=False, verbose=False)

        with self.assertRaises(ValueError):
            r2_in_bins = reconstruction.stratified_coefficient_of_determination(X[:,0:2], X_rec[:,0], idx=idx)

        with self.assertRaises(ValueError):
            r2_in_bins = reconstruction.stratified_coefficient_of_determination(X[:,0], X_rec[:,0:3], idx=idx)

        with self.assertRaises(ValueError):
            r2_in_bins = reconstruction.stratified_coefficient_of_determination(X[:,3:5], X_rec[:,3:5], idx=idx)

        with self.assertRaises(ValueError):
            r2_in_bins = reconstruction.stratified_coefficient_of_determination(X[:,0:2], X_rec[:,0:4], idx=idx, use_global_mean=True, verbose=False)

        with self.assertRaises(ValueError):
            r2_in_bins = reconstruction.stratified_coefficient_of_determination(X[0:10,0], X_rec[:,0], idx=idx)

        with self.assertRaises(ValueError):
            r2_in_bins = reconstruction.stratified_coefficient_of_determination(X[:,0], X_rec[10:50,0], idx=idx)

        with self.assertRaises(ValueError):
            r2_in_bins = reconstruction.stratified_coefficient_of_determination([10,20,30], [1,2,3], idx=idx)

# ------------------------------------------------------------------------------

    def test_reconstruction__stratified_coefficient_of_determination__computation(self):

        pass

# ------------------------------------------------------------------------------
