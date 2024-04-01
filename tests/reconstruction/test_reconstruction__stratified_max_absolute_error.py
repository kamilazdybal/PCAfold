import unittest
import numpy as np
from PCAfold import preprocess
from PCAfold import reduction
from PCAfold import reconstruction

class Reconstruction(unittest.TestCase):

    def test_reconstruction__stratified_mean_squared_error__allowed_calls(self):

        X = np.random.rand(100,10)
        pca_X = reduction.PCA(X, scaling='auto', n_components=2)
        X_rec = pca_X.reconstruct(pca_X.transform(X))
        (idx, bins_borders) = preprocess.variable_bins(X[:,0], k=10, verbose=False)

        maxae_in_bins = reconstruction.stratified_max_absolute_error(X[:,0],
                                                                     X_rec[:,0],
                                                                     idx=idx,
                                                                     verbose=True)

# ------------------------------------------------------------------------------

    def test_reconstruction__stratified_mean_squared_error__not_allowed_calls(self):

        X = np.random.rand(100,10)
        pca_X = reduction.PCA(X, scaling='auto', n_components=2)
        X_rec = pca_X.reconstruct(pca_X.transform(X))
        (idx, bins_borders) = preprocess.variable_bins(X[:,0], k=10, verbose=False)

        with self.assertRaises(ValueError):
            maxae_in_bins = reconstruction.stratified_max_absolute_error(X[:,0:2],
                                                                         X_rec[:,0],
                                                                         idx=idx,
                                                                         verbose=True)

        with self.assertRaises(ValueError):
            maxae_in_bins = reconstruction.stratified_max_absolute_error([],
                                                                         X_rec[:,0],
                                                                         idx=idx,
                                                                         verbose=True)

        with self.assertRaises(ValueError):
            maxae_in_bins = reconstruction.stratified_max_absolute_error(X[:,0],
                                                                         X_rec[:,0:2],
                                                                         idx=idx,
                                                                         verbose=True)

        with self.assertRaises(ValueError):
            maxae_in_bins = reconstruction.stratified_max_absolute_error(X[:,0],
                                                                         [],
                                                                         idx=idx,
                                                                         verbose=True)

        with self.assertRaises(ValueError):
            maxae_in_bins = reconstruction.stratified_max_absolute_error(X[:,0],
                                                                         X_rec[:,0],
                                                                         idx=[],
                                                                         verbose=True)

        with self.assertRaises(ValueError):
            maxae_in_bins = reconstruction.stratified_max_absolute_error(X[:,0],
                                                                         X_rec[:,0],
                                                                         idx=idx,
                                                                         verbose=[])

# ------------------------------------------------------------------------------

    def test_reconstruction__stratified_mean_squared_error__computation(self):

        pass

# ------------------------------------------------------------------------------
