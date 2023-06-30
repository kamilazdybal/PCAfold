import unittest
import numpy as np
from PCAfold import preprocess
from PCAfold import reduction
from PCAfold import analysis

class Analysis(unittest.TestCase):

    def test_preprocess__DensityEstimation_average_knn_distance__allowed_calls(self):

        X = np.random.rand(100,20)
        pca_X = reduction.PCA(X, scaling='none', n_components=2, use_eigendec=True, nocenter=False)
        principal_components = pca_X.transform(X)

        density_estimation = preprocess.DensityEstimation(principal_components, n_neighbors=5)

        try:
            average_distances = density_estimation.average_knn_distance(verbose=False)
        except:
            self.assertTrue(False)

# ------------------------------------------------------------------------------

    def test_preprocess__DensityEstimation_average_knn_distance__not_allowed_calls(self):

        X = np.random.rand(100,20)
        pca_X = reduction.PCA(X, scaling='none', n_components=2, use_eigendec=True, nocenter=False)
        principal_components = pca_X.transform(X)

        density_estimation = preprocess.DensityEstimation(principal_components, n_neighbors=5)

        with self.assertRaises(ValueError):
            average_distances = density_estimation.average_knn_distance(verbose=[1])

# ------------------------------------------------------------------------------

    def test_preprocess__DensityEstimation_average_knn_distance__computation(self):

        try:
            X = np.ones((100,2))
            density_estimation = preprocess.DensityEstimation(X, n_neighbors=2)
            average_distances = density_estimation.average_knn_distance(verbose=False)
            self.assertTrue(np.all(average_distances==0))
        except:
            self.assertTrue(False)

        try:
            X = np.ones((10,2))
            X[0,:] = 2*X[0,:]
            density_estimation = preprocess.DensityEstimation(X, n_neighbors=2)
            average_distances = density_estimation.average_knn_distance(verbose=False)
            self.assertTrue(np.all(average_distances[1::]==0))
            self.assertTrue(average_distances[0]==np.sqrt(2))
        except:
            self.assertTrue(False)

        try:
            X = np.ones((10,2))
            X[0,:] = 2*X[0,:]
            X[1,:] = 1.5*X[1,:]
            density_estimation = preprocess.DensityEstimation(X, n_neighbors=2)
            average_distances = density_estimation.average_knn_distance(verbose=False)
            expected_result = (0.5*np.sqrt(2) + np.sqrt(2))/2
            self.assertTrue(average_distances[0]==expected_result)
        except:
            self.assertTrue(False)

        try:
            X = np.ones((10,2))
            X[0,:] = 2*X[0,:]
            X[1,:] = 1.5*X[1,:]
            density_estimation = preprocess.DensityEstimation(X, n_neighbors=3)
            average_distances = density_estimation.average_knn_distance(verbose=False)
            expected_result = (0.5*np.sqrt(2) + np.sqrt(2) + np.sqrt(2))/3
            self.assertTrue(average_distances[0]==expected_result)
        except:
            self.assertTrue(False)

# ------------------------------------------------------------------------------
