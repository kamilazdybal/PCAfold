import unittest
import numpy as np
from PCAfold import preprocess
from PCAfold import reduction
from PCAfold import analysis

class Analysis(unittest.TestCase):

    def test_analysis__average_knn_distance__allowed_calls(self):

        X = np.random.rand(100,20)
        pca_X = reduction.PCA(X, scaling='none', n_components=2, use_eigendec=True, nocenter=False)
        principal_components = pca_X.transform(X)

        try:
            average_distances = analysis.average_knn_distance(principal_components)
        except:
            self.assertTrue(False)

        try:
            average_distances = analysis.average_knn_distance(principal_components, n_neighbors=5, verbose=False)
        except:
            self.assertTrue(False)

# ------------------------------------------------------------------------------

    def test_analysis__average_knn_distance__not_allowed_calls(self):

        X = np.random.rand(100,20)
        pca_X = reduction.PCA(X, scaling='none', n_components=2, use_eigendec=True, nocenter=False)
        principal_components = pca_X.transform(X)

        with self.assertRaises(ValueError):
            average_distances = analysis.average_knn_distance([1,2,3,4,5])

        with self.assertRaises(ValueError):
            average_distances = analysis.average_knn_distance(1)

        with self.assertRaises(ValueError):
            average_distances = analysis.average_knn_distance(principal_components, n_neighbors=[1])

        with self.assertRaises(ValueError):
            average_distances = analysis.average_knn_distance(principal_components, n_neighbors=2.1)

        with self.assertRaises(ValueError):
            average_distances = analysis.average_knn_distance(principal_components, n_neighbors=1)

        with self.assertRaises(ValueError):
            average_distances = analysis.average_knn_distance(principal_components, verbose=[1])

# ------------------------------------------------------------------------------

    def test_analysis__average_knn_distance__computation(self):

        try:
            X = np.ones((100,2))
            average_distances = analysis.average_knn_distance(X, n_neighbors=2, verbose=False)
            self.assertTrue(np.all(average_distances==0))
        except:
            self.assertTrue(False)

        try:
            X = np.ones((10,2))
            X[0,:] = 2*X[0,:]
            average_distances = analysis.average_knn_distance(X, n_neighbors=2, verbose=False)
            self.assertTrue(np.all(average_distances[1::]==0))
            self.assertTrue(average_distances[0]==np.sqrt(2))
        except:
            self.assertTrue(False)

        try:
            X = np.ones((10,2))
            X[0,:] = 2*X[0,:]
            X[1,:] = 1.5*X[1,:]
            average_distances = analysis.average_knn_distance(X, n_neighbors=2, verbose=False)
            expected_result = (0.5*np.sqrt(2) + np.sqrt(2))/2
            self.assertTrue(average_distances[0]==expected_result)
        except:
            self.assertTrue(False)

        try:
            X = np.ones((10,2))
            X[0,:] = 2*X[0,:]
            X[1,:] = 1.5*X[1,:]
            average_distances = analysis.average_knn_distance(X, n_neighbors=3, verbose=False)
            expected_result = (0.5*np.sqrt(2) + np.sqrt(2) + np.sqrt(2))/3
            self.assertTrue(average_distances[0]==expected_result)
        except:
            self.assertTrue(False)

# ------------------------------------------------------------------------------
