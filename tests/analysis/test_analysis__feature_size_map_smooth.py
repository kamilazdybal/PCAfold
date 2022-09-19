import unittest
import numpy as np
from PCAfold import preprocess
from PCAfold import reduction
from PCAfold import analysis

class Analysis(unittest.TestCase):

    def test_analysis__feature_size_map_smooth__allowed_calls(self):

        X = np.random.rand(100,10)
        variable_names = ['X_' + str(i) for i in range(0,10)]
        pca_X = reduction.PCA(X, n_components=2)
        principal_components = pca_X.transform(X)
        bandwidth_values = np.logspace(-4, 2, 50)
        variance_data = analysis.compute_normalized_variance(principal_components,
                                                    X,
                                                    depvar_names=variable_names,
                                                    bandwidth_values=bandwidth_values)

        feature_size_map = analysis.feature_size_map(variance_data, variable_name='X_1', cutoff=1, starting_bandwidth_idx='peak', verbose=False)

        try:
            updated_feature_size_map = analysis.feature_size_map_smooth(principal_components, feature_size_map, method='median', n_neighbors=4)
        except:
            self.assertTrue(False)

        try:
            updated_feature_size_map = analysis.feature_size_map_smooth(principal_components, feature_size_map, method='mean', n_neighbors=4)
        except:
            self.assertTrue(False)

        try:
            updated_feature_size_map = analysis.feature_size_map_smooth(principal_components, feature_size_map, method='max', n_neighbors=4)
        except:
            self.assertTrue(False)

        try:
            updated_feature_size_map = analysis.feature_size_map_smooth(principal_components, feature_size_map, method='min', n_neighbors=4)
        except:
            self.assertTrue(False)

        try:
            updated_feature_size_map = analysis.feature_size_map_smooth(principal_components, feature_size_map, method='median', n_neighbors=10)
        except:
            self.assertTrue(False)

# ------------------------------------------------------------------------------

    def test_analysis__feature_size_map_smooth__not_allowed_calls(self):

        X = np.random.rand(100,10)
        variable_names = ['X_' + str(i) for i in range(0,10)]
        pca_X = reduction.PCA(X, n_components=2)
        principal_components = pca_X.transform(X)
        bandwidth_values = np.logspace(-4, 2, 50)
        variance_data = analysis.compute_normalized_variance(principal_components,
                                                    X,
                                                    depvar_names=variable_names,
                                                    bandwidth_values=bandwidth_values)

        feature_size_map = analysis.feature_size_map(variance_data, variable_name='X_1', cutoff=1, starting_bandwidth_idx='peak', verbose=False)

        with self.assertRaises(ValueError):
            updated_feature_size_map = analysis.feature_size_map_smooth([1,2,3], feature_size_map, method='median', n_neighbors=10)

        with self.assertRaises(ValueError):
            updated_feature_size_map = analysis.feature_size_map_smooth(principal_components, [], method='median', n_neighbors=10)

        with self.assertRaises(ValueError):
            updated_feature_size_map = analysis.feature_size_map_smooth(principal_components, feature_size_map, method='average', n_neighbors=10)

        with self.assertRaises(ValueError):
            updated_feature_size_map = analysis.feature_size_map_smooth(principal_components, feature_size_map, method=[], n_neighbors=10)

        with self.assertRaises(ValueError):
            updated_feature_size_map = analysis.feature_size_map_smooth(principal_components, feature_size_map, method='median', n_neighbors=-1)

        with self.assertRaises(ValueError):
            updated_feature_size_map = analysis.feature_size_map_smooth(principal_components, feature_size_map, method='median', n_neighbors=101)

        with self.assertRaises(ValueError):
            updated_feature_size_map = analysis.feature_size_map_smooth(principal_components, feature_size_map, method='median', n_neighbors=[])

# ------------------------------------------------------------------------------

    def test_analysis__feature_size_map_smooth__computation(self):

        pass

# ------------------------------------------------------------------------------
