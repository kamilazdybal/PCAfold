import unittest
import numpy as np
from PCAfold import preprocess
from PCAfold import reduction
from PCAfold import analysis

class Analysis(unittest.TestCase):

    def test_analysis__feature_size_map__allowed_calls(self):

        X = np.random.rand(100,10)
        variable_names = ['X_' + str(i) for i in range(0,10)]
        pca_X = reduction.PCA(X, n_components=2)
        principal_components = pca_X.transform(X)
        bandwidth_values = np.logspace(-4, 2, 50)
        variance_data = analysis.compute_normalized_variance(principal_components,
                                                    X,
                                                    depvar_names=variable_names,
                                                    bandwidth_values=bandwidth_values)

        try:
            feature_size_map = analysis.feature_size_map(variance_data, variable_name='X_1', cutoff=1, starting_bandwidth_idx='peak', verbose=False)
        except:
            self.assertTrue(False)

        try:
            feature_size_map = analysis.feature_size_map(variance_data, variable_name='X_2', cutoff=1, starting_bandwidth_idx='peak', verbose=False)
        except:
            self.assertTrue(False)

        try:
            feature_size_map = analysis.feature_size_map(variance_data, variable_name='X_2', cutoff=1, starting_bandwidth_idx=40, verbose=False)
        except:
            self.assertTrue(False)

        try:
            feature_size_map = analysis.feature_size_map(variance_data, variable_name='X_1', cutoff=10.5, starting_bandwidth_idx=20, verbose=False)
        except:
            self.assertTrue(False)

# ------------------------------------------------------------------------------

    def test_analysis__feature_size_map__not_allowed_calls(self):

        X = np.random.rand(100,10)
        variable_names = ['X_' + str(i) for i in range(0,10)]
        pca_X = reduction.PCA(X, n_components=2)
        principal_components = pca_X.transform(X)
        bandwidth_values = np.logspace(-4, 2, 50)
        variance_data = analysis.compute_normalized_variance(principal_components,
                                                    X,
                                                    depvar_names=variable_names,
                                                    bandwidth_values=bandwidth_values)

        # Wrong input type:
        with self.assertRaises(ValueError):
            feature_size_map = analysis.feature_size_map([1,2,3], variable_name='X_1', cutoff=1, starting_bandwidth_idx='peak', verbose=False)

        with self.assertRaises(ValueError):
            feature_size_map = analysis.feature_size_map(variance_data, variable_name=[], cutoff=1, starting_bandwidth_idx='peak', verbose=False)

        with self.assertRaises(ValueError):
            feature_size_map = analysis.feature_size_map(variance_data, variable_name='X_1', cutoff=-1, starting_bandwidth_idx='peak', verbose=False)

        with self.assertRaises(ValueError):
            feature_size_map = analysis.feature_size_map(variance_data, variable_name='X_1', cutoff=101, starting_bandwidth_idx='peak', verbose=False)

        with self.assertRaises(ValueError):
            feature_size_map = analysis.feature_size_map(variance_data, variable_name='X_1', cutoff=1, starting_bandwidth_idx=[], verbose=False)

        with self.assertRaises(ValueError):
            feature_size_map = analysis.feature_size_map(variance_data, variable_name='X_1', cutoff=1, starting_bandwidth_idx='peak', verbose=[])

        # Reference to a non-existent input:
        with self.assertRaises(KeyError):
            feature_size_map = analysis.feature_size_map(variance_data, variable_name='var', cutoff=1, starting_bandwidth_idx='peak', verbose=False)

        with self.assertRaises(IndexError):
            feature_size_map = analysis.feature_size_map(variance_data, variable_name='X_1', cutoff=1, starting_bandwidth_idx=100, verbose=False)

# ------------------------------------------------------------------------------

    def test_analysis__feature_size_map__computation(self):

        pass

# ------------------------------------------------------------------------------
