import unittest
import numpy as np
from PCAfold import preprocess
from PCAfold import reduction
from PCAfold import analysis

class Reduction(unittest.TestCase):

    def test_reduction__PCA_transform__allowed_calls(self):

        test_data_set = np.random.rand(10,2)

        pca = reduction.PCA(test_data_set, scaling='auto')

        try:
            pca.transform(test_data_set)
        except Exception:
            self.assertTrue(False)

        try:
            scores = pca.transform(test_data_set)
        except Exception:
            self.assertTrue(False)

        try:
            x = pca.reconstruct(scores)
        except Exception:
            self.assertTrue(False)

        try:
            scores = pca.transform(test_data_set)
            x = pca.reconstruct(scores)
            difference = abs(test_data_set - x)
            comparison = difference < 10**(-14)
            self.assertTrue(comparison.all())
        except Exception:
            self.assertTrue(False)

# ------------------------------------------------------------------------------

    def test_reduction__PCA_transform__not_allowed_calls(self):

        test_data_set = np.random.rand(10,2)
        test_data_set_2 = np.random.rand(10,3)

        pca = reduction.PCA(test_data_set, scaling='auto')

        with self.assertRaises(ValueError):
            pca.transform(test_data_set_2)

# ------------------------------------------------------------------------------

    def test_reduction__PCA_transform__reconstruct_on_all_available_scalings(self):

        X = np.random.rand(100,10)

        try:
            pca_X = reduction.PCA(X, scaling='none', n_components=0)
            principal_components = pca_X.transform(X)
            X_rec = pca_X.reconstruct(principal_components)
            difference = abs(X - X_rec)
            comparison = difference < 10**(-14)
            self.assertTrue(comparison.all())
        except Exception:
            self.assertTrue(False)

        try:
            pca_X = reduction.PCA(X, scaling='auto', n_components=0)
            principal_components = pca_X.transform(X)
            X_rec = pca_X.reconstruct(principal_components)
            difference = abs(X - X_rec)
            comparison = difference < 10**(-14)
            self.assertTrue(comparison.all())
        except Exception:
            self.assertTrue(False)

        try:
            pca_X = reduction.PCA(X, scaling='range', n_components=0)
            principal_components = pca_X.transform(X)
            X_rec = pca_X.reconstruct(principal_components)
            difference = abs(X - X_rec)
            comparison = difference < 10**(-14)
            self.assertTrue(comparison.all())
        except Exception:
            self.assertTrue(False)

        try:
            pca_X = reduction.PCA(X, scaling='vast', n_components=0)
            principal_components = pca_X.transform(X)
            X_rec = pca_X.reconstruct(principal_components)
            difference = abs(X - X_rec)
            comparison = difference < 10**(-14)
            self.assertTrue(comparison.all())
        except Exception:
            self.assertTrue(False)

        try:
            pca_X = reduction.PCA(X, scaling='pareto', n_components=0)
            principal_components = pca_X.transform(X)
            X_rec = pca_X.reconstruct(principal_components)
            difference = abs(X - X_rec)
            comparison = difference < 10**(-14)
            self.assertTrue(comparison.all())
        except Exception:
            self.assertTrue(False)

        try:
            pca_X = reduction.PCA(X, scaling='max', n_components=0)
            principal_components = pca_X.transform(X)
            X_rec = pca_X.reconstruct(principal_components)
            difference = abs(X - X_rec)
            comparison = difference < 10**(-14)
            self.assertTrue(comparison.all())
        except Exception:
            self.assertTrue(False)

        try:
            pca_X = reduction.PCA(X, scaling='level', n_components=0)
            principal_components = pca_X.transform(X)
            X_rec = pca_X.reconstruct(principal_components)
            difference = abs(X - X_rec)
            comparison = difference < 10**(-14)
            self.assertTrue(comparison.all())
        except Exception:
            self.assertTrue(False)

        try:
            pca_X = reduction.PCA(X, scaling='0to1', n_components=0)
            principal_components = pca_X.transform(X)
            X_rec = pca_X.reconstruct(principal_components)
            difference = abs(X - X_rec)
            comparison = difference < 10**(-14)
            self.assertTrue(comparison.all())
        except Exception:
            self.assertTrue(False)

        try:
            pca_X = reduction.PCA(X, scaling='-1to1', n_components=0)
            principal_components = pca_X.transform(X)
            X_rec = pca_X.reconstruct(principal_components)
            difference = abs(X - X_rec)
            comparison = difference < 10**(-14)
            self.assertTrue(comparison.all())
        except Exception:
            self.assertTrue(False)

        try:
            pca_X = reduction.PCA(X, scaling='poisson', n_components=0)
            principal_components = pca_X.transform(X)
            X_rec = pca_X.reconstruct(principal_components)
            difference = abs(X - X_rec)
            comparison = difference < 10**(-14)
            self.assertTrue(comparison.all())
        except Exception:
            self.assertTrue(False)

        try:
            pca_X = reduction.PCA(X, scaling='vast_2', n_components=0)
            principal_components = pca_X.transform(X)
            X_rec = pca_X.reconstruct(principal_components)
            difference = abs(X - X_rec)
            comparison = difference < 10**(-14)
            self.assertTrue(comparison.all())
        except Exception:
            self.assertTrue(False)

        try:
            pca_X = reduction.PCA(X, scaling='vast_3', n_components=0)
            principal_components = pca_X.transform(X)
            X_rec = pca_X.reconstruct(principal_components)
            difference = abs(X - X_rec)
            comparison = difference < 10**(-14)
            self.assertTrue(comparison.all())
        except Exception:
            self.assertTrue(False)

        try:
            pca_X = reduction.PCA(X, scaling='vast_4', n_components=0)
            principal_components = pca_X.transform(X)
            X_rec = pca_X.reconstruct(principal_components)
            difference = abs(X - X_rec)
            comparison = difference < 10**(-14)
            self.assertTrue(comparison.all())
        except Exception:
            self.assertTrue(False)

# ------------------------------------------------------------------------------

    def test_reduction__PCA_transform__reconstruct_on_all_available_scalings_with_no_centering(self):

        X = np.random.rand(100,10)

        try:
            pca_X = reduction.PCA(X, scaling='none', n_components=0)
            principal_components = pca_X.transform(X, nocenter=True)
            X_rec = pca_X.reconstruct(principal_components, nocenter=True)
            difference = abs(X - X_rec)
            comparison = difference < 10**(-14)
            self.assertTrue(comparison.all())
        except Exception:
            self.assertTrue(False)

        try:
            pca_X = reduction.PCA(X, scaling='auto', n_components=0)
            principal_components = pca_X.transform(X, nocenter=True)
            X_rec = pca_X.reconstruct(principal_components, nocenter=True)
            difference = abs(X - X_rec)
            comparison = difference < 10**(-14)
            self.assertTrue(comparison.all())
        except Exception:
            self.assertTrue(False)

        try:
            pca_X = reduction.PCA(X, scaling='range', n_components=0)
            principal_components = pca_X.transform(X, nocenter=True)
            X_rec = pca_X.reconstruct(principal_components, nocenter=True)
            difference = abs(X - X_rec)
            comparison = difference < 10**(-14)
            self.assertTrue(comparison.all())
        except Exception:
            self.assertTrue(False)

        try:
            pca_X = reduction.PCA(X, scaling='vast', n_components=0)
            principal_components = pca_X.transform(X, nocenter=True)
            X_rec = pca_X.reconstruct(principal_components, nocenter=True)
            difference = abs(X - X_rec)
            comparison = difference < 10**(-14)
            self.assertTrue(comparison.all())
        except Exception:
            self.assertTrue(False)

        try:
            pca_X = reduction.PCA(X, scaling='pareto', n_components=0)
            principal_components = pca_X.transform(X, nocenter=True)
            X_rec = pca_X.reconstruct(principal_components, nocenter=True)
            difference = abs(X - X_rec)
            comparison = difference < 10**(-14)
            self.assertTrue(comparison.all())
        except Exception:
            self.assertTrue(False)

        try:
            pca_X = reduction.PCA(X, scaling='max', n_components=0)
            principal_components = pca_X.transform(X, nocenter=True)
            X_rec = pca_X.reconstruct(principal_components, nocenter=True)
            difference = abs(X - X_rec)
            comparison = difference < 10**(-14)
            self.assertTrue(comparison.all())
        except Exception:
            self.assertTrue(False)

        try:
            pca_X = reduction.PCA(X, scaling='level', n_components=0)
            principal_components = pca_X.transform(X, nocenter=True)
            X_rec = pca_X.reconstruct(principal_components, nocenter=True)
            difference = abs(X - X_rec)
            comparison = difference < 10**(-14)
            self.assertTrue(comparison.all())
        except Exception:
            self.assertTrue(False)

        try:
            pca_X = reduction.PCA(X, scaling='0to1', n_components=0)
            principal_components = pca_X.transform(X, nocenter=True)
            X_rec = pca_X.reconstruct(principal_components, nocenter=True)
            difference = abs(X - X_rec)
            comparison = difference < 10**(-14)
            self.assertTrue(comparison.all())
        except Exception:
            self.assertTrue(False)

        try:
            pca_X = reduction.PCA(X, scaling='-1to1', n_components=0)
            principal_components = pca_X.transform(X, nocenter=True)
            X_rec = pca_X.reconstruct(principal_components, nocenter=True)
            difference = abs(X - X_rec)
            comparison = difference < 10**(-14)
            self.assertTrue(comparison.all())
        except Exception:
            self.assertTrue(False)

        try:
            pca_X = reduction.PCA(X, scaling='poisson', n_components=0)
            principal_components = pca_X.transform(X, nocenter=True)
            X_rec = pca_X.reconstruct(principal_components, nocenter=True)
            difference = abs(X - X_rec)
            comparison = difference < 10**(-14)
            self.assertTrue(comparison.all())
        except Exception:
            self.assertTrue(False)

        try:
            pca_X = reduction.PCA(X, scaling='vast_2', n_components=0)
            principal_components = pca_X.transform(X, nocenter=True)
            X_rec = pca_X.reconstruct(principal_components, nocenter=True)
            difference = abs(X - X_rec)
            comparison = difference < 10**(-14)
            self.assertTrue(comparison.all())
        except Exception:
            self.assertTrue(False)

        try:
            pca_X = reduction.PCA(X, scaling='vast_3', n_components=0)
            principal_components = pca_X.transform(X, nocenter=True)
            X_rec = pca_X.reconstruct(principal_components, nocenter=True)
            difference = abs(X - X_rec)
            comparison = difference < 10**(-14)
            self.assertTrue(comparison.all())
        except Exception:
            self.assertTrue(False)

        try:
            pca_X = reduction.PCA(X, scaling='vast_4', n_components=0)
            principal_components = pca_X.transform(X, nocenter=True)
            X_rec = pca_X.reconstruct(principal_components, nocenter=True)
            difference = abs(X - X_rec)
            comparison = difference < 10**(-14)
            self.assertTrue(comparison.all())
        except Exception:
            self.assertTrue(False)

# ------------------------------------------------------------------------------

    def test_reduction__PCA_transform__reconstruct_on_all_available_scalings_using_different_X(self):

        X_init = np.random.rand(100,10)
        X = np.random.rand(60,10)

        try:
            pca_X = reduction.PCA(X_init, scaling='none', n_components=0)
            principal_components = pca_X.transform(X)
            X_rec = pca_X.reconstruct(principal_components)
            difference = abs(X - X_rec)
            comparison = difference < 10**(-14)
            self.assertTrue(comparison.all())
        except Exception:
            self.assertTrue(False)

        try:
            pca_X = reduction.PCA(X_init, scaling='auto', n_components=0)
            principal_components = pca_X.transform(X)
            X_rec = pca_X.reconstruct(principal_components)
            difference = abs(X - X_rec)
            comparison = difference < 10**(-14)
            self.assertTrue(comparison.all())
        except Exception:
            self.assertTrue(False)

        try:
            pca_X = reduction.PCA(X_init, scaling='range', n_components=0)
            principal_components = pca_X.transform(X)
            X_rec = pca_X.reconstruct(principal_components)
            difference = abs(X - X_rec)
            comparison = difference < 10**(-14)
            self.assertTrue(comparison.all())
        except Exception:
            self.assertTrue(False)

        try:
            pca_X = reduction.PCA(X_init, scaling='vast', n_components=0)
            principal_components = pca_X.transform(X)
            X_rec = pca_X.reconstruct(principal_components)
            difference = abs(X - X_rec)
            comparison = difference < 10**(-14)
            self.assertTrue(comparison.all())
        except Exception:
            self.assertTrue(False)

        try:
            pca_X = reduction.PCA(X_init, scaling='pareto', n_components=0)
            principal_components = pca_X.transform(X)
            X_rec = pca_X.reconstruct(principal_components)
            difference = abs(X - X_rec)
            comparison = difference < 10**(-14)
            self.assertTrue(comparison.all())
        except Exception:
            self.assertTrue(False)

        try:
            pca_X = reduction.PCA(X_init, scaling='max', n_components=0)
            principal_components = pca_X.transform(X)
            X_rec = pca_X.reconstruct(principal_components)
            difference = abs(X - X_rec)
            comparison = difference < 10**(-14)
            self.assertTrue(comparison.all())
        except Exception:
            self.assertTrue(False)

        try:
            pca_X = reduction.PCA(X_init, scaling='level', n_components=0)
            principal_components = pca_X.transform(X)
            X_rec = pca_X.reconstruct(principal_components)
            difference = abs(X - X_rec)
            comparison = difference < 10**(-14)
            self.assertTrue(comparison.all())
        except Exception:
            self.assertTrue(False)

        try:
            pca_X = reduction.PCA(X, scaling='0to1', n_components=0)
            principal_components = pca_X.transform(X)
            X_rec = pca_X.reconstruct(principal_components)
            difference = abs(X - X_rec)
            comparison = difference < 10**(-14)
            self.assertTrue(comparison.all())
        except Exception:
            self.assertTrue(False)

        try:
            pca_X = reduction.PCA(X, scaling='-1to1', n_components=0)
            principal_components = pca_X.transform(X)
            X_rec = pca_X.reconstruct(principal_components)
            difference = abs(X - X_rec)
            comparison = difference < 10**(-14)
            self.assertTrue(comparison.all())
        except Exception:
            self.assertTrue(False)

        try:
            pca_X = reduction.PCA(X_init, scaling='poisson', n_components=0)
            principal_components = pca_X.transform(X)
            X_rec = pca_X.reconstruct(principal_components)
            difference = abs(X - X_rec)
            comparison = difference < 10**(-14)
            self.assertTrue(comparison.all())
        except Exception:
            self.assertTrue(False)

        try:
            pca_X = reduction.PCA(X_init, scaling='vast_2', n_components=0)
            principal_components = pca_X.transform(X)
            X_rec = pca_X.reconstruct(principal_components)
            difference = abs(X - X_rec)
            comparison = difference < 10**(-14)
            self.assertTrue(comparison.all())
        except Exception:
            self.assertTrue(False)

        try:
            pca_X = reduction.PCA(X_init, scaling='vast_3', n_components=0)
            principal_components = pca_X.transform(X)
            X_rec = pca_X.reconstruct(principal_components)
            difference = abs(X - X_rec)
            comparison = difference < 10**(-14)
            self.assertTrue(comparison.all())
        except Exception:
            self.assertTrue(False)

        try:
            pca_X = reduction.PCA(X_init, scaling='vast_4', n_components=0)
            principal_components = pca_X.transform(X)
            X_rec = pca_X.reconstruct(principal_components)
            difference = abs(X - X_rec)
            comparison = difference < 10**(-14)
            self.assertTrue(comparison.all())
        except Exception:
            self.assertTrue(False)

# ------------------------------------------------------------------------------

    def test_reduction__PCA_transform__reconstruct_on_all_available_scalings_using_different_X_with_no_centering(self):

        X_init = np.random.rand(100,10)
        X = np.random.rand(60,10)

        try:
            pca_X = reduction.PCA(X_init, scaling='none', n_components=0)
            principal_components = pca_X.transform(X, nocenter=True)
            X_rec = pca_X.reconstruct(principal_components, nocenter=True)
            difference = abs(X - X_rec)
            comparison = difference < 10**(-14)
            self.assertTrue(comparison.all())
        except Exception:
            self.assertTrue(False)

        try:
            pca_X = reduction.PCA(X_init, scaling='auto', n_components=0)
            principal_components = pca_X.transform(X, nocenter=True)
            X_rec = pca_X.reconstruct(principal_components, nocenter=True)
            difference = abs(X - X_rec)
            comparison = difference < 10**(-14)
            self.assertTrue(comparison.all())
        except Exception:
            self.assertTrue(False)

        try:
            pca_X = reduction.PCA(X_init, scaling='range', n_components=0)
            principal_components = pca_X.transform(X, nocenter=True)
            X_rec = pca_X.reconstruct(principal_components, nocenter=True)
            difference = abs(X - X_rec)
            comparison = difference < 10**(-14)
            self.assertTrue(comparison.all())
        except Exception:
            self.assertTrue(False)

        try:
            pca_X = reduction.PCA(X_init, scaling='vast', n_components=0)
            principal_components = pca_X.transform(X, nocenter=True)
            X_rec = pca_X.reconstruct(principal_components, nocenter=True)
            difference = abs(X - X_rec)
            comparison = difference < 10**(-14)
            self.assertTrue(comparison.all())
        except Exception:
            self.assertTrue(False)

        try:
            pca_X = reduction.PCA(X_init, scaling='pareto', n_components=0)
            principal_components = pca_X.transform(X, nocenter=True)
            X_rec = pca_X.reconstruct(principal_components, nocenter=True)
            difference = abs(X - X_rec)
            comparison = difference < 10**(-14)
            self.assertTrue(comparison.all())
        except Exception:
            self.assertTrue(False)

        try:
            pca_X = reduction.PCA(X_init, scaling='max', n_components=0)
            principal_components = pca_X.transform(X, nocenter=True)
            X_rec = pca_X.reconstruct(principal_components, nocenter=True)
            difference = abs(X - X_rec)
            comparison = difference < 10**(-14)
            self.assertTrue(comparison.all())
        except Exception:
            self.assertTrue(False)

        try:
            pca_X = reduction.PCA(X_init, scaling='level', n_components=0)
            principal_components = pca_X.transform(X, nocenter=True)
            X_rec = pca_X.reconstruct(principal_components, nocenter=True)
            difference = abs(X - X_rec)
            comparison = difference < 10**(-14)
            self.assertTrue(comparison.all())
        except Exception:
            self.assertTrue(False)

        try:
            pca_X = reduction.PCA(X, scaling='0to1', n_components=0)
            principal_components = pca_X.transform(X, nocenter=True)
            X_rec = pca_X.reconstruct(principal_components, nocenter=True)
            difference = abs(X - X_rec)
            comparison = difference < 10**(-14)
            self.assertTrue(comparison.all())
        except Exception:
            self.assertTrue(False)

        try:
            pca_X = reduction.PCA(X, scaling='-1to1', n_components=0)
            principal_components = pca_X.transform(X, nocenter=True)
            X_rec = pca_X.reconstruct(principal_components, nocenter=True)
            difference = abs(X - X_rec)
            comparison = difference < 10**(-14)
            self.assertTrue(comparison.all())
        except Exception:
            self.assertTrue(False)

        try:
            pca_X = reduction.PCA(X_init, scaling='poisson', n_components=0)
            principal_components = pca_X.transform(X, nocenter=True)
            X_rec = pca_X.reconstruct(principal_components, nocenter=True)
            difference = abs(X - X_rec)
            comparison = difference < 10**(-14)
            self.assertTrue(comparison.all())
        except Exception:
            self.assertTrue(False)

        try:
            pca_X = reduction.PCA(X_init, scaling='vast_2', n_components=0)
            principal_components = pca_X.transform(X)
            X_rec = pca_X.reconstruct(principal_components)
            difference = abs(X - X_rec)
            comparison = difference < 10**(-14)
            self.assertTrue(comparison.all())
        except Exception:
            self.assertTrue(False)

        try:
            pca_X = reduction.PCA(X_init, scaling='vast_3', n_components=0)
            principal_components = pca_X.transform(X, nocenter=True)
            X_rec = pca_X.reconstruct(principal_components, nocenter=True)
            difference = abs(X - X_rec)
            comparison = difference < 10**(-14)
            self.assertTrue(comparison.all())
        except Exception:
            self.assertTrue(False)

        try:
            pca_X = reduction.PCA(X_init, scaling='vast_4', n_components=0)
            principal_components = pca_X.transform(X, nocenter=True)
            X_rec = pca_X.reconstruct(principal_components, nocenter=True)
            difference = abs(X - X_rec)
            comparison = difference < 10**(-14)
            self.assertTrue(comparison.all())
        except Exception:
            self.assertTrue(False)

# ------------------------------------------------------------------------------
