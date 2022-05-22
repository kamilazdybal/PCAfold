import unittest
import numpy as np
from PCAfold import preprocess
from PCAfold import reduction
from PCAfold import analysis

class Reduction(unittest.TestCase):

    def test_reduction__PCA_calculate_r2__allowed_calls(self):

        test_data_set = np.random.rand(100,20)
        r2_test = np.ones((20,))

        try:
            pca_X = reduction.PCA(test_data_set, scaling='auto', n_components=20, use_eigendec=True, nocenter=False)
            r2_values = pca_X.calculate_r2(test_data_set)
            comparison = r2_values == r2_test
            self.assertTrue(comparison.all())
        except Exception:
            self.assertTrue(False)

# ------------------------------------------------------------------------------

    def test_reduction__PCA_calculate_r2__all_available_scalings(self):

        test_data_set = np.random.rand(1000,20)
        r2_test = np.ones((20,))

        try:
            pca_X = reduction.PCA(test_data_set, scaling='none', n_components=20, use_eigendec=True, nocenter=False)
            r2_values = pca_X.calculate_r2(test_data_set)
            self.assertTrue(np.allclose(r2_values, r2_test))
        except Exception:
            self.assertTrue(False)

        try:
            pca_X = reduction.PCA(test_data_set, scaling='auto', n_components=20, use_eigendec=True, nocenter=False)
            r2_values = pca_X.calculate_r2(test_data_set)
            self.assertTrue(np.allclose(r2_values, r2_test))
        except Exception:
            self.assertTrue(False)

        try:
            pca_X = reduction.PCA(test_data_set, scaling='pareto', n_components=20, use_eigendec=True, nocenter=False)
            r2_values = pca_X.calculate_r2(test_data_set)
            self.assertTrue(np.allclose(r2_values, r2_test))
        except Exception:
            self.assertTrue(False)

        try:
            pca_X = reduction.PCA(test_data_set, scaling='vast', n_components=20, use_eigendec=True, nocenter=False)
            r2_values = pca_X.calculate_r2(test_data_set)
            self.assertTrue(np.allclose(r2_values, r2_test))
        except Exception:
            self.assertTrue(False)

        try:
            pca_X = reduction.PCA(test_data_set, scaling='range', n_components=20, use_eigendec=True, nocenter=False)
            r2_values = pca_X.calculate_r2(test_data_set)
            self.assertTrue(np.allclose(r2_values, r2_test))
        except Exception:
            self.assertTrue(False)

        try:
            pca_X = reduction.PCA(test_data_set, scaling='0to1', n_components=20, use_eigendec=True, nocenter=False)
            r2_values = pca_X.calculate_r2(test_data_set)
            self.assertTrue(np.allclose(r2_values, r2_test))
        except Exception:
            self.assertTrue(False)

        try:
            pca_X = reduction.PCA(test_data_set, scaling='-1to1', n_components=20, use_eigendec=True, nocenter=False)
            r2_values = pca_X.calculate_r2(test_data_set)
            self.assertTrue(np.allclose(r2_values, r2_test))
        except Exception:
            self.assertTrue(False)

        try:
            pca_X = reduction.PCA(test_data_set, scaling='level', n_components=20, use_eigendec=True, nocenter=False)
            r2_values = pca_X.calculate_r2(test_data_set)
            self.assertTrue(np.allclose(r2_values, r2_test))
        except Exception:
            self.assertTrue(False)

        try:
            pca_X = reduction.PCA(test_data_set, scaling='max', n_components=20, use_eigendec=True, nocenter=False)
            r2_values = pca_X.calculate_r2(test_data_set)
            self.assertTrue(np.allclose(r2_values, r2_test))
        except Exception:
            self.assertTrue(False)

        try:
            pca_X = reduction.PCA(test_data_set, scaling='poisson', n_components=20, use_eigendec=True, nocenter=False)
            r2_values = pca_X.calculate_r2(test_data_set)
            self.assertTrue(np.allclose(r2_values, r2_test))
        except Exception:
            self.assertTrue(False)

        try:
            pca_X = reduction.PCA(test_data_set, scaling='vast_2', n_components=20, use_eigendec=True, nocenter=False)
            r2_values = pca_X.calculate_r2(test_data_set)
            self.assertTrue(np.allclose(r2_values, r2_test))
        except Exception:
            self.assertTrue(False)

        try:
            pca_X = reduction.PCA(test_data_set, scaling='vast_3', n_components=20, use_eigendec=True, nocenter=False)
            r2_values = pca_X.calculate_r2(test_data_set)
            self.assertTrue(np.allclose(r2_values, r2_test))
        except Exception:
            self.assertTrue(False)

        try:
            pca_X = reduction.PCA(test_data_set, scaling='vast_4', n_components=20, use_eigendec=True, nocenter=False)
            r2_values = pca_X.calculate_r2(test_data_set)
            self.assertTrue(np.allclose(r2_values, r2_test))
        except Exception:
            self.assertTrue(False)

# ------------------------------------------------------------------------------

    def test_reduction__PCA_calculate_r2__all_available_scalings_with_no_centering(self):

        test_data_set = np.random.rand(1000,20)
        r2_test = np.ones((20,))

        try:
            pca_X = reduction.PCA(test_data_set, scaling='none', n_components=20, use_eigendec=True, nocenter=True)
            r2_values = pca_X.calculate_r2(test_data_set)
            self.assertTrue(np.allclose(r2_values, r2_test))
        except Exception:
            self.assertTrue(False)

        try:
            pca_X = reduction.PCA(test_data_set, scaling='auto', n_components=20, use_eigendec=True, nocenter=True)
            r2_values = pca_X.calculate_r2(test_data_set)
            self.assertTrue(np.allclose(r2_values, r2_test))
        except Exception:
            self.assertTrue(False)

        try:
            pca_X = reduction.PCA(test_data_set, scaling='pareto', n_components=20, use_eigendec=True, nocenter=True)
            r2_values = pca_X.calculate_r2(test_data_set)
            self.assertTrue(np.allclose(r2_values, r2_test))
        except Exception:
            self.assertTrue(False)

        try:
            pca_X = reduction.PCA(test_data_set, scaling='vast', n_components=20, use_eigendec=True, nocenter=True)
            r2_values = pca_X.calculate_r2(test_data_set)
            self.assertTrue(np.allclose(r2_values, r2_test))
        except Exception:
            self.assertTrue(False)

        try:
            pca_X = reduction.PCA(test_data_set, scaling='range', n_components=20, use_eigendec=True, nocenter=True)
            r2_values = pca_X.calculate_r2(test_data_set)
            self.assertTrue(np.allclose(r2_values, r2_test))
        except Exception:
            self.assertTrue(False)

        try:
            pca_X = reduction.PCA(test_data_set, scaling='0to1', n_components=20, use_eigendec=True, nocenter=True)
            r2_values = pca_X.calculate_r2(test_data_set)
            self.assertTrue(np.allclose(r2_values, r2_test))
        except Exception:
            self.assertTrue(False)

        try:
            pca_X = reduction.PCA(test_data_set, scaling='-1to1', n_components=20, use_eigendec=True, nocenter=True)
            r2_values = pca_X.calculate_r2(test_data_set)
            self.assertTrue(np.allclose(r2_values, r2_test))
        except Exception:
            self.assertTrue(False)

        try:
            pca_X = reduction.PCA(test_data_set, scaling='level', n_components=20, use_eigendec=True, nocenter=True)
            r2_values = pca_X.calculate_r2(test_data_set)
            self.assertTrue(np.allclose(r2_values, r2_test))
        except Exception:
            self.assertTrue(False)

        try:
            pca_X = reduction.PCA(test_data_set, scaling='max', n_components=20, use_eigendec=True, nocenter=True)
            r2_values = pca_X.calculate_r2(test_data_set)
            self.assertTrue(np.allclose(r2_values, r2_test))
        except Exception:
            self.assertTrue(False)

        try:
            pca_X = reduction.PCA(test_data_set, scaling='poisson', n_components=20, use_eigendec=True, nocenter=True)
            r2_values = pca_X.calculate_r2(test_data_set)
            self.assertTrue(np.allclose(r2_values, r2_test))
        except Exception:
            self.assertTrue(False)

        try:
            pca_X = reduction.PCA(test_data_set, scaling='vast_2', n_components=20, use_eigendec=True, nocenter=True)
            r2_values = pca_X.calculate_r2(test_data_set)
            self.assertTrue(np.allclose(r2_values, r2_test))
        except Exception:
            self.assertTrue(False)

        try:
            pca_X = reduction.PCA(test_data_set, scaling='vast_3', n_components=20, use_eigendec=True, nocenter=True)
            r2_values = pca_X.calculate_r2(test_data_set)
            self.assertTrue(np.allclose(r2_values, r2_test))
        except Exception:
            self.assertTrue(False)

        try:
            pca_X = reduction.PCA(test_data_set, scaling='vast_4', n_components=20, use_eigendec=True, nocenter=True)
            r2_values = pca_X.calculate_r2(test_data_set)
            self.assertTrue(np.allclose(r2_values, r2_test))
        except Exception:
            self.assertTrue(False)

# ------------------------------------------------------------------------------
