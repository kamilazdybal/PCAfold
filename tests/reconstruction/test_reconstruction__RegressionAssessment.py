import unittest
import numpy as np
from PCAfold import preprocess
from PCAfold import reduction
from PCAfold import reconstruction

class Reconstruction(unittest.TestCase):

    def test_reconstruction__RegressionAssessment__allowed_class_init(self):

        X = np.random.rand(100,3)
        pca_X = reduction.PCA(X, scaling='auto', n_components=2)
        X_rec = pca_X.reconstruct(pca_X.transform(X))

        try:
            regression_metrics = reconstruction.RegressionAssessment(X, X_rec)
        except:
            self.assertTrue(False)

        try:
            regression_metrics = reconstruction.RegressionAssessment(X, X_rec, variable_names=['X1', 'X2', 'X3'], norm='range', tolerance=0.01)
        except:
            self.assertTrue(False)

# ------------------------------------------------------------------------------

    def test_reconstruction__RegressionAssessment__not_allowed_class_init(self):

        X = np.random.rand(100,3)
        pca_X = reduction.PCA(X, scaling='auto', n_components=2)
        X_rec = pca_X.reconstruct(pca_X.transform(X))

        with self.assertRaises(ValueError):
            regression_metrics = reconstruction.RegressionAssessment(X, [1])

        with self.assertRaises(ValueError):
            regression_metrics = reconstruction.RegressionAssessment([1], X_rec)

        with self.assertRaises(ValueError):
            regression_metrics = reconstruction.RegressionAssessment(X, X_rec, variable_names=1)

        with self.assertRaises(ValueError):
            regression_metrics = reconstruction.RegressionAssessment(X, X_rec, norm=1)

        with self.assertRaises(ValueError):
            regression_metrics = reconstruction.RegressionAssessment(X, X_rec, tolerance=[1])

# ------------------------------------------------------------------------------

    def test_reconstruction__RegressionAssessment__access_attributes(self):

        X = np.random.rand(100,3)
        pca_X = reduction.PCA(X, scaling='auto', n_components=2)
        X_rec = pca_X.reconstruct(pca_X.transform(X))

        try:
            regression_metrics = reconstruction.RegressionAssessment(X, X_rec)
            r2 = regression_metrics.coefficient_of_determination
            mae = regression_metrics.mean_absolute_error
            mse = regression_metrics.mean_squared_error
            rmse = regression_metrics.root_mean_squared_error
            nrmse = regression_metrics.normalized_root_mean_squared_error
            gde = regression_metrics.good_direction_estimate
        except:
            self.assertTrue(False)

# ------------------------------------------------------------------------------
