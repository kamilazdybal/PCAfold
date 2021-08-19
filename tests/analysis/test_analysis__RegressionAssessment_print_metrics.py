import unittest
import numpy as np
from PCAfold import preprocess
from PCAfold import reduction
from PCAfold import analysis

class Analysis(unittest.TestCase):

    def test_analysis__RegressionAssessment_print_metrics__allowed_calls(self):

        X = np.random.rand(100,3)
        pca_X = reduction.PCA(X, scaling='auto', n_components=2)
        X_rec = pca_X.reconstruct(pca_X.transform(X))

        try:
            regression_metrics = analysis.RegressionAssessment(X, X_rec)
            regression_metrics.print_metrics([])
            regression_metrics.print_metrics([], float_format='%.2f')
        except:
            self.assertTrue(False)

        try:
            regression_metrics = analysis.RegressionAssessment(X, X_rec, variable_names=['X1', 'X2', 'X3'], norm='range', tolerance=0.01)
            regression_metrics.print_metrics([], float_format='%.2f')
        except:
            self.assertTrue(False)

# ------------------------------------------------------------------------------

    def test_analysis__RegressionAssessment_print_metrics__not_allowed_calls(self):

        X = np.random.rand(100,3)
        pca_X = reduction.PCA(X, scaling='auto', n_components=2)
        X_rec = pca_X.reconstruct(pca_X.transform(X))
        regression_metrics = analysis.RegressionAssessment(X, X_rec)

        with self.assertRaises(ValueError):
            regression_metrics.print_metrics(['none'])

        with self.assertRaises(ValueError):
            regression_metrics.print_metrics([1])

        with self.assertRaises(ValueError):
            regression_metrics.print_metrics('raw')

        with self.assertRaises(ValueError):
            regression_metrics.print_metrics(['raw'], float_format=[1])

        with self.assertRaises(ValueError):
            regression_metrics.print_metrics(['raw'], float_format=1)

# ------------------------------------------------------------------------------
