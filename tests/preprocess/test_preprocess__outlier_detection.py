import unittest
import numpy as np
from PCAfold import preprocess
from PCAfold import reduction
from PCAfold import analysis

class Preprocess(unittest.TestCase):

    def test_preprocess__outlier_detection__allowed_calls(self):

        X = np.random.rand(100,10)

        try:
            (idx_outliers_removed, idx_outliers) = preprocess.outlier_detection(X, scaling='auto', method='MULTIVARIATE TRIMMING', trimming_threshold=0.6)
            self.assertTrue(not np.any(np.in1d(idx_outliers_removed, idx_outliers)))
        except Exception:
            self.assertTrue(False)

        try:
            (idx_outliers_removed, idx_outliers) = preprocess.outlier_detection(X, scaling='none', method='MULTIVARIATE TRIMMING', trimming_threshold=0.6)
            self.assertTrue(not np.any(np.in1d(idx_outliers_removed, idx_outliers)))
        except Exception:
            self.assertTrue(False)

        try:
            (idx_outliers_removed, idx_outliers) = preprocess.outlier_detection(X, scaling='auto', method='MULTIVARIATE TRIMMING', trimming_threshold=0.2)
            self.assertTrue(not np.any(np.in1d(idx_outliers_removed, idx_outliers)))
        except Exception:
            self.assertTrue(False)

        try:
            (idx_outliers_removed, idx_outliers) = preprocess.outlier_detection(X, scaling='auto', method='MULTIVARIATE TRIMMING', trimming_threshold=0.1)
            self.assertTrue(not np.any(np.in1d(idx_outliers_removed, idx_outliers)))
        except Exception:
            self.assertTrue(False)

        try:
            (idx_outliers_removed, idx_outliers) = preprocess.outlier_detection(X, scaling='auto', method='PC CLASSIFIER')
            self.assertTrue(not np.any(np.in1d(idx_outliers_removed, idx_outliers)))
        except Exception:
            self.assertTrue(False)

        try:
            (idx_outliers_removed, idx_outliers) = preprocess.outlier_detection(X, scaling='range', method='PC CLASSIFIER')
            self.assertTrue(not np.any(np.in1d(idx_outliers_removed, idx_outliers)))
        except Exception:
            self.assertTrue(False)

        try:
            (idx_outliers_removed, idx_outliers) = preprocess.outlier_detection(X, scaling='pareto', method='PC CLASSIFIER')
            self.assertTrue(not np.any(np.in1d(idx_outliers_removed, idx_outliers)))
        except Exception:
            self.assertTrue(False)

        try:
            (idx_outliers_removed, idx_outliers) = preprocess.outlier_detection(X, scaling='none', method='PC CLASSIFIER', trimming_threshold=0.0)
            self.assertTrue(not np.any(np.in1d(idx_outliers_removed, idx_outliers)))
        except Exception:
            self.assertTrue(False)

        try:
            (idx_outliers_removed, idx_outliers) = preprocess.outlier_detection(X, scaling='none', method='PC CLASSIFIER', trimming_threshold=1.0)
            self.assertTrue(not np.any(np.in1d(idx_outliers_removed, idx_outliers)))
        except Exception:
            self.assertTrue(False)

        try:
            (idx_outliers_removed, idx_outliers) = preprocess.outlier_detection(X, scaling='auto', method='PC CLASSIFIER', quantile_threshold=0.9)
            self.assertTrue(not np.any(np.in1d(idx_outliers_removed, idx_outliers)))
        except Exception:
            self.assertTrue(False)

        try:
            (idx_outliers_removed, idx_outliers) = preprocess.outlier_detection(X, scaling='range', method='PC CLASSIFIER', quantile_threshold=0.99)
            self.assertTrue(not np.any(np.in1d(idx_outliers_removed, idx_outliers)))
        except Exception:
            self.assertTrue(False)

        try:
            (idx_outliers_removed, idx_outliers) = preprocess.outlier_detection(X, scaling='pareto', method='PC CLASSIFIER', quantile_threshold=0.8)
            self.assertTrue(not np.any(np.in1d(idx_outliers_removed, idx_outliers)))
        except Exception:
            self.assertTrue(False)

        try:
            (idx_outliers_removed, idx_outliers) = preprocess.outlier_detection(X, scaling='none', method='PC CLASSIFIER', trimming_threshold=0.0, quantile_threshold=0.5)
            self.assertTrue(not np.any(np.in1d(idx_outliers_removed, idx_outliers)))
        except Exception:
            self.assertTrue(False)

        try:
            (idx_outliers_removed, idx_outliers) = preprocess.outlier_detection(X, scaling='none', method='PC CLASSIFIER', trimming_threshold=1.0, quantile_threshold=0.9)
            self.assertTrue(not np.any(np.in1d(idx_outliers_removed, idx_outliers)))
        except Exception:
            self.assertTrue(False)

# ------------------------------------------------------------------------------

    def test_preprocess__outlier_detection__not_allowed_calls(self):

        X = np.random.rand(100,10)

        with self.assertRaises(ValueError):
            (idx_outliers_removed, idx_outliers) = preprocess.outlier_detection(X, scaling='scaling')

        with self.assertRaises(ValueError):
            (idx_outliers_removed, idx_outliers) = preprocess.outlier_detection(X, scaling='auto', trimming_threshold=1)

        with self.assertRaises(ValueError):
            (idx_outliers_removed, idx_outliers) = preprocess.outlier_detection(X, scaling='auto', quantile_threshold=1)

        with self.assertRaises(ValueError):
            (idx_outliers_removed, idx_outliers) = preprocess.outlier_detection(X, scaling='none', method='method')

        with self.assertRaises(ValueError):
            (idx_outliers_removed, idx_outliers) = preprocess.outlier_detection(X, scaling='none', verbose=1)

        with self.assertRaises(ValueError):
            (idx_outliers_removed, idx_outliers) = preprocess.outlier_detection(X, scaling='none', verbose=0)

        with self.assertRaises(ValueError):
            (idx_outliers_removed, idx_outliers) = preprocess.outlier_detection(X, scaling='none', trimming_threshold=-1)

        with self.assertRaises(ValueError):
            (idx_outliers_removed, idx_outliers) = preprocess.outlier_detection(X, scaling='none', trimming_threshold=1.1)

        with self.assertRaises(ValueError):
            (idx_outliers_removed, idx_outliers) = preprocess.outlier_detection(X, scaling='none', method='PC CLASSIFIER', quantile_threshold=1.1)

        with self.assertRaises(ValueError):
            (idx_outliers_removed, idx_outliers) = preprocess.outlier_detection(X, scaling='none', method='PC CLASSIFIER', quantile_threshold=-1)

        with self.assertRaises(ValueError):
            (idx_outliers_removed, idx_outliers) = preprocess.outlier_detection(X, scaling='none', method='PC CLASSIFIER', trimming_threshold=0.9, quantile_threshold=1.1)

        with self.assertRaises(ValueError):
            (idx_outliers_removed, idx_outliers) = preprocess.outlier_detection(X, scaling='none', method='PC CLASSIFIER', trimming_threshold=0.9, quantile_threshold=-1)

        with self.assertRaises(ValueError):
            (idx_outliers_removed, idx_outliers) = preprocess.outlier_detection([1,2,3], scaling='auto')

        X = np.random.rand(100,)

        with self.assertRaises(ValueError):
            (idx_outliers_removed, idx_outliers) = preprocess.outlier_detection(X, scaling='auto')

# ------------------------------------------------------------------------------
