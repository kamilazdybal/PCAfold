import unittest
import numpy as np
from PCAfold import preprocess
from PCAfold import reduction
from PCAfold import analysis

class Preprocess(unittest.TestCase):

    def test_preprocess__representative_sample_size__allowed_calls(self):

        x, y = np.meshgrid(np.linspace(-1,1,50), np.linspace(-1,1,50))
        xy = np.hstack((x.ravel()[:,None],y.ravel()[:,None]))

        phi_1 = np.exp(-((x*x+y*y) / (1 * 1**2)))
        phi_1 = phi_1.ravel()[:,None]

        phi_2 = np.exp(1*x*y)
        phi_2 = phi_2.ravel()[:,None]

        depvars = np.column_stack((phi_1, phi_2))
        depvars, _, _ = preprocess.center_scale(depvars, scaling='0to1')

        percentages = list(np.linspace(1,99.9,10))

        thresholds = [0.0001, 0.0001]

        try:
            (threshold_idx, sample_sizes, statistics) = preprocess.representative_sample_size(depvars, percentages, thresholds)
        except Exception:
            self.assertTrue(False)

        try:
            (threshold_idx, sample_sizes, statistics) = preprocess.representative_sample_size(depvars, percentages, thresholds, method='kl-divergence')
        except Exception:
            self.assertTrue(False)

        try:
            (threshold_idx, sample_sizes, statistics) = preprocess.representative_sample_size(depvars, percentages, thresholds, method='kl-divergence', statistics='mean')
        except Exception:
            self.assertTrue(False)

        try:
            (threshold_idx, sample_sizes, statistics) = preprocess.representative_sample_size(depvars, percentages, thresholds, method='kl-divergence', statistics='mean', n_resamples=10)
        except Exception:
            self.assertTrue(False)

        try:
            (threshold_idx, sample_sizes, statistics) = preprocess.representative_sample_size(depvars, percentages, thresholds, method='kl-divergence', statistics='mean', n_resamples=1)
        except Exception:
            self.assertTrue(False)

# ------------------------------------------------------------------------------

    def test_preprocess__representative_sample_size__not_allowed_calls(self):

        x, y = np.meshgrid(np.linspace(-1,1,100), np.linspace(-1,1,100))
        xy = np.hstack((x.ravel()[:,None],y.ravel()[:,None]))

        phi_1 = np.exp(-((x*x+y*y) / (1 * 1**2)))
        phi_1 = phi_1.ravel()[:,None]

        phi_2 = np.exp(1*x*y)
        phi_2 = phi_2.ravel()[:,None]

        depvars = np.column_stack((phi_1, phi_2))
        depvars, _, _ = preprocess.center_scale(depvars, scaling='0to1')

        percentages = list(np.linspace(1,99.9,100))

        thresholds = [0.0001, 0.0001]

        with self.assertRaises(ValueError):
            (threshold_idx, sample_sizes, statistics) = preprocess.representative_sample_size(depvars, [], thresholds)

        with self.assertRaises(ValueError):
            (threshold_idx, sample_sizes, statistics) = preprocess.representative_sample_size(depvars, [10,50,200], thresholds)

        with self.assertRaises(ValueError):
            (threshold_idx, sample_sizes, statistics) = preprocess.representative_sample_size(depvars, percentages, thresholds, method=None)

        with self.assertRaises(ValueError):
            (threshold_idx, sample_sizes, statistics) = preprocess.representative_sample_size(depvars, percentages, thresholds, method='method')

        with self.assertRaises(ValueError):
            (threshold_idx, sample_sizes, statistics) = preprocess.representative_sample_size(depvars, percentages, thresholds, statistics='statistics')

        with self.assertRaises(ValueError):
            (threshold_idx, sample_sizes, statistics) = preprocess.representative_sample_size(depvars, percentages, thresholds, n_resamples=0)

        with self.assertRaises(ValueError):
            (threshold_idx, sample_sizes, statistics) = preprocess.representative_sample_size(depvars, percentages, thresholds, n_resamples=-1)

        with self.assertRaises(ValueError):
            (threshold_idx, sample_sizes, statistics) = preprocess.representative_sample_size(depvars, percentages, thresholds, random_seed=[])

        with self.assertRaises(ValueError):
            (threshold_idx, sample_sizes, statistics) = preprocess.representative_sample_size(depvars, percentages, thresholds, verbose=[])

# ------------------------------------------------------------------------------

    def test_preprocess__representative_sample_size__computation(self):

        pass

# ------------------------------------------------------------------------------

    def test_preprocess__representative_sample_size__method_kl_divergence(self):

        x, y = np.meshgrid(np.linspace(-1,1,20), np.linspace(-1,1,20))
        xy = np.hstack((x.ravel()[:,None],y.ravel()[:,None]))

        depvars = np.exp(-((x*x+y*y) / (1 * 1**2)))
        depvars = depvars.ravel()[:,None]
        depvars, _, _ = preprocess.center_scale(depvars, scaling='0to1')

        percentages = list(np.linspace(1,99.9,10))

        thresholds = [0.0001]

        try:
            (threshold_idx, sample_sizes, statistics) = preprocess.representative_sample_size(depvars, percentages, thresholds, method='kl-divergence')
        except Exception:
            self.assertTrue(False)

# ------------------------------------------------------------------------------

    def test_preprocess__representative_sample_size__method_std(self):

        x, y = np.meshgrid(np.linspace(-1,1,20), np.linspace(-1,1,20))
        xy = np.hstack((x.ravel()[:,None],y.ravel()[:,None]))

        depvars = np.exp(-((x*x+y*y) / (1 * 1**2)))
        depvars = depvars.ravel()[:,None]
        depvars, _, _ = preprocess.center_scale(depvars, scaling='0to1')

        percentages = list(np.linspace(1,99.9,10))

        thresholds = [0.001 * np.std(depvars)]

        try:
            (threshold_idx, sample_sizes, statistics) = preprocess.representative_sample_size(depvars, percentages, thresholds, method='std')
        except Exception:
            self.assertTrue(False)

# ------------------------------------------------------------------------------

    def test_preprocess__representative_sample_size__method_median(self):

        x, y = np.meshgrid(np.linspace(-1,1,20), np.linspace(-1,1,20))
        xy = np.hstack((x.ravel()[:,None],y.ravel()[:,None]))

        depvars = np.exp(-((x*x+y*y) / (1 * 1**2)))
        depvars = depvars.ravel()[:,None]
        depvars, _, _ = preprocess.center_scale(depvars, scaling='0to1')

        percentages = list(np.linspace(1,99.9,10))

        thresholds = [0.001 * np.median(depvars)]

        try:
            (threshold_idx, sample_sizes, statistics) = preprocess.representative_sample_size(depvars, percentages, thresholds, method='median')
        except Exception:
            self.assertTrue(False)

# ------------------------------------------------------------------------------

    def test_preprocess__representative_sample_size__method_mean(self):

        x, y = np.meshgrid(np.linspace(-1,1,20), np.linspace(-1,1,20))
        xy = np.hstack((x.ravel()[:,None],y.ravel()[:,None]))

        depvars = np.exp(-((x*x+y*y) / (1 * 1**2)))
        depvars = depvars.ravel()[:,None]
        depvars, _, _ = preprocess.center_scale(depvars, scaling='0to1')

        percentages = list(np.linspace(1,99.9,10))

        thresholds = [0.001 * np.mean(depvars)]

        try:
            (threshold_idx, sample_sizes, statistics) = preprocess.representative_sample_size(depvars, percentages, thresholds, method='mean')
        except Exception:
            self.assertTrue(False)

# ------------------------------------------------------------------------------

    def test_preprocess__representative_sample_size__method_variance(self):

        x, y = np.meshgrid(np.linspace(-1,1,20), np.linspace(-1,1,20))
        xy = np.hstack((x.ravel()[:,None],y.ravel()[:,None]))

        depvars = np.exp(-((x*x+y*y) / (1 * 1**2)))
        depvars = depvars.ravel()[:,None]
        depvars, _, _ = preprocess.center_scale(depvars, scaling='0to1')

        percentages = list(np.linspace(1,99.9,10))

        thresholds = [0.001 * np.var(depvars)]

        try:
            (threshold_idx, sample_sizes, statistics) = preprocess.representative_sample_size(depvars, percentages, thresholds, method='variance')
        except Exception:
            self.assertTrue(False)
