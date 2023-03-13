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

        try:
            (sample_sizes, statistics) = preprocess.representative_sample_size(depvars, percentages)
        except Exception:
            self.assertTrue(False)

        try:
            (sample_sizes, statistics) = preprocess.representative_sample_size(depvars, percentages, method='kl-divergence')
        except Exception:
            self.assertTrue(False)

        try:
            (sample_sizes, statistics) = preprocess.representative_sample_size(depvars, percentages, method='kl-divergence', statistics='mean')
        except Exception:
            self.assertTrue(False)

        try:
            (sample_sizes, statistics) = preprocess.representative_sample_size(depvars, percentages, method='kl-divergence', statistics='mean', n_resamples=10)
        except Exception:
            self.assertTrue(False)

        try:
            (sample_sizes, statistics) = preprocess.representative_sample_size(depvars, percentages, method='kl-divergence', statistics='mean', n_resamples=1)
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

        with self.assertRaises(ValueError):
            (sample_sizes, statistics) = preprocess.representative_sample_size(depvars, [])

        with self.assertRaises(ValueError):
            (sample_sizes, statistics) = preprocess.representative_sample_size(depvars, [10,50,200])

        with self.assertRaises(ValueError):
            (sample_sizes, statistics) = preprocess.representative_sample_size(depvars, percentages, method=None)

        with self.assertRaises(ValueError):
            (sample_sizes, statistics) = preprocess.representative_sample_size(depvars, percentages, method='method')

        with self.assertRaises(ValueError):
            (sample_sizes, statistics) = preprocess.representative_sample_size(depvars, percentages, statistics='statistics')

        with self.assertRaises(ValueError):
            (sample_sizes, statistics) = preprocess.representative_sample_size(depvars, percentages, n_resamples=0)

        with self.assertRaises(ValueError):
            (sample_sizes, statistics) = preprocess.representative_sample_size(depvars, percentages, n_resamples=-1)

        with self.assertRaises(ValueError):
            (sample_sizes, statistics) = preprocess.representative_sample_size(depvars, percentages, threshold=[])

        with self.assertRaises(ValueError):
            (sample_sizes, statistics) = preprocess.representative_sample_size(depvars, percentages, random_seed=[])

        with self.assertRaises(ValueError):
            (sample_sizes, statistics) = preprocess.representative_sample_size(depvars, percentages, verbose=[])

# ------------------------------------------------------------------------------

    def test_preprocess__representative_sample_size__computation(self):

        pass
