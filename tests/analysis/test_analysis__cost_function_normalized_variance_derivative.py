import unittest
import numpy as np
from PCAfold import preprocess
from PCAfold import reduction
from PCAfold import analysis

class Analysis(unittest.TestCase):

    def test_analysis__cost_function_normalized_variance_derivative__allowed_calls(self):

        X = np.random.rand(100,10)
        variable_names = ['X_' + str(i) for i in range(0,10)]
        pca_X = reduction.PCA(X, n_components=2)
        principal_components = pca_X.transform(X)
        bandwidth_values = np.logspace(-4, 2, 10)

        variance_data = analysis.compute_normalized_variance(principal_components,
                                                    X,
                                                    depvar_names=variable_names,
                                                    bandwidth_values=bandwidth_values)

        try:
            cost = analysis.cost_function_normalized_variance_derivative(variance_data)
        except:
            self.assertTrue(False)

# ------------------------------------------------------------------------------

    def test_analysis__cost_function_normalized_variance_derivative__all_penalties(self):

        X = np.random.rand(100,10)
        variable_names = ['X_' + str(i) for i in range(0,10)]
        pca_X = reduction.PCA(X, n_components=2)
        principal_components = pca_X.transform(X)
        bandwidth_values = np.logspace(-4, 2, 10)

        variance_data = analysis.compute_normalized_variance(principal_components,
                                                    X,
                                                    depvar_names=variable_names,
                                                    bandwidth_values=bandwidth_values)
        try:
            cost = analysis.cost_function_normalized_variance_derivative(variance_data,
                                                                penalty_function=None,
                                                                power=0.5,
                                                                vertical_shift=1,
                                                                norm='max',
                                                                integrate_to_peak=True)
            cost = analysis.cost_function_normalized_variance_derivative(variance_data,
                                                                penalty_function='peak',
                                                                power=0.5,
                                                                vertical_shift=1,
                                                                norm='max',
                                                                integrate_to_peak=True)
            cost = analysis.cost_function_normalized_variance_derivative(variance_data,
                                                                penalty_function='sigma',
                                                                power=0.5,
                                                                vertical_shift=1,
                                                                norm='max',
                                                                integrate_to_peak=True)
            cost = analysis.cost_function_normalized_variance_derivative(variance_data,
                                                                penalty_function='log-sigma-over-peak',
                                                                power=0.5,
                                                                vertical_shift=1,
                                                                norm='max',
                                                                integrate_to_peak=True)
        except:
            self.assertTrue(False)

        try:
            cost = analysis.cost_function_normalized_variance_derivative(variance_data,
                                                                penalty_function=None,
                                                                power=0.5,
                                                                vertical_shift=1,
                                                                norm='max',
                                                                integrate_to_peak=False)
            cost = analysis.cost_function_normalized_variance_derivative(variance_data,
                                                                penalty_function='peak',
                                                                power=0.5,
                                                                vertical_shift=1,
                                                                norm='max',
                                                                integrate_to_peak=False)
            cost = analysis.cost_function_normalized_variance_derivative(variance_data,
                                                                penalty_function='sigma',
                                                                power=0.5,
                                                                vertical_shift=1,
                                                                norm='max',
                                                                integrate_to_peak=False)
            cost = analysis.cost_function_normalized_variance_derivative(variance_data,
                                                                penalty_function='log-sigma-over-peak',
                                                                power=0.5,
                                                                vertical_shift=1,
                                                                norm='max',
                                                                integrate_to_peak=False)
        except:
            self.assertTrue(False)

# ------------------------------------------------------------------------------

    def test_analysis__cost_function_normalized_variance_derivative__all_norms(self):

        X = np.random.rand(100,10)
        variable_names = ['X_' + str(i) for i in range(0,10)]
        pca_X = reduction.PCA(X, n_components=2)
        principal_components = pca_X.transform(X)
        bandwidth_values = np.logspace(-4, 2, 10)

        variance_data = analysis.compute_normalized_variance(principal_components,
                                                    X,
                                                    depvar_names=variable_names,
                                                    bandwidth_values=bandwidth_values)
        try:
            cost = analysis.cost_function_normalized_variance_derivative(variance_data,
                                                                penalty_function=None,
                                                                norm=None,
                                                                integrate_to_peak=True)
            cost = analysis.cost_function_normalized_variance_derivative(variance_data,
                                                                penalty_function=None,
                                                                norm='average',
                                                                integrate_to_peak=True)
            cost = analysis.cost_function_normalized_variance_derivative(variance_data,
                                                                penalty_function=None,
                                                                norm='cumulative',
                                                                integrate_to_peak=True)
            cost = analysis.cost_function_normalized_variance_derivative(variance_data,
                                                                penalty_function=None,
                                                                norm='max',
                                                                integrate_to_peak=True)
            cost = analysis.cost_function_normalized_variance_derivative(variance_data,
                                                                penalty_function=None,
                                                                norm='median',
                                                                integrate_to_peak=True)
            cost = analysis.cost_function_normalized_variance_derivative(variance_data,
                                                                penalty_function=None,
                                                                norm='min',
                                                                integrate_to_peak=True)
        except:
            self.assertTrue(False)

        try:
            cost = analysis.cost_function_normalized_variance_derivative(variance_data,
                                                                penalty_function=None,
                                                                norm=None,
                                                                integrate_to_peak=False)
            cost = analysis.cost_function_normalized_variance_derivative(variance_data,
                                                                penalty_function=None,
                                                                norm='average',
                                                                integrate_to_peak=False)
            cost = analysis.cost_function_normalized_variance_derivative(variance_data,
                                                                penalty_function=None,
                                                                norm='cumulative',
                                                                integrate_to_peak=False)
            cost = analysis.cost_function_normalized_variance_derivative(variance_data,
                                                                penalty_function=None,
                                                                norm='max',
                                                                integrate_to_peak=False)
            cost = analysis.cost_function_normalized_variance_derivative(variance_data,
                                                                penalty_function=None,
                                                                norm='median',
                                                                integrate_to_peak=False)
            cost = analysis.cost_function_normalized_variance_derivative(variance_data,
                                                                penalty_function=None,
                                                                norm='min',
                                                                integrate_to_peak=False)
        except:
            self.assertTrue(False)

# ------------------------------------------------------------------------------

    def test_analysis__cost_function_normalized_variance_derivative__not_allowed_calls(self):

        X = np.random.rand(100,10)
        variable_names = ['X_' + str(i) for i in range(0,10)]
        pca_X = reduction.PCA(X, n_components=2)
        principal_components = pca_X.transform(X)
        bandwidth_values = np.logspace(-4, 2, 50)

        variance_data = analysis.compute_normalized_variance(principal_components,
                                                    X,
                                                    depvar_names=variable_names,
                                                    bandwidth_values=bandwidth_values)
        with self.assertRaises(ValueError):
            cost = analysis.cost_function_normalized_variance_derivative(variance_data, penalty_function='test')

        with self.assertRaises(ValueError):
            cost = analysis.cost_function_normalized_variance_derivative(variance_data, power=[])

        with self.assertRaises(ValueError):
            cost = analysis.cost_function_normalized_variance_derivative(variance_data, vertical_shift=[])

        with self.assertRaises(ValueError):
            cost = analysis.cost_function_normalized_variance_derivative(variance_data, norm='test')

        with self.assertRaises(ValueError):
            cost = analysis.cost_function_normalized_variance_derivative(variance_data, integrate_to_peak=[])

# ------------------------------------------------------------------------------

    def test_analysis__cost_function_normalized_variance_derivative__computation(self):

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
            cost = analysis.cost_function_normalized_variance_derivative(variance_data,
                                                                penalty_function=None)
            self.assertTrue(type(cost)==list)
            self.assertTrue(len(cost)==10)
            for i in range(0,10):
                self.assertTrue(cost[i]>0)
        except:
            self.assertTrue(False)

        try:
            for i in range(0,10):
                X = np.random.rand(100,4)
                variable_names = ['X_' + str(i) for i in range(0,4)]
                pca_X = reduction.PCA(X, n_components=2)
                principal_components = pca_X.transform(X)

                variance_data = analysis.compute_normalized_variance(principal_components,
                                                            X,
                                                            depvar_names=variable_names,
                                                            bandwidth_values=bandwidth_values)

                cost = analysis.cost_function_normalized_variance_derivative(variance_data,
                                                                    penalty_function='log-sigma-over-peak',
                                                                    norm='average')

                cost_to_peak = analysis.cost_function_normalized_variance_derivative(variance_data,
                                                                    penalty_function='log-sigma-over-peak',
                                                                    norm='average',
                                                                    integrate_to_peak=True)

                self.assertTrue(cost_to_peak < cost)

                cost = analysis.cost_function_normalized_variance_derivative(variance_data,
                                                                    penalty_function='log-sigma-over-peak',
                                                                    norm='cumulative')

                cost_to_peak = analysis.cost_function_normalized_variance_derivative(variance_data,
                                                                    penalty_function='log-sigma-over-peak',
                                                                    norm='cumulative',
                                                                    integrate_to_peak=True)

                self.assertTrue(cost_to_peak < cost)

                cost = analysis.cost_function_normalized_variance_derivative(variance_data,
                                                                    penalty_function='log-sigma-over-peak',
                                                                    norm='max')

                cost_to_peak = analysis.cost_function_normalized_variance_derivative(variance_data,
                                                                    penalty_function='log-sigma-over-peak',
                                                                    norm='max',
                                                                    integrate_to_peak=True)

                self.assertTrue(cost_to_peak < cost)

                cost = analysis.cost_function_normalized_variance_derivative(variance_data,
                                                                    penalty_function='log-sigma-over-peak',
                                                                    norm='median')

                cost_to_peak = analysis.cost_function_normalized_variance_derivative(variance_data,
                                                                    penalty_function='log-sigma-over-peak',
                                                                    norm='median',
                                                                    integrate_to_peak=True)

                self.assertTrue(cost_to_peak < cost)

                cost = analysis.cost_function_normalized_variance_derivative(variance_data,
                                                                    penalty_function='log-sigma-over-peak',
                                                                    norm='min')

                cost_to_peak = analysis.cost_function_normalized_variance_derivative(variance_data,
                                                                    penalty_function='log-sigma-over-peak',
                                                                    norm='min',
                                                                    integrate_to_peak=True)

                self.assertTrue(cost_to_peak < cost)

        except:
            self.assertTrue(False)

# ------------------------------------------------------------------------------
