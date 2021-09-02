import unittest
import numpy as np
from PCAfold import preprocess
from PCAfold import reduction
from PCAfold import analysis

class Analysis(unittest.TestCase):

    def test_analysis__manifold_informed_feature_selection__allowed_calls(self):

        X = np.random.rand(100,5)
        X_source = np.random.rand(100,5)
        variable_names = ['X1', 'X2', 'X3', 'X4', 'X5']
        scaling='auto'
        bandwidth_values = bandwidth_values = np.logspace(-4, 2, 50)

        # Direct integration:
        try:
            (selected_variables, costs) = analysis.manifold_informed_feature_selection(X, X_source, variable_names, scaling, bandwidth_values, target_manifold_dimensionality=2, direct_integration=True)
        except Exception:
            self.assertTrue(False)

        d_hat_variables = X[:,0:4]

        try:
            (selected_variables, costs) = analysis.manifold_informed_feature_selection(X, X_source, variable_names, scaling, bandwidth_values, d_hat_variables=d_hat_variables, add_transformed_source=False, target_manifold_dimensionality=2, direct_integration=True)
        except Exception:
            self.assertTrue(False)

        try:
            (selected_variables, costs) = analysis.manifold_informed_feature_selection(X, X_source, variable_names, scaling, bandwidth_values, d_hat_variables=d_hat_variables, add_transformed_source=True, target_manifold_dimensionality=2, direct_integration=True)
        except Exception:
            self.assertTrue(False)

        try:
            (selected_variables, costs) = analysis.manifold_informed_feature_selection(X, X_source, variable_names, scaling, bandwidth_values, d_hat_variables=d_hat_variables, add_transformed_source=False, target_manifold_dimensionality=3, direct_integration=True)
        except Exception:
            self.assertTrue(False)

        # From normalized variance:
        try:
            (selected_variables, costs) = analysis.manifold_informed_feature_selection(X, X_source, variable_names, scaling, bandwidth_values, target_manifold_dimensionality=2, direct_integration=False)
        except Exception:
            self.assertTrue(False)

        d_hat_variables = X[:,0:4]

        try:
            (selected_variables, costs) = analysis.manifold_informed_feature_selection(X, X_source, variable_names, scaling, bandwidth_values, d_hat_variables=d_hat_variables, add_transformed_source=False, target_manifold_dimensionality=2, direct_integration=False)
        except Exception:
            self.assertTrue(False)

        try:
            (selected_variables, costs) = analysis.manifold_informed_feature_selection(X, X_source, variable_names, scaling, bandwidth_values, d_hat_variables=d_hat_variables, add_transformed_source=True, target_manifold_dimensionality=2, direct_integration=False)
        except Exception:
            self.assertTrue(False)

        try:
            (selected_variables, costs) = analysis.manifold_informed_feature_selection(X, X_source, variable_names, scaling, bandwidth_values, d_hat_variables=d_hat_variables, add_transformed_source=False, target_manifold_dimensionality=3, direct_integration=False)
        except Exception:
            self.assertTrue(False)

# ------------------------------------------------------------------------------

    def test_analysis__manifold_informed_feature_selection__not_allowed_calls(self):

        X = np.random.rand(100,5)
        X_source = np.random.rand(100,5)
        variable_names = ['X1', 'X2', 'X3', 'X4', 'X5']
        scaling='auto'
        bandwidth_values = bandwidth_values = np.logspace(-4, 2, 50)

        # Need to specify d_hat_variables or set add_transformed_source to True:
        with self.assertRaises(ValueError):
            (selected_variables, costs) = analysis.manifold_informed_feature_selection(X, X_source, variable_names, scaling, bandwidth_values, add_transformed_source=False, direct_integration=True)

        with self.assertRaises(ValueError):
            (selected_variables, costs) = analysis.manifold_informed_feature_selection(X, X_source, variable_names, scaling, bandwidth_values, add_transformed_source=False, direct_integration=False)

        # Wrong type:
        with self.assertRaises(ValueError):
            (selected_variables, costs) = analysis.manifold_informed_feature_selection(X, X_source, variable_names, scaling, bandwidth_values, d_hat_variables=[1])

        with self.assertRaises(ValueError):
            (selected_variables, costs) = analysis.manifold_informed_feature_selection(X, X_source, variable_names, scaling, bandwidth_values, add_transformed_source=[1])

        with self.assertRaises(ValueError):
            (selected_variables, costs) = analysis.manifold_informed_feature_selection(X, X_source, variable_names, scaling, bandwidth_values, target_manifold_dimensionality=[1])

        with self.assertRaises(ValueError):
            (selected_variables, costs) = analysis.manifold_informed_feature_selection(X, X_source, variable_names, scaling, bandwidth_values, bootstrap_variables=1)

        with self.assertRaises(ValueError):
            (selected_variables, costs) = analysis.manifold_informed_feature_selection(X, X_source, variable_names, scaling, bandwidth_values, weight_area=[1])

        with self.assertRaises(ValueError):
            (selected_variables, costs) = analysis.manifold_informed_feature_selection(X, X_source, variable_names, scaling, bandwidth_values, direct_integration=[1])

        with self.assertRaises(ValueError):
            (selected_variables, costs) = analysis.manifold_informed_feature_selection(X, X_source, variable_names, scaling, bandwidth_values, verbose=[1])

        with self.assertRaises(ValueError):
            (selected_variables, costs) = analysis.manifold_informed_feature_selection([1], X_source, variable_names, scaling, bandwidth_values)

        with self.assertRaises(ValueError):
            (selected_variables, costs) = analysis.manifold_informed_feature_selection(X, [1], variable_names, scaling, bandwidth_values)

        with self.assertRaises(ValueError):
            (selected_variables, costs) = analysis.manifold_informed_feature_selection(X, X_source, 'A', scaling, bandwidth_values)

        with self.assertRaises(ValueError):
            (selected_variables, costs) = analysis.manifold_informed_feature_selection(X, X_source, variable_names, [1], bandwidth_values)

        with self.assertRaises(ValueError):
            (selected_variables, costs) = analysis.manifold_informed_feature_selection(X, X_source, variable_names, scaling, [1])

# ------------------------------------------------------------------------------

    def test_analysis__manifold_informed_feature_selection__computation(self):

        X = np.random.rand(100,5)
        X_source = np.random.rand(100,5)
        variable_names = ['X1', 'X2', 'X3', 'X4', 'X5']
        scaling='auto'
        bandwidth_values = bandwidth_values = np.logspace(-4, 2, 50)

        # Make sure that the selected_variables entries are reasonable:
        try:
            (selected_variables, costs) = analysis.manifold_informed_feature_selection(X, X_source, variable_names, scaling, bandwidth_values, target_manifold_dimensionality=3, direct_integration=True)
            self.assertTrue(len(selected_variables) > 0)
            self.assertTrue(isinstance(selected_variables, list))
            for i in selected_variables:
                self.assertTrue(i >=0 and i <= 5)
        except Exception:
            self.assertTrue(False)

        # Make sure that the costs entries are reasonable:
        try:
            (selected_variables, costs) = analysis.manifold_informed_feature_selection(X, X_source, variable_names, scaling, bandwidth_values, target_manifold_dimensionality=3, direct_integration=True)
            self.assertTrue(len(costs) > 0)
            self.assertTrue(isinstance(costs, list))
            for i in costs:
                self.assertTrue(i > 0)
            self.assertTrue(len(costs) == len(selected_variables))
        except Exception:
            self.assertTrue(False)

        # Make sure that the selected_variables entries are reasonable:
        try:
            (selected_variables, costs) = analysis.manifold_informed_feature_selection(X, X_source, variable_names, scaling, bandwidth_values, target_manifold_dimensionality=3, direct_integration=False)
            self.assertTrue(len(selected_variables) > 0)
            self.assertTrue(isinstance(selected_variables, list))
            for i in selected_variables:
                self.assertTrue(i >=0 and i <= 5)
        except Exception:
            self.assertTrue(False)

        # Make sure that the costs entries are reasonable:
        try:
            (selected_variables, costs) = analysis.manifold_informed_feature_selection(X, X_source, variable_names, scaling, bandwidth_values, target_manifold_dimensionality=3, direct_integration=False)
            self.assertTrue(len(costs) > 0)
            self.assertTrue(isinstance(costs, list))
            for i in costs:
                self.assertTrue(i > 0)
            self.assertTrue(len(costs) == len(selected_variables))
        except Exception:
            self.assertTrue(False)

# ------------------------------------------------------------------------------
