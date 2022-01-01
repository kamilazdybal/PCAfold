import unittest
import numpy as np
from PCAfold import preprocess
from PCAfold import reduction
from PCAfold import analysis

class Analysis(unittest.TestCase):

    def test_analysis__manifold_informed_backward_elimination__allowed_calls(self):

        X = np.random.rand(100,5)
        X_source = np.random.rand(100,5)
        variable_names = ['X1', 'X2', 'X3', 'X4', 'X5']
        scaling='auto'
        bandwidth_values = bandwidth_values = np.logspace(-4, 2, 50)
        target_variables = X[:,0:4]

        try:
            (ordered_variables, selected_variables, optimized_cost, costs) = analysis.manifold_informed_backward_elimination(X, X_source, variable_names, scaling, bandwidth_values)
        except Exception:
            self.assertTrue(False)

        try:
            (ordered_variables, selected_variables, optimized_cost, costs) = analysis.manifold_informed_backward_elimination(X, X_source, variable_names, scaling, bandwidth_values, target_variables=None, add_transformed_source=True, target_manifold_dimensionality=2,  penalty_function=None, norm='max', integrate_to_peak=False)
        except Exception:
            self.assertTrue(False)

        try:
            (ordered_variables, selected_variables, optimized_cost, costs) = analysis.manifold_informed_backward_elimination(X, X_source, variable_names, scaling, bandwidth_values, target_variables=target_variables)
        except Exception:
            self.assertTrue(False)

        try:
            (ordered_variables, selected_variables, optimized_cost, costs) = analysis.manifold_informed_backward_elimination(X, X_source, variable_names, scaling, bandwidth_values, target_variables=target_variables, add_transformed_source=True, target_manifold_dimensionality=3, penalty_function='peak', norm='cumulative', integrate_to_peak=True)
        except Exception:
            self.assertTrue(False)

        try:
            (ordered_variables, selected_variables, optimized_cost, costs) = analysis.manifold_informed_backward_elimination(X, X_source, variable_names, scaling, bandwidth_values, target_variables=target_variables, add_transformed_source=False, target_manifold_dimensionality=3, penalty_function='peak', norm='cumulative', integrate_to_peak=True)
        except Exception:
            self.assertTrue(False)

        # Test PC source terms in various log spaces:
        try:
            (ordered_variables, selected_variables, optimized_cost, costs) = analysis.manifold_informed_backward_elimination(X, X_source, variable_names, scaling, bandwidth_values, target_variables=None, add_transformed_source=True, source_space='symlog', target_manifold_dimensionality=2)
        except Exception:
            self.assertTrue(False)

        try:
            (ordered_variables, selected_variables, optimized_cost, costs) = analysis.manifold_informed_backward_elimination(X, X_source, variable_names, scaling, bandwidth_values, target_variables=None, add_transformed_source=True, source_space='continuous-symlog', target_manifold_dimensionality=2)
        except Exception:
            self.assertTrue(False)

        try:
            (ordered_variables, selected_variables, optimized_cost, costs) = analysis.manifold_informed_backward_elimination(X, X_source, variable_names, scaling, bandwidth_values, target_variables=None, add_transformed_source=True, source_space='original-and-symlog', target_manifold_dimensionality=2)
        except Exception:
            self.assertTrue(False)

        try:
            (ordered_variables, selected_variables, optimized_cost, costs) = analysis.manifold_informed_backward_elimination(X, X_source, variable_names, scaling, bandwidth_values, target_variables=None, add_transformed_source=True, source_space='original-and-continuous-symlog', target_manifold_dimensionality=2)
        except Exception:
            self.assertTrue(False)

# ------------------------------------------------------------------------------

    def test_analysis__manifold_informed_backward_elimination__not_allowed_calls(self):

        X = np.random.rand(100,5)
        X_source = np.random.rand(100,5)
        variable_names = ['X1', 'X2', 'X3', 'X4', 'X5']
        scaling='auto'
        bandwidth_values = bandwidth_values = np.logspace(-4, 2, 50)

        # Need to specify target_variables or set add_transformed_source to True:
        with self.assertRaises(ValueError):
            (ordered_variables, selected_variables, optimized_cost, costs) = analysis.manifold_informed_backward_elimination(X, X_source, variable_names, scaling, bandwidth_values, add_transformed_source=False)

        # Wrong type:
        with self.assertRaises(ValueError):
            (ordered_variables, selected_variables, optimized_cost, costs) = analysis.manifold_informed_backward_elimination(X, X_source, variable_names, scaling, bandwidth_values, target_variables=[1])

        with self.assertRaises(ValueError):
            (ordered_variables, selected_variables, optimized_cost, costs) = analysis.manifold_informed_backward_elimination(X, X_source, variable_names, scaling, bandwidth_values, add_transformed_source=[1])

        with self.assertRaises(ValueError):
            (ordered_variables, selected_variables, optimized_cost, costs) = analysis.manifold_informed_backward_elimination(X, X_source, variable_names, scaling, bandwidth_values, source_space=[1])

        with self.assertRaises(ValueError):
            (ordered_variables, selected_variables, optimized_cost, costs) = analysis.manifold_informed_backward_elimination(X, X_source, variable_names, scaling, bandwidth_values, target_manifold_dimensionality=[1])

        with self.assertRaises(ValueError):
            (ordered_variables, selected_variables, optimized_cost, costs) = analysis.manifold_informed_backward_elimination(X, X_source, variable_names, scaling, bandwidth_values, penalty_function=[1])

        with self.assertRaises(ValueError):
            (ordered_variables, selected_variables, optimized_cost, costs) = analysis.manifold_informed_backward_elimination(X, X_source, variable_names, scaling, bandwidth_values, norm=[1])

        with self.assertRaises(ValueError):
            (ordered_variables, selected_variables, optimized_cost, costs) = analysis.manifold_informed_backward_elimination(X, X_source, variable_names, scaling, bandwidth_values, integrate_to_peak=[1])

        with self.assertRaises(ValueError):
            (ordered_variables, selected_variables, optimized_cost, costs) = analysis.manifold_informed_backward_elimination(X, X_source, variable_names, scaling, bandwidth_values, verbose=[1])

        with self.assertRaises(ValueError):
            (ordered_variables, selected_variables, optimized_cost, costs) = analysis.manifold_informed_backward_elimination([1], X_source, variable_names, scaling, bandwidth_values)

        with self.assertRaises(ValueError):
            (ordered_variables, selected_variables, optimized_cost, costs) = analysis.manifold_informed_backward_elimination(X, [1], variable_names, scaling, bandwidth_values)

        with self.assertRaises(ValueError):
            (ordered_variables, selected_variables, optimized_cost, costs) = analysis.manifold_informed_backward_elimination(X, X_source, 'A', scaling, bandwidth_values)

        with self.assertRaises(ValueError):
            (ordered_variables, selected_variables, optimized_cost, costs) = analysis.manifold_informed_backward_elimination(X, X_source, variable_names, [1], bandwidth_values)

        with self.assertRaises(ValueError):
            (ordered_variables, selected_variables, optimized_cost, costs) = analysis.manifold_informed_backward_elimination(X, X_source, variable_names, scaling, [1])

# ------------------------------------------------------------------------------

    def test_analysis__manifold_informed_backward_elimination__computation(self):

        X = np.random.rand(100,5)
        X_source = np.random.rand(100,5)
        variable_names = ['X1', 'X2', 'X3', 'X4', 'X5']
        scaling='auto'
        bandwidth_values = bandwidth_values = np.logspace(-4, 2, 50)

        # Make sure that the ordered_variables entries are reasonable:
        try:
            (ordered_variables, selected_variables, optimized_cost, costs) = analysis.manifold_informed_backward_elimination(X, X_source, variable_names, scaling, bandwidth_values, target_manifold_dimensionality=3)
            self.assertTrue(len(ordered_variables) > 0)
            self.assertTrue(isinstance(ordered_variables, list))
            for i in ordered_variables:
                self.assertTrue(i >=0 and i <= 5)
        except Exception:
            self.assertTrue(False)

        # Make sure that the selected_variables entries are reasonable:
        try:
            (ordered_variables, selected_variables, optimized_cost, costs) = analysis.manifold_informed_backward_elimination(X, X_source, variable_names, scaling, bandwidth_values, target_manifold_dimensionality=3)
            self.assertTrue(len(selected_variables) > 0)
            self.assertTrue(isinstance(selected_variables, list))
            for i in selected_variables:
                self.assertTrue(i >=0 and i <= 5)
        except Exception:
            self.assertTrue(False)

        # Make sure that the costs entries are reasonable:
        try:
            (ordered_variables, selected_variables, optimized_cost, costs) = analysis.manifold_informed_backward_elimination(X, X_source, variable_names, scaling, bandwidth_values, target_manifold_dimensionality=3)
            self.assertTrue(len(costs) > 0)
            self.assertTrue(isinstance(costs, list))
            for i in costs:
                self.assertTrue(i > 0)
            self.assertTrue(len(costs) == len(ordered_variables))
        except Exception:
            self.assertTrue(False)

        # Make sure that the selected_variables entries correspond to the minimal cost:
        try:
            (ordered_variables, selected_variables, optimized_cost, costs) = analysis.manifold_informed_backward_elimination(X, X_source, variable_names, scaling, bandwidth_values, target_manifold_dimensionality=3)
            (min_cost_function_index, ) = np.where(costs==np.min(costs))
            min_cost_function_index = int(min_cost_function_index)
            selected_costs = costs[0:min_cost_function_index]
            self.assertTrue(selected_variables[-1] == ordered_variables[min_cost_function_index-1])
            self.assertTrue(len(selected_variables) == len(selected_costs))
        except Exception:
            self.assertTrue(False)

# ------------------------------------------------------------------------------
