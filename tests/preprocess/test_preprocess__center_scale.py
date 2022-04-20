import unittest
import numpy as np
from PCAfold import preprocess
from PCAfold import reduction
from PCAfold import analysis

class Preprocess(unittest.TestCase):

    def test_preprocess__center_scale__allowed_calls(self):

        pass

# ------------------------------------------------------------------------------

    def test_preprocess__center_scale__not_allowed_calls(self):

        X = np.random.rand(100,20)

        with self.assertRaises(ValueError):
            (X_cs, X_center, X_scale) = preprocess.center_scale(X, 'none', nocenter=1)

        with self.assertRaises(ValueError):
            (X_cs, X_center, X_scale) = preprocess.center_scale([1,2,3], 'none')

        with self.assertRaises(ValueError):
            (X_cs, X_center, X_scale) = preprocess.center_scale(X, 1)

        X = np.random.rand(100,)

        with self.assertRaises(ValueError):
            (X_cs, X_center, X_scale) = preprocess.center_scale(X, 'none')

# ------------------------------------------------------------------------------

    def test_preprocess__center_scale__all_possible_C_and_D(self):

        test_data_set = np.random.rand(100,20)

        # Instantiations that should work:
        try:
            (X_cs, X_center, X_scale) = preprocess.center_scale(test_data_set, 'none', nocenter=False)
            (X_cs, X_center, X_scale) = preprocess.center_scale(test_data_set, 'auto', nocenter=False)
            (X_cs, X_center, X_scale) = preprocess.center_scale(test_data_set, 'std', nocenter=False)
            (X_cs, X_center, X_scale) = preprocess.center_scale(test_data_set, 'pareto', nocenter=False)
            (X_cs, X_center, X_scale) = preprocess.center_scale(test_data_set, 'vast', nocenter=False)
            (X_cs, X_center, X_scale) = preprocess.center_scale(test_data_set, 'range', nocenter=False)
            (X_cs, X_center, X_scale) = preprocess.center_scale(test_data_set, '0to1', nocenter=False)
            (X_cs, X_center, X_scale) = preprocess.center_scale(test_data_set, '-1to1', nocenter=False)
            (X_cs, X_center, X_scale) = preprocess.center_scale(test_data_set, 'level', nocenter=False)
            (X_cs, X_center, X_scale) = preprocess.center_scale(test_data_set, 'max', nocenter=False)
            (X_cs, X_center, X_scale) = preprocess.center_scale(test_data_set, 'variance', nocenter=False)
            (X_cs, X_center, X_scale) = preprocess.center_scale(test_data_set, 'median', nocenter=False)
            (X_cs, X_center, X_scale) = preprocess.center_scale(test_data_set, 'poisson', nocenter=False)
            (X_cs, X_center, X_scale) = preprocess.center_scale(test_data_set, 'vast_2', nocenter=False)
            (X_cs, X_center, X_scale) = preprocess.center_scale(test_data_set, 'vast_3', nocenter=False)
            (X_cs, X_center, X_scale) = preprocess.center_scale(test_data_set, 'vast_4', nocenter=False)

            (X_cs, X_center, X_scale) = preprocess.center_scale(test_data_set, 'none', nocenter=True)
            (X_cs, X_center, X_scale) = preprocess.center_scale(test_data_set, 'auto', nocenter=True)
            (X_cs, X_center, X_scale) = preprocess.center_scale(test_data_set, 'std', nocenter=True)
            (X_cs, X_center, X_scale) = preprocess.center_scale(test_data_set, 'pareto', nocenter=True)
            (X_cs, X_center, X_scale) = preprocess.center_scale(test_data_set, 'vast', nocenter=True)
            (X_cs, X_center, X_scale) = preprocess.center_scale(test_data_set, 'range', nocenter=True)
            (X_cs, X_center, X_scale) = preprocess.center_scale(test_data_set, '0to1', nocenter=True)
            (X_cs, X_center, X_scale) = preprocess.center_scale(test_data_set, '-1to1', nocenter=True)
            (X_cs, X_center, X_scale) = preprocess.center_scale(test_data_set, 'level', nocenter=True)
            (X_cs, X_center, X_scale) = preprocess.center_scale(test_data_set, 'max', nocenter=True)
            (X_cs, X_center, X_scale) = preprocess.center_scale(test_data_set, 'variance', nocenter=True)
            (X_cs, X_center, X_scale) = preprocess.center_scale(test_data_set, 'median', nocenter=True)
            (X_cs, X_center, X_scale) = preprocess.center_scale(test_data_set, 'poisson', nocenter=True)
            (X_cs, X_center, X_scale) = preprocess.center_scale(test_data_set, 'vast_2', nocenter=True)
            (X_cs, X_center, X_scale) = preprocess.center_scale(test_data_set, 'vast_3', nocenter=True)
            (X_cs, X_center, X_scale) = preprocess.center_scale(test_data_set, 'vast_4', nocenter=True)
        except Exception:
            self.assertTrue(False)

# ------------------------------------------------------------------------------

    def test_preprocess__center_scale__on_1D_variable(self):

        test_1D_variable = np.random.rand(100,1)

        # Instantiations that should work:
        try:
            (X_cs, X_center, X_scale) = preprocess.center_scale(test_1D_variable, 'none', nocenter=False)
            (X_cs, X_center, X_scale) = preprocess.center_scale(test_1D_variable, 'auto', nocenter=False)
            (X_cs, X_center, X_scale) = preprocess.center_scale(test_1D_variable, 'std', nocenter=False)
            (X_cs, X_center, X_scale) = preprocess.center_scale(test_1D_variable, 'pareto', nocenter=False)
            (X_cs, X_center, X_scale) = preprocess.center_scale(test_1D_variable, 'vast', nocenter=False)
            (X_cs, X_center, X_scale) = preprocess.center_scale(test_1D_variable, 'range', nocenter=False)
            (X_cs, X_center, X_scale) = preprocess.center_scale(test_1D_variable, '0to1', nocenter=False)
            (X_cs, X_center, X_scale) = preprocess.center_scale(test_1D_variable, '-1to1', nocenter=False)
            (X_cs, X_center, X_scale) = preprocess.center_scale(test_1D_variable, 'level', nocenter=False)
            (X_cs, X_center, X_scale) = preprocess.center_scale(test_1D_variable, 'max', nocenter=False)
            (X_cs, X_center, X_scale) = preprocess.center_scale(test_1D_variable, 'poisson', nocenter=False)
            (X_cs, X_center, X_scale) = preprocess.center_scale(test_1D_variable, 'variance', nocenter=False)
            (X_cs, X_center, X_scale) = preprocess.center_scale(test_1D_variable, 'median', nocenter=False)
            (X_cs, X_center, X_scale) = preprocess.center_scale(test_1D_variable, 'vast_2', nocenter=False)
            (X_cs, X_center, X_scale) = preprocess.center_scale(test_1D_variable, 'vast_3', nocenter=False)
            (X_cs, X_center, X_scale) = preprocess.center_scale(test_1D_variable, 'vast_4', nocenter=False)

            (X_cs, X_center, X_scale) = preprocess.center_scale(test_1D_variable, 'none', nocenter=True)
            (X_cs, X_center, X_scale) = preprocess.center_scale(test_1D_variable, 'auto', nocenter=True)
            (X_cs, X_center, X_scale) = preprocess.center_scale(test_1D_variable, 'std', nocenter=True)
            (X_cs, X_center, X_scale) = preprocess.center_scale(test_1D_variable, 'pareto', nocenter=True)
            (X_cs, X_center, X_scale) = preprocess.center_scale(test_1D_variable, 'vast', nocenter=True)
            (X_cs, X_center, X_scale) = preprocess.center_scale(test_1D_variable, 'range', nocenter=True)
            (X_cs, X_center, X_scale) = preprocess.center_scale(test_1D_variable, '0to1', nocenter=True)
            (X_cs, X_center, X_scale) = preprocess.center_scale(test_1D_variable, '-1to1', nocenter=True)
            (X_cs, X_center, X_scale) = preprocess.center_scale(test_1D_variable, 'level', nocenter=True)
            (X_cs, X_center, X_scale) = preprocess.center_scale(test_1D_variable, 'max', nocenter=True)
            (X_cs, X_center, X_scale) = preprocess.center_scale(test_1D_variable, 'poisson', nocenter=True)
            (X_cs, X_center, X_scale) = preprocess.center_scale(test_1D_variable, 'variance', nocenter=True)
            (X_cs, X_center, X_scale) = preprocess.center_scale(test_1D_variable, 'median', nocenter=True)
            (X_cs, X_center, X_scale) = preprocess.center_scale(test_1D_variable, 'vast_2', nocenter=True)
            (X_cs, X_center, X_scale) = preprocess.center_scale(test_1D_variable, 'vast_3', nocenter=True)
            (X_cs, X_center, X_scale) = preprocess.center_scale(test_1D_variable, 'vast_4', nocenter=True)
        except Exception:
            self.assertTrue(False)

# ------------------------------------------------------------------------------

    def test_preprocess__center_scale__ZeroToOne(self):

        tolerance = 10**-10

        try:
            test_data_set = np.random.rand(100,10)
            (X_cs, X_center, X_scale) = preprocess.center_scale(test_data_set, '0to1', nocenter=False)
            for i in range(0,10):
                self.assertTrue((np.min(X_cs[:,i]) > (- tolerance)) and (np.min(X_cs[:,i]) < tolerance))
        except Exception:
            self.assertTrue(False)

        try:
            test_data_set = np.random.rand(1000,1)
            (X_cs, X_center, X_scale) = preprocess.center_scale(test_data_set, '0to1', nocenter=False)
            for i in range(0,1):
                self.assertTrue((np.min(X_cs[:,i]) > (- tolerance)) and (np.min(X_cs[:,i]) < tolerance))
        except Exception:
            self.assertTrue(False)

        try:
            test_data_set = np.random.rand(2000,1)
            (X_cs, X_center, X_scale) = preprocess.center_scale(test_data_set, '0to1', nocenter=False)
            for i in range(0,1):
                self.assertTrue((np.min(X_cs[:,i]) > (- tolerance)) and (np.min(X_cs[:,i]) < tolerance))
        except Exception:
            self.assertTrue(False)

        try:
            test_data_set = np.random.rand(100,10)
            (X_cs, X_center, X_scale) = preprocess.center_scale(test_data_set, '0to1', nocenter=False)
            for i in range(0,10):
                self.assertTrue((np.max(X_cs[:,i]) > (1 - tolerance)) and (np.max(X_cs[:,i]) < (1 + tolerance)))
        except Exception:
            self.assertTrue(False)

        try:
            test_data_set = np.random.rand(1000,1)
            (X_cs, X_center, X_scale) = preprocess.center_scale(test_data_set, '0to1', nocenter=False)
            for i in range(0,1):
                self.assertTrue((np.max(X_cs[:,i]) > (1 - tolerance)) and (np.max(X_cs[:,i]) < (1 + tolerance)))
        except Exception:
            self.assertTrue(False)

        try:
            test_data_set = np.random.rand(2000,1)
            (X_cs, X_center, X_scale) = preprocess.center_scale(test_data_set, '0to1', nocenter=False)
            for i in range(0,1):
                self.assertTrue((np.max(X_cs[:,i]) > (1 - tolerance)) and (np.max(X_cs[:,i]) < (1 + tolerance)))
        except Exception:
            self.assertTrue(False)

# ------------------------------------------------------------------------------

    def test_preprocess__center_scale__MinusOneToOne(self):

        tolerance = 10**-10

        try:
            test_data_set = np.random.rand(100,10)
            (X_cs, X_center, X_scale) = preprocess.center_scale(test_data_set, '-1to1', nocenter=False)
            for i in range(0,10):
                self.assertTrue((np.min(X_cs[:,i]) > (-1 - tolerance)) and (np.min(X_cs[:,i]) < -1 + tolerance))
        except Exception:
            self.assertTrue(False)

        try:
            test_data_set = np.random.rand(1000,1)
            (X_cs, X_center, X_scale) = preprocess.center_scale(test_data_set, '-1to1', nocenter=False)
            for i in range(0,1):
                self.assertTrue((np.min(X_cs[:,i]) > (-1 - tolerance)) and (np.min(X_cs[:,i]) < -1 + tolerance))
        except Exception:
            self.assertTrue(False)

        try:
            test_data_set = np.random.rand(2000,1)
            (X_cs, X_center, X_scale) = preprocess.center_scale(test_data_set, '-1to1', nocenter=False)
            for i in range(0,1):
                self.assertTrue((np.min(X_cs[:,i]) > (-1 - tolerance)) and (np.min(X_cs[:,i]) < -1 + tolerance))
        except Exception:
            self.assertTrue(False)

        try:
            test_data_set = np.random.rand(100,10)
            (X_cs, X_center, X_scale) = preprocess.center_scale(test_data_set, '-1to1', nocenter=False)
            for i in range(0,10):
                self.assertTrue((np.max(X_cs[:,i]) > (1 - tolerance)) and (np.max(X_cs[:,i]) < (1 + tolerance)))
        except Exception:
            self.assertTrue(False)

        try:
            test_data_set = np.random.rand(1000,1)
            (X_cs, X_center, X_scale) = preprocess.center_scale(test_data_set, '-1to1', nocenter=False)
            for i in range(0,1):
                self.assertTrue((np.max(X_cs[:,i]) > (1 - tolerance)) and (np.max(X_cs[:,i]) < (1 + tolerance)))
        except Exception:
            self.assertTrue(False)

        try:
            test_data_set = np.random.rand(2000,1)
            (X_cs, X_center, X_scale) = preprocess.center_scale(test_data_set, '-1to1', nocenter=False)
            for i in range(0,1):
                self.assertTrue((np.max(X_cs[:,i]) > (1 - tolerance)) and (np.max(X_cs[:,i]) < (1 + tolerance)))
        except Exception:
            self.assertTrue(False)

# ------------------------------------------------------------------------------

    def test_preprocess__center_scale__C_and_D_properties(self):

        # This function tests if known properties of centers or scales hold:
        test_data_set = np.random.rand(100,20)
        means = np.mean(test_data_set, axis=0)
        stds = np.std(test_data_set, axis=0)
        zeros = np.zeros((20,))
        ones = np.ones((20,))

        try:
            (X_cs, X_center, X_scale) = preprocess.center_scale(test_data_set, 'std', nocenter=False)
            comparison = X_center == means
            self.assertTrue(comparison.all())
        except Exception:
            self.assertTrue(False)

        try:
            (X_cs, X_center, X_scale) = preprocess.center_scale(test_data_set, 'std', nocenter=True)
            difference = abs(X_scale - stds)
            comparison = difference < 10**(-14)
            self.assertTrue(comparison.all())
            comparison = X_center == zeros
            self.assertTrue(comparison.all())
        except Exception:
            self.assertTrue(False)

        try:
            (X_cs, X_center, X_scale) = preprocess.center_scale(test_data_set, 'none', nocenter=False)
            comparison = X_scale == ones
            self.assertTrue(comparison.all())
        except Exception:
            self.assertTrue(False)

        test_data_set = np.random.rand(100,1)

        try:
            (X_cs, X_center, X_scale) = preprocess.center_scale(test_data_set, 'none', nocenter=False)
            (n_observations, n_variables) = X_cs.shape
            (n_centers,) = X_center.shape
            (n_scales,) = X_scale.shape
            self.assertTrue(n_observations==100)
            self.assertTrue(n_variables==1)
            self.assertTrue(n_centers==1)
            self.assertTrue(n_scales==1)
            self.assertTrue(isinstance(X_cs, np.ndarray))
            self.assertTrue(isinstance(X_center, np.ndarray))
            self.assertTrue(isinstance(X_scale, np.ndarray))
        except:
            self.assertTrue(False)

        test_data_set = np.random.rand(100,2)

        try:
            (X_cs, X_center, X_scale) = preprocess.center_scale(test_data_set, 'none', nocenter=False)
            (n_observations, n_variables) = X_cs.shape
            (n_centers,) = X_center.shape
            (n_scales,) = X_scale.shape
            self.assertTrue(n_observations==100)
            self.assertTrue(n_variables==2)
            self.assertTrue(n_centers==2)
            self.assertTrue(n_scales==2)
            self.assertTrue(isinstance(X_cs, np.ndarray))
            self.assertTrue(isinstance(X_center, np.ndarray))
            self.assertTrue(isinstance(X_scale, np.ndarray))
        except:
            self.assertTrue(False)

# ------------------------------------------------------------------------------

    def test_preprocess__center_scale__data_with_constant_nonzero_variables(self):

        X = np.random.rand(100,20)
        X[:,10] = np.ones((100,))

        # Scalings ['none', '', 'max', 'level', 'median', 'poisson'] accept constant, non-zero variables in a matrix:
        try:
            (X_cs, X_center, X_scale) = preprocess.center_scale(X, 'none')
            (X_cs, X_center, X_scale) = preprocess.center_scale(X, '')
            (X_cs, X_center, X_scale) = preprocess.center_scale(X, 'max')
            (X_cs, X_center, X_scale) = preprocess.center_scale(X, 'level')
            (X_cs, X_center, X_scale) = preprocess.center_scale(X, 'median')
            (X_cs, X_center, X_scale) = preprocess.center_scale(X, 'poisson')
        except:
            self.assertTrue(False)

        # Other scalings result in division by zero:
        with self.assertRaises(ValueError):
            (X_cs, X_center, X_scale) = preprocess.center_scale(X, 'auto')
        with self.assertRaises(ValueError):
            (X_cs, X_center, X_scale) = preprocess.center_scale(X, 'std')
        with self.assertRaises(ValueError):
            (X_cs, X_center, X_scale) = preprocess.center_scale(X, 'pareto')
        with self.assertRaises(ValueError):
            (X_cs, X_center, X_scale) = preprocess.center_scale(X, 'vast')
        with self.assertRaises(ValueError):
            (X_cs, X_center, X_scale) = preprocess.center_scale(X, 'range')
        with self.assertRaises(ValueError):
            (X_cs, X_center, X_scale) = preprocess.center_scale(X, '0to1')
        with self.assertRaises(ValueError):
            (X_cs, X_center, X_scale) = preprocess.center_scale(X, '-1to1')
        with self.assertRaises(ValueError):
            (X_cs, X_center, X_scale) = preprocess.center_scale(X, 'variance')
        with self.assertRaises(ValueError):
            (X_cs, X_center, X_scale) = preprocess.center_scale(X, 'vast_2')
        with self.assertRaises(ValueError):
            (X_cs, X_center, X_scale) = preprocess.center_scale(X, 'vast_3')
        with self.assertRaises(ValueError):
            (X_cs, X_center, X_scale) = preprocess.center_scale(X, 'vast_4')

# ------------------------------------------------------------------------------

    def test_preprocess__center_scale__data_with_zeroed_variables(self):

        X = np.random.rand(100,20)
        X[:,10] = np.zeros((100,))

        # Scalings ['none', ''] accept zeroed variables in a matrix:
        try:
            (X_cs, X_center, X_scale) = preprocess.center_scale(X, 'none')
            (X_cs, X_center, X_scale) = preprocess.center_scale(X, '')
        except:
            self.assertTrue(False)

        # Other scalings result in division by zero:
        with self.assertRaises(ValueError):
            (X_cs, X_center, X_scale) = preprocess.center_scale(X, 'auto')
        with self.assertRaises(ValueError):
            (X_cs, X_center, X_scale) = preprocess.center_scale(X, 'std')
        with self.assertRaises(ValueError):
            (X_cs, X_center, X_scale) = preprocess.center_scale(X, 'pareto')
        with self.assertRaises(ValueError):
            (X_cs, X_center, X_scale) = preprocess.center_scale(X, 'vast')
        with self.assertRaises(ValueError):
            (X_cs, X_center, X_scale) = preprocess.center_scale(X, 'range')
        with self.assertRaises(ValueError):
            (X_cs, X_center, X_scale) = preprocess.center_scale(X, '0to1')
        with self.assertRaises(ValueError):
            (X_cs, X_center, X_scale) = preprocess.center_scale(X, '-1to1')
        with self.assertRaises(ValueError):
            (X_cs, X_center, X_scale) = preprocess.center_scale(X, 'level')
        with self.assertRaises(ValueError):
            (X_cs, X_center, X_scale) = preprocess.center_scale(X, 'max')
        with self.assertRaises(ValueError):
            (X_cs, X_center, X_scale) = preprocess.center_scale(X, 'variance')
        with self.assertRaises(ValueError):
            (X_cs, X_center, X_scale) = preprocess.center_scale(X, 'median')
        with self.assertRaises(ValueError):
            (X_cs, X_center, X_scale) = preprocess.center_scale(X, 'poisson')
        with self.assertRaises(ValueError):
            (X_cs, X_center, X_scale) = preprocess.center_scale(X, 'vast_2')
        with self.assertRaises(ValueError):
            (X_cs, X_center, X_scale) = preprocess.center_scale(X, 'vast_3')
        with self.assertRaises(ValueError):
            (X_cs, X_center, X_scale) = preprocess.center_scale(X, 'vast_4')

# ------------------------------------------------------------------------------
