import unittest
import numpy as np
from PCAfold import preprocess
from PCAfold import reduction
from PCAfold import PCA
from PCAfold import PreProcessing
from PCAfold import KernelDensity


class TestManipulation(unittest.TestCase):

    def test_center_scale_not_allowed_calls(self):

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

    def test_center_scale_all_possible_C_and_D(self):

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
            (X_cs, X_center, X_scale) = preprocess.center_scale(test_data_set, 'poisson', nocenter=True)
            (X_cs, X_center, X_scale) = preprocess.center_scale(test_data_set, 'vast_2', nocenter=True)
            (X_cs, X_center, X_scale) = preprocess.center_scale(test_data_set, 'vast_3', nocenter=True)
            (X_cs, X_center, X_scale) = preprocess.center_scale(test_data_set, 'vast_4', nocenter=True)
        except Exception:
            self.assertTrue(False)

    def test_center_scale_on_1D_variable(self):

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
            (X_cs, X_center, X_scale) = preprocess.center_scale(test_1D_variable, 'vast_2', nocenter=True)
            (X_cs, X_center, X_scale) = preprocess.center_scale(test_1D_variable, 'vast_3', nocenter=True)
            (X_cs, X_center, X_scale) = preprocess.center_scale(test_1D_variable, 'vast_4', nocenter=True)
        except Exception:
            self.assertTrue(False)

    def test_center_scale_ZeroToOne(self):

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

    def test_center_scale_MinusOneToOne(self):

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

    def test_center_scale_C_and_D_properties(self):

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

    def test_invert_center_scale(self):
        # This function tests all possible inversions of center_scale function:
        test_data_set = np.random.rand(200,20)

        try:
            (X_cs, X_center, X_scale) = preprocess.center_scale(test_data_set, 'none', nocenter=False)
            X = preprocess.invert_center_scale(X_cs, X_center, X_scale)
            difference = abs(X - test_data_set)
            comparison = difference < 10**(-14)
            self.assertTrue(comparison.all())
            (X_cs, X_center, X_scale) = preprocess.center_scale(test_data_set, 'auto', nocenter=False)
            X = preprocess.invert_center_scale(X_cs, X_center, X_scale)
            difference = abs(X - test_data_set)
            comparison = difference < 10**(-14)
            self.assertTrue(comparison.all())
            (X_cs, X_center, X_scale) = preprocess.center_scale(test_data_set, 'std', nocenter=False)
            X = preprocess.invert_center_scale(X_cs, X_center, X_scale)
            difference = abs(X - test_data_set)
            comparison = difference < 10**(-14)
            self.assertTrue(comparison.all())
            (X_cs, X_center, X_scale) = preprocess.center_scale(test_data_set, 'pareto', nocenter=False)
            X = preprocess.invert_center_scale(X_cs, X_center, X_scale)
            difference = abs(X - test_data_set)
            comparison = difference < 10**(-14)
            self.assertTrue(comparison.all())
            (X_cs, X_center, X_scale) = preprocess.center_scale(test_data_set, 'vast', nocenter=False)
            X = preprocess.invert_center_scale(X_cs, X_center, X_scale)
            difference = abs(X - test_data_set)
            comparison = difference < 10**(-14)
            self.assertTrue(comparison.all())
            (X_cs, X_center, X_scale) = preprocess.center_scale(test_data_set, 'range', nocenter=False)
            X = preprocess.invert_center_scale(X_cs, X_center, X_scale)
            difference = abs(X - test_data_set)
            comparison = difference < 10**(-14)
            self.assertTrue(comparison.all())
            (X_cs, X_center, X_scale) = preprocess.center_scale(test_data_set, '0to1', nocenter=False)
            X = preprocess.invert_center_scale(X_cs, X_center, X_scale)
            difference = abs(X - test_data_set)
            comparison = difference < 10**(-14)
            self.assertTrue(comparison.all())
            (X_cs, X_center, X_scale) = preprocess.center_scale(test_data_set, '-1to1', nocenter=False)
            X = preprocess.invert_center_scale(X_cs, X_center, X_scale)
            difference = abs(X - test_data_set)
            comparison = difference < 10**(-14)
            self.assertTrue(comparison.all())
            (X_cs, X_center, X_scale) = preprocess.center_scale(test_data_set, 'level', nocenter=False)
            X = preprocess.invert_center_scale(X_cs, X_center, X_scale)
            difference = abs(X - test_data_set)
            comparison = difference < 10**(-14)
            self.assertTrue(comparison.all())
            (X_cs, X_center, X_scale) = preprocess.center_scale(test_data_set, 'max', nocenter=False)
            X = preprocess.invert_center_scale(X_cs, X_center, X_scale)
            difference = abs(X - test_data_set)
            comparison = difference < 10**(-14)
            self.assertTrue(comparison.all())
            (X_cs, X_center, X_scale) = preprocess.center_scale(test_data_set, 'poisson', nocenter=False)
            X = preprocess.invert_center_scale(X_cs, X_center, X_scale)
            difference = abs(X - test_data_set)
            comparison = difference < 10**(-14)
            self.assertTrue(comparison.all())
            (X_cs, X_center, X_scale) = preprocess.center_scale(test_data_set, 'vast_2', nocenter=False)
            X = preprocess.invert_center_scale(X_cs, X_center, X_scale)
            difference = abs(X - test_data_set)
            comparison = difference < 10**(-14)
            self.assertTrue(comparison.all())
            (X_cs, X_center, X_scale) = preprocess.center_scale(test_data_set, 'vast_3', nocenter=False)
            X = preprocess.invert_center_scale(X_cs, X_center, X_scale)
            difference = abs(X - test_data_set)
            comparison = difference < 10**(-14)
            self.assertTrue(comparison.all())
            (X_cs, X_center, X_scale) = preprocess.center_scale(test_data_set, 'vast_4', nocenter=False)
            X = preprocess.invert_center_scale(X_cs, X_center, X_scale)
            difference = abs(X - test_data_set)
            comparison = difference < 10**(-14)
            self.assertTrue(comparison.all())
        except Exception:
            self.assertTrue(False)

        try:
            (X_cs, X_center, X_scale) = preprocess.center_scale(test_data_set, 'none', nocenter=True)
            X = preprocess.invert_center_scale(X_cs, X_center, X_scale)
            difference = abs(X - test_data_set)
            comparison = difference < 10**(-14)
            self.assertTrue(comparison.all())
            (X_cs, X_center, X_scale) = preprocess.center_scale(test_data_set, 'auto', nocenter=True)
            X = preprocess.invert_center_scale(X_cs, X_center, X_scale)
            difference = abs(X - test_data_set)
            comparison = difference < 10**(-14)
            self.assertTrue(comparison.all())
            (X_cs, X_center, X_scale) = preprocess.center_scale(test_data_set, 'std', nocenter=True)
            X = preprocess.invert_center_scale(X_cs, X_center, X_scale)
            difference = abs(X - test_data_set)
            comparison = difference < 10**(-14)
            self.assertTrue(comparison.all())
            (X_cs, X_center, X_scale) = preprocess.center_scale(test_data_set, 'pareto', nocenter=True)
            X = preprocess.invert_center_scale(X_cs, X_center, X_scale)
            difference = abs(X - test_data_set)
            comparison = difference < 10**(-14)
            self.assertTrue(comparison.all())
            (X_cs, X_center, X_scale) = preprocess.center_scale(test_data_set, 'vast', nocenter=True)
            X = preprocess.invert_center_scale(X_cs, X_center, X_scale)
            difference = abs(X - test_data_set)
            comparison = difference < 10**(-14)
            self.assertTrue(comparison.all())
            (X_cs, X_center, X_scale) = preprocess.center_scale(test_data_set, 'range', nocenter=True)
            X = preprocess.invert_center_scale(X_cs, X_center, X_scale)
            difference = abs(X - test_data_set)
            comparison = difference < 10**(-14)
            self.assertTrue(comparison.all())
            (X_cs, X_center, X_scale) = preprocess.center_scale(test_data_set, '0to1', nocenter=True)
            X = preprocess.invert_center_scale(X_cs, X_center, X_scale)
            difference = abs(X - test_data_set)
            comparison = difference < 10**(-14)
            self.assertTrue(comparison.all())
            (X_cs, X_center, X_scale) = preprocess.center_scale(test_data_set, '-1to1', nocenter=True)
            X = preprocess.invert_center_scale(X_cs, X_center, X_scale)
            difference = abs(X - test_data_set)
            comparison = difference < 10**(-14)
            self.assertTrue(comparison.all())
            (X_cs, X_center, X_scale) = preprocess.center_scale(test_data_set, 'level', nocenter=True)
            X = preprocess.invert_center_scale(X_cs, X_center, X_scale)
            difference = abs(X - test_data_set)
            comparison = difference < 10**(-14)
            self.assertTrue(comparison.all())
            (X_cs, X_center, X_scale) = preprocess.center_scale(test_data_set, 'max', nocenter=True)
            X = preprocess.invert_center_scale(X_cs, X_center, X_scale)
            difference = abs(X - test_data_set)
            comparison = difference < 10**(-14)
            self.assertTrue(comparison.all())
            (X_cs, X_center, X_scale) = preprocess.center_scale(test_data_set, 'poisson', nocenter=True)
            X = preprocess.invert_center_scale(X_cs, X_center, X_scale)
            difference = abs(X - test_data_set)
            comparison = difference < 10**(-14)
            self.assertTrue(comparison.all())
            (X_cs, X_center, X_scale) = preprocess.center_scale(test_data_set, 'vast_2', nocenter=True)
            X = preprocess.invert_center_scale(X_cs, X_center, X_scale)
            difference = abs(X - test_data_set)
            comparison = difference < 10**(-14)
            self.assertTrue(comparison.all())
            (X_cs, X_center, X_scale) = preprocess.center_scale(test_data_set, 'vast_3', nocenter=True)
            X = preprocess.invert_center_scale(X_cs, X_center, X_scale)
            difference = abs(X - test_data_set)
            comparison = difference < 10**(-14)
            self.assertTrue(comparison.all())
            (X_cs, X_center, X_scale) = preprocess.center_scale(test_data_set, 'vast_4', nocenter=True)
            X = preprocess.invert_center_scale(X_cs, X_center, X_scale)
            difference = abs(X - test_data_set)
            comparison = difference < 10**(-14)
            self.assertTrue(comparison.all())
        except Exception:
            self.assertTrue(False)

    def test_invert_center_scale_not_allowed_calls(self):

        X_cs = np.random.rand(100,)
        X_center = np.array([0])
        X_scale = np.array([1])

        with self.assertRaises(ValueError):
            X = preprocess.invert_center_scale(X_cs, X_center, X_scale)

        with self.assertRaises(ValueError):
            X = preprocess.invert_center_scale([1,2,3], X_center, X_scale)

        X_cs = np.random.rand(100,2)

        with self.assertRaises(ValueError):
            X = preprocess.invert_center_scale(X_cs, X_center, X_scale)

        X_scale = np.array([1,2])

        with self.assertRaises(ValueError):
            X = preprocess.invert_center_scale(X_cs, [0,0], X_scale)

        X_center = np.array([1,2])

        with self.assertRaises(ValueError):
            X = preprocess.invert_center_scale(X_cs, X_center, [1,1])

        X_cs = np.random.rand(100,10)
        X_center = np.array([0,0,0,0])
        X_scale = np.array([1,1,2,2])

        with self.assertRaises(ValueError):
            X = preprocess.invert_center_scale(X_cs, X_center, X_scale)

    def test_invert_center_scale_on_1D_variable(self):
        # This function tests all possible inversions of center_scale function:
        test_data_set = np.random.rand(200,1)

        try:
            (X_cs, X_center, X_scale) = preprocess.center_scale(test_data_set, 'none', nocenter=False)
            X = preprocess.invert_center_scale(X_cs, X_center, X_scale)
            difference = abs(X - test_data_set)
            comparison = difference < 10**(-14)
            self.assertTrue(comparison.all())
            (X_cs, X_center, X_scale) = preprocess.center_scale(test_data_set, 'auto', nocenter=False)
            X = preprocess.invert_center_scale(X_cs, X_center, X_scale)
            difference = abs(X - test_data_set)
            comparison = difference < 10**(-14)
            self.assertTrue(comparison.all())
            (X_cs, X_center, X_scale) = preprocess.center_scale(test_data_set, 'std', nocenter=False)
            X = preprocess.invert_center_scale(X_cs, X_center, X_scale)
            difference = abs(X - test_data_set)
            comparison = difference < 10**(-14)
            self.assertTrue(comparison.all())
            (X_cs, X_center, X_scale) = preprocess.center_scale(test_data_set, 'pareto', nocenter=False)
            X = preprocess.invert_center_scale(X_cs, X_center, X_scale)
            difference = abs(X - test_data_set)
            comparison = difference < 10**(-14)
            self.assertTrue(comparison.all())
            (X_cs, X_center, X_scale) = preprocess.center_scale(test_data_set, 'vast', nocenter=False)
            X = preprocess.invert_center_scale(X_cs, X_center, X_scale)
            difference = abs(X - test_data_set)
            comparison = difference < 10**(-14)
            self.assertTrue(comparison.all())
            (X_cs, X_center, X_scale) = preprocess.center_scale(test_data_set, 'range', nocenter=False)
            X = preprocess.invert_center_scale(X_cs, X_center, X_scale)
            difference = abs(X - test_data_set)
            comparison = difference < 10**(-14)
            self.assertTrue(comparison.all())
            (X_cs, X_center, X_scale) = preprocess.center_scale(test_data_set, '0to1', nocenter=False)
            X = preprocess.invert_center_scale(X_cs, X_center, X_scale)
            difference = abs(X - test_data_set)
            comparison = difference < 10**(-14)
            self.assertTrue(comparison.all())
            (X_cs, X_center, X_scale) = preprocess.center_scale(test_data_set, '-1to1', nocenter=False)
            X = preprocess.invert_center_scale(X_cs, X_center, X_scale)
            difference = abs(X - test_data_set)
            comparison = difference < 10**(-14)
            self.assertTrue(comparison.all())
            (X_cs, X_center, X_scale) = preprocess.center_scale(test_data_set, 'level', nocenter=False)
            X = preprocess.invert_center_scale(X_cs, X_center, X_scale)
            difference = abs(X - test_data_set)
            comparison = difference < 10**(-14)
            self.assertTrue(comparison.all())
            (X_cs, X_center, X_scale) = preprocess.center_scale(test_data_set, 'max', nocenter=False)
            X = preprocess.invert_center_scale(X_cs, X_center, X_scale)
            difference = abs(X - test_data_set)
            comparison = difference < 10**(-14)
            self.assertTrue(comparison.all())
            (X_cs, X_center, X_scale) = preprocess.center_scale(test_data_set, 'poisson', nocenter=False)
            X = preprocess.invert_center_scale(X_cs, X_center, X_scale)
            difference = abs(X - test_data_set)
            comparison = difference < 10**(-14)
            self.assertTrue(comparison.all())
            (X_cs, X_center, X_scale) = preprocess.center_scale(test_data_set, 'vast_2', nocenter=False)
            X = preprocess.invert_center_scale(X_cs, X_center, X_scale)
            difference = abs(X - test_data_set)
            comparison = difference < 10**(-14)
            self.assertTrue(comparison.all())
            (X_cs, X_center, X_scale) = preprocess.center_scale(test_data_set, 'vast_3', nocenter=False)
            X = preprocess.invert_center_scale(X_cs, X_center, X_scale)
            difference = abs(X - test_data_set)
            comparison = difference < 10**(-14)
            self.assertTrue(comparison.all())
            (X_cs, X_center, X_scale) = preprocess.center_scale(test_data_set, 'vast_4', nocenter=False)
            X = preprocess.invert_center_scale(X_cs, X_center, X_scale)
            difference = abs(X - test_data_set)
            comparison = difference < 10**(-14)
            self.assertTrue(comparison.all())
        except Exception:
            self.assertTrue(False)

        try:
            (X_cs, X_center, X_scale) = preprocess.center_scale(test_data_set, 'none', nocenter=True)
            X = preprocess.invert_center_scale(X_cs, X_center, X_scale)
            difference = abs(X - test_data_set)
            comparison = difference < 10**(-14)
            self.assertTrue(comparison.all())
            (X_cs, X_center, X_scale) = preprocess.center_scale(test_data_set, 'auto', nocenter=True)
            X = preprocess.invert_center_scale(X_cs, X_center, X_scale)
            difference = abs(X - test_data_set)
            comparison = difference < 10**(-14)
            self.assertTrue(comparison.all())
            (X_cs, X_center, X_scale) = preprocess.center_scale(test_data_set, 'std', nocenter=True)
            X = preprocess.invert_center_scale(X_cs, X_center, X_scale)
            difference = abs(X - test_data_set)
            comparison = difference < 10**(-14)
            self.assertTrue(comparison.all())
            (X_cs, X_center, X_scale) = preprocess.center_scale(test_data_set, 'pareto', nocenter=True)
            X = preprocess.invert_center_scale(X_cs, X_center, X_scale)
            difference = abs(X - test_data_set)
            comparison = difference < 10**(-14)
            self.assertTrue(comparison.all())
            (X_cs, X_center, X_scale) = preprocess.center_scale(test_data_set, 'vast', nocenter=True)
            X = preprocess.invert_center_scale(X_cs, X_center, X_scale)
            difference = abs(X - test_data_set)
            comparison = difference < 10**(-14)
            self.assertTrue(comparison.all())
            (X_cs, X_center, X_scale) = preprocess.center_scale(test_data_set, 'range', nocenter=True)
            X = preprocess.invert_center_scale(X_cs, X_center, X_scale)
            difference = abs(X - test_data_set)
            comparison = difference < 10**(-14)
            self.assertTrue(comparison.all())
            (X_cs, X_center, X_scale) = preprocess.center_scale(test_data_set, '0to1', nocenter=True)
            X = preprocess.invert_center_scale(X_cs, X_center, X_scale)
            difference = abs(X - test_data_set)
            comparison = difference < 10**(-14)
            self.assertTrue(comparison.all())
            (X_cs, X_center, X_scale) = preprocess.center_scale(test_data_set, '-1to1', nocenter=True)
            X = preprocess.invert_center_scale(X_cs, X_center, X_scale)
            difference = abs(X - test_data_set)
            comparison = difference < 10**(-14)
            self.assertTrue(comparison.all())
            (X_cs, X_center, X_scale) = preprocess.center_scale(test_data_set, 'level', nocenter=True)
            X = preprocess.invert_center_scale(X_cs, X_center, X_scale)
            difference = abs(X - test_data_set)
            comparison = difference < 10**(-14)
            self.assertTrue(comparison.all())
            (X_cs, X_center, X_scale) = preprocess.center_scale(test_data_set, 'max', nocenter=True)
            X = preprocess.invert_center_scale(X_cs, X_center, X_scale)
            difference = abs(X - test_data_set)
            comparison = difference < 10**(-14)
            self.assertTrue(comparison.all())
            (X_cs, X_center, X_scale) = preprocess.center_scale(test_data_set, 'poisson', nocenter=True)
            X = preprocess.invert_center_scale(X_cs, X_center, X_scale)
            difference = abs(X - test_data_set)
            comparison = difference < 10**(-14)
            self.assertTrue(comparison.all())
            (X_cs, X_center, X_scale) = preprocess.center_scale(test_data_set, 'vast_2', nocenter=True)
            X = preprocess.invert_center_scale(X_cs, X_center, X_scale)
            difference = abs(X - test_data_set)
            comparison = difference < 10**(-14)
            self.assertTrue(comparison.all())
            (X_cs, X_center, X_scale) = preprocess.center_scale(test_data_set, 'vast_3', nocenter=True)
            X = preprocess.invert_center_scale(X_cs, X_center, X_scale)
            difference = abs(X - test_data_set)
            comparison = difference < 10**(-14)
            self.assertTrue(comparison.all())
            (X_cs, X_center, X_scale) = preprocess.center_scale(test_data_set, 'vast_4', nocenter=True)
            X = preprocess.invert_center_scale(X_cs, X_center, X_scale)
            difference = abs(X - test_data_set)
            comparison = difference < 10**(-14)
            self.assertTrue(comparison.all())
        except Exception:
            self.assertTrue(False)

    def test_invert_center_scale_single_variable(self):

        try:
            test_data_set = np.ones((200,1))
            X_result = 2*np.ones((200,1))
            X = preprocess.invert_center_scale(test_data_set, np.array([0]), np.array([2]))
            comparison = X == X_result
            self.assertTrue(comparison.all())
        except Exception:
            self.assertTrue(False)

        try:
            test_data_set = np.ones((200,1))
            X_result = 3*np.ones((200,1))
            X = preprocess.invert_center_scale(test_data_set, np.array([1]), np.array([2]))
            comparison = X == X_result
            self.assertTrue(comparison.all())
        except Exception:
            self.assertTrue(False)

        try:
            test_data_set = np.ones((200,1))
            X_result = 2*np.ones((200,1))
            X = preprocess.invert_center_scale(test_data_set, np.array([0]), np.array([2]))
            comparison = X == X_result
            self.assertTrue(comparison.all())
        except Exception:
            self.assertTrue(False)

        try:
            test_data_set = np.ones((200,1))
            X_result = 3*np.ones((200,1))
            X = preprocess.invert_center_scale(test_data_set, np.array([1]), np.array([2]))
            comparison = X == X_result
            self.assertTrue(comparison.all())
        except Exception:
            self.assertTrue(False)

    def test_remove_constant_vars(self):

        test_data_set = np.random.rand(100,20)

        try:
            # Inject two constant columns:
            test_data_set_constant = np.hstack((test_data_set[:,0:3], 2.4*np.ones((100,1)), test_data_set[:,3:15], -8.1*np.ones((100,1)), test_data_set[:,15::]))
            idx_removed_check = [3,16]
            idx_retained_check = [0,1,2,4,5,6,7,8,9,10,11,12,13,14,15,17,18,19,20,21]
            (X_removed, idx_removed, idx_retained) = preprocess.remove_constant_vars(test_data_set_constant)
            comparison = X_removed == test_data_set
            self.assertTrue(comparison.all())
            self.assertTrue(idx_removed == idx_removed_check)
            self.assertTrue(idx_retained == idx_retained_check)
        except Exception:
            self.assertTrue(False)

        try:
            # Inject a constant column that has values close to zero:
            close_to_zero_column = -10**(-14)*np.ones((100,1))
            close_to_zero_column[20:30,:] = -10**(-13)
            close_to_zero_column[80:85,:] = -10**(-14)
            test_data_set_constant = np.hstack((test_data_set[:,0:3], close_to_zero_column, test_data_set[:,3::]))
            idx_removed_check = [3]
            idx_retained_check = [0,1,2,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
            (X_removed, idx_removed, idx_retained) = preprocess.remove_constant_vars(test_data_set_constant)
            comparison = X_removed == test_data_set
            self.assertTrue(comparison.all())
            self.assertTrue(idx_removed == idx_removed_check)
            self.assertTrue(idx_retained == idx_retained_check)
        except Exception:
            self.assertTrue(False)

        try:
            # Inject a constant column that has values close to zero:
            close_to_zero_column = -10**(-14)*np.ones((100,1))
            close_to_zero_column[20:30,:] = 10**(-13)
            close_to_zero_column[80:85,:] = 10**(-14)
            test_data_set_constant = np.hstack((test_data_set[:,0:3], close_to_zero_column, test_data_set[:,3::]))
            idx_removed_check = [3]
            idx_retained_check = [0,1,2,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
            (X_removed, idx_removed, idx_retained) = preprocess.remove_constant_vars(test_data_set_constant)
            comparison = X_removed == test_data_set
            self.assertTrue(comparison.all())
            self.assertTrue(idx_removed == idx_removed_check)
            self.assertTrue(idx_retained == idx_retained_check)
        except Exception:
            self.assertTrue(False)

    def test_remove_constant_vars_allowed_calls(self):

        X = np.random.rand(100,20)

        try:
            (X_removed, idx_removed, idx_retained) = preprocess.remove_constant_vars(X)
        except Exception:
            self.assertTrue(False)

        X = np.random.rand(100,1)

        try:
            (X_removed, idx_removed, idx_retained) = preprocess.remove_constant_vars(X)
        except Exception:
            self.assertTrue(False)

    def test_remove_constant_vars_not_allowed_calls(self):

        X = np.random.rand(100,)

        with self.assertRaises(ValueError):
            (X_removed, idx_removed, idx_retained) = preprocess.remove_constant_vars(X)

        with self.assertRaises(ValueError):
            (X_removed, idx_removed, idx_retained) = preprocess.remove_constant_vars([1,2,3,4,5])

        X = np.random.rand(100,20)

        with self.assertRaises(ValueError):
            (X_removed, idx_removed, idx_retained) = preprocess.remove_constant_vars(X, maxtol='none')

        with self.assertRaises(ValueError):
            (X_removed, idx_removed, idx_retained) = preprocess.remove_constant_vars(X, rangetol='none')

    def test_order_variables_allowed_calls(self):

        X = np.array([[100, 1, 10, 0.1],
                      [200, 2, 20, 0.2],
                      [300, 3, 30, 0.3]])

        X_descending = np.array([[100, 10, 1, 0.1],
                                 [200, 20, 2, 0.2],
                                 [300, 30, 3, 0.3]])

        idx_descending = [0, 2, 1, 3]

        X_ascending = np.array([[0.1, 1, 10, 100],
                                [0.2, 2, 20, 200],
                                [0.3, 3, 30, 300]])

        idx_ascending = [3, 1, 2, 0]

        try:
            (X_ordered, idx) = preprocess.order_variables(X, method='mean', descending=True)
            self.assertTrue(np.array_equal(X_ordered, X_descending))
            self.assertTrue(np.array_equal(idx, idx_descending))
        except Exception:
            self.assertTrue(False)

        try:
            (X_ordered, idx) = preprocess.order_variables(X, method='mean', descending=False)
            self.assertTrue(np.array_equal(X_ordered, X_ascending))
            self.assertTrue(np.array_equal(idx, idx_ascending))
        except Exception:
            self.assertTrue(False)

        try:
            (X_ordered, idx) = preprocess.order_variables(X, method='min', descending=False)
            (X_ordered, idx) = preprocess.order_variables(X, method='max', descending=False)
            (X_ordered, idx) = preprocess.order_variables(X, method='std', descending=False)
            (X_ordered, idx) = preprocess.order_variables(X, method='var', descending=False)
        except Exception:
            self.assertTrue(False)

        X = np.ones((10,1))

        try:
            (X_ordered, idx) = preprocess.order_variables(X, method='mean', descending=False)
            (X_ordered, idx) = preprocess.order_variables(X, method='min', descending=False)
            (X_ordered, idx) = preprocess.order_variables(X, method='max', descending=False)
            (X_ordered, idx) = preprocess.order_variables(X, method='std', descending=False)
            (X_ordered, idx) = preprocess.order_variables(X, method='var', descending=False)
        except Exception:
            self.assertTrue(False)

    def test_order_variables_not_allowed_calls(self):

        X = np.array([[100, 1, 10, 0.1],
                      [200, 2, 20, 0.2],
                      [300, 3, 30, 0.3]])

        with self.assertRaises(ValueError):
            (X_ordered, idx) = preprocess.order_variables(X, method='mean', descending=1)

        with self.assertRaises(ValueError):
            (X_ordered, idx) = preprocess.order_variables(X, method='none', descending=True)

        with self.assertRaises(ValueError):
            (X_ordered, idx) = preprocess.order_variables(X, method=1, descending=True)

        with self.assertRaises(ValueError):
            (X_ordered, idx) = preprocess.order_variables([1,2,3], method='mean', descending=True)

        X = np.array([1,10,100])

        with self.assertRaises(ValueError):
            (X_ordered, idx) = preprocess.order_variables(X, method='mean', descending=True)

    def test_PreProcessing(self):

        test_data_set = np.random.rand(100,20)

        # Inject two constant columns:
        test_data_set_constant = np.hstack((test_data_set[:,0:3], 2.4*np.ones((100,1)), test_data_set[:,3:15], -8.1*np.ones((100,1)), test_data_set[:,15::]))
        idx_removed_check = [3,16]
        idx_retained_check = [0,1,2,4,5,6,7,8,9,10,11,12,13,14,15,17,18,19,20,21]

        try:
            preprocessed = PreProcessing(test_data_set_constant, scaling='none', nocenter=True)
            comparison = preprocessed.X_removed == test_data_set
            self.assertTrue(comparison.all())
            self.assertTrue(preprocessed.idx_removed == idx_removed_check)
            self.assertTrue(preprocessed.idx_retained == idx_retained_check)
            self.assertTrue(np.shape(preprocessed.X_cs) == (100,20))
        except Exception:
            self.assertTrue(False)

    def test_PreProcessing_not_allowed_attribute_setting(self):

        test_data_set = np.random.rand(100,20)
        pp = PreProcessing(test_data_set, scaling='auto')

        with self.assertRaises(AttributeError):
            pp.X_cs = 1
        with self.assertRaises(AttributeError):
            pp.X_center = 1
        with self.assertRaises(AttributeError):
            pp.X_scale = 1
        with self.assertRaises(AttributeError):
            pp.X_removed = 1
        with self.assertRaises(AttributeError):
            pp.idx_removed = 1
        with self.assertRaises(AttributeError):
            pp.idx_retained = 1

    def test_outlier_detection_allowed_calls(self):

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

    def test_outlier_detection_not_allowed_calls(self):

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

    def test_KernelDensity_allowed_calls(self):

        X = np.random.rand(100,20)

        try:
            kerneld = KernelDensity(X, X[:,1])
        except Exception:
            self.assertTrue(False)

        try:
            kerneld = KernelDensity(X, X[:,4:9])
        except Exception:
            self.assertTrue(False)

        try:
            kerneld = KernelDensity(X, X[:,0])
        except Exception:
            self.assertTrue(False)

        try:
            kerneld = KernelDensity(X, X)
        except Exception:
            self.assertTrue(False)

        try:
            kerneld.X_weighted
            kerneld.weights
        except Exception:
            self.assertTrue(False)

    def test_KernelDensity_not_allowed_calls(self):

        X = np.random.rand(100,20)
        kerneld = KernelDensity(X, X[:,1])

        with self.assertRaises(AttributeError):
            kerneld.X_weighted = 1

        with self.assertRaises(AttributeError):
            kerneld.weights = 1

        with self.assertRaises(ValueError):
            kerneld = KernelDensity(X, X[20:30,1])

        with self.assertRaises(ValueError):
            kerneld = KernelDensity(X, X[20:30,:])
