import unittest
import numpy as np
from PCAfold import preprocess
from PCAfold import reduction
from PCAfold import PCA
from PCAfold import PreProcessing


class TestManipulation(unittest.TestCase):

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
            (X_cs, X_center, X_scale) = preprocess.center_scale(test_data_set, '-1to1', nocenter=True)
            (X_cs, X_center, X_scale) = preprocess.center_scale(test_data_set, 'level', nocenter=True)
            (X_cs, X_center, X_scale) = preprocess.center_scale(test_data_set, 'max', nocenter=True)
            (X_cs, X_center, X_scale) = preprocess.center_scale(test_data_set, 'poisson', nocenter=True)
            (X_cs, X_center, X_scale) = preprocess.center_scale(test_data_set, 'vast_2', nocenter=True)
            (X_cs, X_center, X_scale) = preprocess.center_scale(test_data_set, 'vast_3', nocenter=True)
            (X_cs, X_center, X_scale) = preprocess.center_scale(test_data_set, 'vast_4', nocenter=True)
        except Exception:
            self.assertTrue(False)

    def test_center_scale_on_0D_variable(self):

        test_0D_variable = np.random.rand(100,)

        # Instantiations that should work:
        try:
            (X_cs, X_center, X_scale) = preprocess.center_scale(test_0D_variable, 'none', nocenter=False)
            (X_cs, X_center, X_scale) = preprocess.center_scale(test_0D_variable, 'auto', nocenter=False)
            (X_cs, X_center, X_scale) = preprocess.center_scale(test_0D_variable, 'std', nocenter=False)
            (X_cs, X_center, X_scale) = preprocess.center_scale(test_0D_variable, 'pareto', nocenter=False)
            (X_cs, X_center, X_scale) = preprocess.center_scale(test_0D_variable, 'vast', nocenter=False)
            (X_cs, X_center, X_scale) = preprocess.center_scale(test_0D_variable, 'range', nocenter=False)
            (X_cs, X_center, X_scale) = preprocess.center_scale(test_0D_variable, '-1to1', nocenter=False)
            (X_cs, X_center, X_scale) = preprocess.center_scale(test_0D_variable, 'level', nocenter=False)
            (X_cs, X_center, X_scale) = preprocess.center_scale(test_0D_variable, 'max', nocenter=False)
            (X_cs, X_center, X_scale) = preprocess.center_scale(test_0D_variable, 'poisson', nocenter=False)
            (X_cs, X_center, X_scale) = preprocess.center_scale(test_0D_variable, 'vast_2', nocenter=False)
            (X_cs, X_center, X_scale) = preprocess.center_scale(test_0D_variable, 'vast_3', nocenter=False)
            (X_cs, X_center, X_scale) = preprocess.center_scale(test_0D_variable, 'vast_4', nocenter=False)

            (X_cs, X_center, X_scale) = preprocess.center_scale(test_0D_variable, 'none', nocenter=True)
            (X_cs, X_center, X_scale) = preprocess.center_scale(test_0D_variable, 'auto', nocenter=True)
            (X_cs, X_center, X_scale) = preprocess.center_scale(test_0D_variable, 'std', nocenter=True)
            (X_cs, X_center, X_scale) = preprocess.center_scale(test_0D_variable, 'pareto', nocenter=True)
            (X_cs, X_center, X_scale) = preprocess.center_scale(test_0D_variable, 'vast', nocenter=True)
            (X_cs, X_center, X_scale) = preprocess.center_scale(test_0D_variable, 'range', nocenter=True)
            (X_cs, X_center, X_scale) = preprocess.center_scale(test_0D_variable, '-1to1', nocenter=True)
            (X_cs, X_center, X_scale) = preprocess.center_scale(test_0D_variable, 'level', nocenter=True)
            (X_cs, X_center, X_scale) = preprocess.center_scale(test_0D_variable, 'max', nocenter=True)
            (X_cs, X_center, X_scale) = preprocess.center_scale(test_0D_variable, 'poisson', nocenter=True)
            (X_cs, X_center, X_scale) = preprocess.center_scale(test_0D_variable, 'vast_2', nocenter=True)
            (X_cs, X_center, X_scale) = preprocess.center_scale(test_0D_variable, 'vast_3', nocenter=True)
            (X_cs, X_center, X_scale) = preprocess.center_scale(test_0D_variable, 'vast_4', nocenter=True)
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
            (X_cs, X_center, X_scale) = preprocess.center_scale(test_1D_variable, '-1to1', nocenter=True)
            (X_cs, X_center, X_scale) = preprocess.center_scale(test_1D_variable, 'level', nocenter=True)
            (X_cs, X_center, X_scale) = preprocess.center_scale(test_1D_variable, 'max', nocenter=True)
            (X_cs, X_center, X_scale) = preprocess.center_scale(test_1D_variable, 'poisson', nocenter=True)
            (X_cs, X_center, X_scale) = preprocess.center_scale(test_1D_variable, 'vast_2', nocenter=True)
            (X_cs, X_center, X_scale) = preprocess.center_scale(test_1D_variable, 'vast_3', nocenter=True)
            (X_cs, X_center, X_scale) = preprocess.center_scale(test_1D_variable, 'vast_4', nocenter=True)
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
            test_data_set = np.random.rand(1000,)
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
            test_data_set = np.random.rand(1000,)
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
            self.assertTrue(X_center.all() == means.all())
        except Exception:
            self.assertTrue(False)

        try:
            (X_cs, X_center, X_scale) = preprocess.center_scale(test_data_set, 'std', nocenter=True)
            self.assertTrue(X_scale.all() == stds.all())
            self.assertTrue(X_center.all() == zeros.all())
        except Exception:
            self.assertTrue(False)

        try:
            (X_cs, X_center, X_scale) = preprocess.center_scale(test_data_set, 'none', nocenter=False)
            self.assertTrue(X_scale.all() == ones.all())
        except Exception:
            self.assertTrue(False)

    def test_invert_center_scale(self):
        # This function tests all possible inversions of center_scale function:
        test_data_set = np.random.rand(200,20)

        try:
            (X_cs, X_center, X_scale) = preprocess.center_scale(test_data_set, 'none', nocenter=False)
            X = preprocess.invert_center_scale(X_cs, X_center, X_scale)
            self.assertTrue(X.all() == test_data_set.all())
            (X_cs, X_center, X_scale) = preprocess.center_scale(test_data_set, 'auto', nocenter=False)
            X = preprocess.invert_center_scale(X_cs, X_center, X_scale)
            self.assertTrue(X.all() == test_data_set.all())
            (X_cs, X_center, X_scale) = preprocess.center_scale(test_data_set, 'std', nocenter=False)
            X = preprocess.invert_center_scale(X_cs, X_center, X_scale)
            self.assertTrue(X.all() == test_data_set.all())
            (X_cs, X_center, X_scale) = preprocess.center_scale(test_data_set, 'pareto', nocenter=False)
            X = preprocess.invert_center_scale(X_cs, X_center, X_scale)
            self.assertTrue(X.all() == test_data_set.all())
            (X_cs, X_center, X_scale) = preprocess.center_scale(test_data_set, 'vast', nocenter=False)
            X = preprocess.invert_center_scale(X_cs, X_center, X_scale)
            self.assertTrue(X.all() == test_data_set.all())
            (X_cs, X_center, X_scale) = preprocess.center_scale(test_data_set, 'range', nocenter=False)
            X = preprocess.invert_center_scale(X_cs, X_center, X_scale)
            self.assertTrue(X.all() == test_data_set.all())
            (X_cs, X_center, X_scale) = preprocess.center_scale(test_data_set, '-1to1', nocenter=False)
            X = preprocess.invert_center_scale(X_cs, X_center, X_scale)
            self.assertTrue(X.all() == test_data_set.all())
            (X_cs, X_center, X_scale) = preprocess.center_scale(test_data_set, 'level', nocenter=False)
            X = preprocess.invert_center_scale(X_cs, X_center, X_scale)
            self.assertTrue(X.all() == test_data_set.all())
            (X_cs, X_center, X_scale) = preprocess.center_scale(test_data_set, 'max', nocenter=False)
            X = preprocess.invert_center_scale(X_cs, X_center, X_scale)
            self.assertTrue(X.all() == test_data_set.all())
            (X_cs, X_center, X_scale) = preprocess.center_scale(test_data_set, 'poisson', nocenter=False)
            X = preprocess.invert_center_scale(X_cs, X_center, X_scale)
            self.assertTrue(X.all() == test_data_set.all())
            (X_cs, X_center, X_scale) = preprocess.center_scale(test_data_set, 'vast_2', nocenter=False)
            X = preprocess.invert_center_scale(X_cs, X_center, X_scale)
            self.assertTrue(X.all() == test_data_set.all())
            (X_cs, X_center, X_scale) = preprocess.center_scale(test_data_set, 'vast_3', nocenter=False)
            X = preprocess.invert_center_scale(X_cs, X_center, X_scale)
            self.assertTrue(X.all() == test_data_set.all())
            (X_cs, X_center, X_scale) = preprocess.center_scale(test_data_set, 'vast_4', nocenter=False)
            X = preprocess.invert_center_scale(X_cs, X_center, X_scale)
            self.assertTrue(X.all() == test_data_set.all())
        except Exception:
            self.assertTrue(False)

        try:
            (X_cs, X_center, X_scale) = preprocess.center_scale(test_data_set, 'none', nocenter=True)
            X = preprocess.invert_center_scale(X_cs, X_center, X_scale)
            self.assertTrue(X.all() == test_data_set.all())
            (X_cs, X_center, X_scale) = preprocess.center_scale(test_data_set, 'auto', nocenter=True)
            X = preprocess.invert_center_scale(X_cs, X_center, X_scale)
            self.assertTrue(X.all() == test_data_set.all())
            (X_cs, X_center, X_scale) = preprocess.center_scale(test_data_set, 'std', nocenter=True)
            X = preprocess.invert_center_scale(X_cs, X_center, X_scale)
            self.assertTrue(X.all() == test_data_set.all())
            (X_cs, X_center, X_scale) = preprocess.center_scale(test_data_set, 'pareto', nocenter=True)
            X = preprocess.invert_center_scale(X_cs, X_center, X_scale)
            self.assertTrue(X.all() == test_data_set.all())
            (X_cs, X_center, X_scale) = preprocess.center_scale(test_data_set, 'vast', nocenter=True)
            X = preprocess.invert_center_scale(X_cs, X_center, X_scale)
            self.assertTrue(X.all() == test_data_set.all())
            (X_cs, X_center, X_scale) = preprocess.center_scale(test_data_set, 'range', nocenter=True)
            X = preprocess.invert_center_scale(X_cs, X_center, X_scale)
            self.assertTrue(X.all() == test_data_set.all())
            (X_cs, X_center, X_scale) = preprocess.center_scale(test_data_set, '-1to1', nocenter=True)
            X = preprocess.invert_center_scale(X_cs, X_center, X_scale)
            self.assertTrue(X.all() == test_data_set.all())
            (X_cs, X_center, X_scale) = preprocess.center_scale(test_data_set, 'level', nocenter=True)
            X = preprocess.invert_center_scale(X_cs, X_center, X_scale)
            self.assertTrue(X.all() == test_data_set.all())
            (X_cs, X_center, X_scale) = preprocess.center_scale(test_data_set, 'max', nocenter=True)
            X = preprocess.invert_center_scale(X_cs, X_center, X_scale)
            self.assertTrue(X.all() == test_data_set.all())
            (X_cs, X_center, X_scale) = preprocess.center_scale(test_data_set, 'poisson', nocenter=True)
            X = preprocess.invert_center_scale(X_cs, X_center, X_scale)
            self.assertTrue(X.all() == test_data_set.all())
            (X_cs, X_center, X_scale) = preprocess.center_scale(test_data_set, 'vast_2', nocenter=True)
            X = preprocess.invert_center_scale(X_cs, X_center, X_scale)
            self.assertTrue(X.all() == test_data_set.all())
            (X_cs, X_center, X_scale) = preprocess.center_scale(test_data_set, 'vast_3', nocenter=True)
            X = preprocess.invert_center_scale(X_cs, X_center, X_scale)
            self.assertTrue(X.all() == test_data_set.all())
            (X_cs, X_center, X_scale) = preprocess.center_scale(test_data_set, 'vast_4', nocenter=True)
            X = preprocess.invert_center_scale(X_cs, X_center, X_scale)
            self.assertTrue(X.all() == test_data_set.all())
        except Exception:
            self.assertTrue(False)

    def test_invert_center_scale_on_0D_variable(self):
        # This function tests all possible inversions of center_scale function:
        test_data_set = np.random.rand(200,)

        try:
            (X_cs, X_center, X_scale) = preprocess.center_scale(test_data_set, 'none', nocenter=False)
            X = preprocess.invert_center_scale(X_cs, X_center, X_scale)
            self.assertTrue(X.all() == test_data_set.all())
            (X_cs, X_center, X_scale) = preprocess.center_scale(test_data_set, 'auto', nocenter=False)
            X = preprocess.invert_center_scale(X_cs, X_center, X_scale)
            self.assertTrue(X.all() == test_data_set.all())
            (X_cs, X_center, X_scale) = preprocess.center_scale(test_data_set, 'std', nocenter=False)
            X = preprocess.invert_center_scale(X_cs, X_center, X_scale)
            self.assertTrue(X.all() == test_data_set.all())
            (X_cs, X_center, X_scale) = preprocess.center_scale(test_data_set, 'pareto', nocenter=False)
            X = preprocess.invert_center_scale(X_cs, X_center, X_scale)
            self.assertTrue(X.all() == test_data_set.all())
            (X_cs, X_center, X_scale) = preprocess.center_scale(test_data_set, 'vast', nocenter=False)
            X = preprocess.invert_center_scale(X_cs, X_center, X_scale)
            self.assertTrue(X.all() == test_data_set.all())
            (X_cs, X_center, X_scale) = preprocess.center_scale(test_data_set, 'range', nocenter=False)
            X = preprocess.invert_center_scale(X_cs, X_center, X_scale)
            self.assertTrue(X.all() == test_data_set.all())
            (X_cs, X_center, X_scale) = preprocess.center_scale(test_data_set, '-1to1', nocenter=False)
            X = preprocess.invert_center_scale(X_cs, X_center, X_scale)
            self.assertTrue(X.all() == test_data_set.all())
            (X_cs, X_center, X_scale) = preprocess.center_scale(test_data_set, 'level', nocenter=False)
            X = preprocess.invert_center_scale(X_cs, X_center, X_scale)
            self.assertTrue(X.all() == test_data_set.all())
            (X_cs, X_center, X_scale) = preprocess.center_scale(test_data_set, 'max', nocenter=False)
            X = preprocess.invert_center_scale(X_cs, X_center, X_scale)
            self.assertTrue(X.all() == test_data_set.all())
            (X_cs, X_center, X_scale) = preprocess.center_scale(test_data_set, 'poisson', nocenter=False)
            X = preprocess.invert_center_scale(X_cs, X_center, X_scale)
            self.assertTrue(X.all() == test_data_set.all())
            (X_cs, X_center, X_scale) = preprocess.center_scale(test_data_set, 'vast_2', nocenter=False)
            X = preprocess.invert_center_scale(X_cs, X_center, X_scale)
            self.assertTrue(X.all() == test_data_set.all())
            (X_cs, X_center, X_scale) = preprocess.center_scale(test_data_set, 'vast_3', nocenter=False)
            X = preprocess.invert_center_scale(X_cs, X_center, X_scale)
            self.assertTrue(X.all() == test_data_set.all())
            (X_cs, X_center, X_scale) = preprocess.center_scale(test_data_set, 'vast_4', nocenter=False)
            X = preprocess.invert_center_scale(X_cs, X_center, X_scale)
            self.assertTrue(X.all() == test_data_set.all())
        except Exception:
            self.assertTrue(False)

        try:
            (X_cs, X_center, X_scale) = preprocess.center_scale(test_data_set, 'none', nocenter=True)
            X = preprocess.invert_center_scale(X_cs, X_center, X_scale)
            self.assertTrue(X.all() == test_data_set.all())
            (X_cs, X_center, X_scale) = preprocess.center_scale(test_data_set, 'auto', nocenter=True)
            X = preprocess.invert_center_scale(X_cs, X_center, X_scale)
            self.assertTrue(X.all() == test_data_set.all())
            (X_cs, X_center, X_scale) = preprocess.center_scale(test_data_set, 'std', nocenter=True)
            X = preprocess.invert_center_scale(X_cs, X_center, X_scale)
            self.assertTrue(X.all() == test_data_set.all())
            (X_cs, X_center, X_scale) = preprocess.center_scale(test_data_set, 'pareto', nocenter=True)
            X = preprocess.invert_center_scale(X_cs, X_center, X_scale)
            self.assertTrue(X.all() == test_data_set.all())
            (X_cs, X_center, X_scale) = preprocess.center_scale(test_data_set, 'vast', nocenter=True)
            X = preprocess.invert_center_scale(X_cs, X_center, X_scale)
            self.assertTrue(X.all() == test_data_set.all())
            (X_cs, X_center, X_scale) = preprocess.center_scale(test_data_set, 'range', nocenter=True)
            X = preprocess.invert_center_scale(X_cs, X_center, X_scale)
            self.assertTrue(X.all() == test_data_set.all())
            (X_cs, X_center, X_scale) = preprocess.center_scale(test_data_set, '-1to1', nocenter=True)
            X = preprocess.invert_center_scale(X_cs, X_center, X_scale)
            self.assertTrue(X.all() == test_data_set.all())
            (X_cs, X_center, X_scale) = preprocess.center_scale(test_data_set, 'level', nocenter=True)
            X = preprocess.invert_center_scale(X_cs, X_center, X_scale)
            self.assertTrue(X.all() == test_data_set.all())
            (X_cs, X_center, X_scale) = preprocess.center_scale(test_data_set, 'max', nocenter=True)
            X = preprocess.invert_center_scale(X_cs, X_center, X_scale)
            self.assertTrue(X.all() == test_data_set.all())
            (X_cs, X_center, X_scale) = preprocess.center_scale(test_data_set, 'poisson', nocenter=True)
            X = preprocess.invert_center_scale(X_cs, X_center, X_scale)
            self.assertTrue(X.all() == test_data_set.all())
            (X_cs, X_center, X_scale) = preprocess.center_scale(test_data_set, 'vast_2', nocenter=True)
            X = preprocess.invert_center_scale(X_cs, X_center, X_scale)
            self.assertTrue(X.all() == test_data_set.all())
            (X_cs, X_center, X_scale) = preprocess.center_scale(test_data_set, 'vast_3', nocenter=True)
            X = preprocess.invert_center_scale(X_cs, X_center, X_scale)
            self.assertTrue(X.all() == test_data_set.all())
            (X_cs, X_center, X_scale) = preprocess.center_scale(test_data_set, 'vast_4', nocenter=True)
            X = preprocess.invert_center_scale(X_cs, X_center, X_scale)
            self.assertTrue(X.all() == test_data_set.all())
        except Exception:
            self.assertTrue(False)

    def test_invert_center_scale_on_1D_variable(self):
        # This function tests all possible inversions of center_scale function:
        test_data_set = np.random.rand(200,1)

        try:
            (X_cs, X_center, X_scale) = preprocess.center_scale(test_data_set, 'none', nocenter=False)
            X = preprocess.invert_center_scale(X_cs, X_center, X_scale)
            self.assertTrue(X.all() == test_data_set.all())
            (X_cs, X_center, X_scale) = preprocess.center_scale(test_data_set, 'auto', nocenter=False)
            X = preprocess.invert_center_scale(X_cs, X_center, X_scale)
            self.assertTrue(X.all() == test_data_set.all())
            (X_cs, X_center, X_scale) = preprocess.center_scale(test_data_set, 'std', nocenter=False)
            X = preprocess.invert_center_scale(X_cs, X_center, X_scale)
            self.assertTrue(X.all() == test_data_set.all())
            (X_cs, X_center, X_scale) = preprocess.center_scale(test_data_set, 'pareto', nocenter=False)
            X = preprocess.invert_center_scale(X_cs, X_center, X_scale)
            self.assertTrue(X.all() == test_data_set.all())
            (X_cs, X_center, X_scale) = preprocess.center_scale(test_data_set, 'vast', nocenter=False)
            X = preprocess.invert_center_scale(X_cs, X_center, X_scale)
            self.assertTrue(X.all() == test_data_set.all())
            (X_cs, X_center, X_scale) = preprocess.center_scale(test_data_set, 'range', nocenter=False)
            X = preprocess.invert_center_scale(X_cs, X_center, X_scale)
            self.assertTrue(X.all() == test_data_set.all())
            (X_cs, X_center, X_scale) = preprocess.center_scale(test_data_set, '-1to1', nocenter=False)
            X = preprocess.invert_center_scale(X_cs, X_center, X_scale)
            self.assertTrue(X.all() == test_data_set.all())
            (X_cs, X_center, X_scale) = preprocess.center_scale(test_data_set, 'level', nocenter=False)
            X = preprocess.invert_center_scale(X_cs, X_center, X_scale)
            self.assertTrue(X.all() == test_data_set.all())
            (X_cs, X_center, X_scale) = preprocess.center_scale(test_data_set, 'max', nocenter=False)
            X = preprocess.invert_center_scale(X_cs, X_center, X_scale)
            self.assertTrue(X.all() == test_data_set.all())
            (X_cs, X_center, X_scale) = preprocess.center_scale(test_data_set, 'poisson', nocenter=False)
            X = preprocess.invert_center_scale(X_cs, X_center, X_scale)
            self.assertTrue(X.all() == test_data_set.all())
            (X_cs, X_center, X_scale) = preprocess.center_scale(test_data_set, 'vast_2', nocenter=False)
            X = preprocess.invert_center_scale(X_cs, X_center, X_scale)
            self.assertTrue(X.all() == test_data_set.all())
            (X_cs, X_center, X_scale) = preprocess.center_scale(test_data_set, 'vast_3', nocenter=False)
            X = preprocess.invert_center_scale(X_cs, X_center, X_scale)
            self.assertTrue(X.all() == test_data_set.all())
            (X_cs, X_center, X_scale) = preprocess.center_scale(test_data_set, 'vast_4', nocenter=False)
            X = preprocess.invert_center_scale(X_cs, X_center, X_scale)
            self.assertTrue(X.all() == test_data_set.all())
        except Exception:
            self.assertTrue(False)

        try:
            (X_cs, X_center, X_scale) = preprocess.center_scale(test_data_set, 'none', nocenter=True)
            X = preprocess.invert_center_scale(X_cs, X_center, X_scale)
            self.assertTrue(X.all() == test_data_set.all())
            (X_cs, X_center, X_scale) = preprocess.center_scale(test_data_set, 'auto', nocenter=True)
            X = preprocess.invert_center_scale(X_cs, X_center, X_scale)
            self.assertTrue(X.all() == test_data_set.all())
            (X_cs, X_center, X_scale) = preprocess.center_scale(test_data_set, 'std', nocenter=True)
            X = preprocess.invert_center_scale(X_cs, X_center, X_scale)
            self.assertTrue(X.all() == test_data_set.all())
            (X_cs, X_center, X_scale) = preprocess.center_scale(test_data_set, 'pareto', nocenter=True)
            X = preprocess.invert_center_scale(X_cs, X_center, X_scale)
            self.assertTrue(X.all() == test_data_set.all())
            (X_cs, X_center, X_scale) = preprocess.center_scale(test_data_set, 'vast', nocenter=True)
            X = preprocess.invert_center_scale(X_cs, X_center, X_scale)
            self.assertTrue(X.all() == test_data_set.all())
            (X_cs, X_center, X_scale) = preprocess.center_scale(test_data_set, 'range', nocenter=True)
            X = preprocess.invert_center_scale(X_cs, X_center, X_scale)
            self.assertTrue(X.all() == test_data_set.all())
            (X_cs, X_center, X_scale) = preprocess.center_scale(test_data_set, '-1to1', nocenter=True)
            X = preprocess.invert_center_scale(X_cs, X_center, X_scale)
            self.assertTrue(X.all() == test_data_set.all())
            (X_cs, X_center, X_scale) = preprocess.center_scale(test_data_set, 'level', nocenter=True)
            X = preprocess.invert_center_scale(X_cs, X_center, X_scale)
            self.assertTrue(X.all() == test_data_set.all())
            (X_cs, X_center, X_scale) = preprocess.center_scale(test_data_set, 'max', nocenter=True)
            X = preprocess.invert_center_scale(X_cs, X_center, X_scale)
            self.assertTrue(X.all() == test_data_set.all())
            (X_cs, X_center, X_scale) = preprocess.center_scale(test_data_set, 'poisson', nocenter=True)
            X = preprocess.invert_center_scale(X_cs, X_center, X_scale)
            self.assertTrue(X.all() == test_data_set.all())
            (X_cs, X_center, X_scale) = preprocess.center_scale(test_data_set, 'vast_2', nocenter=True)
            X = preprocess.invert_center_scale(X_cs, X_center, X_scale)
            self.assertTrue(X.all() == test_data_set.all())
            (X_cs, X_center, X_scale) = preprocess.center_scale(test_data_set, 'vast_3', nocenter=True)
            X = preprocess.invert_center_scale(X_cs, X_center, X_scale)
            self.assertTrue(X.all() == test_data_set.all())
            (X_cs, X_center, X_scale) = preprocess.center_scale(test_data_set, 'vast_4', nocenter=True)
            X = preprocess.invert_center_scale(X_cs, X_center, X_scale)
            self.assertTrue(X.all() == test_data_set.all())
        except Exception:
            self.assertTrue(False)

    def test_invert_center_scale_single_variable(self):

        try:
            test_data_set = np.ones((200,))
            X_result = 2*np.ones((200,))
            X = preprocess.invert_center_scale(test_data_set, 0, 2)
            self.assertTrue(X.all() == X_result.all())
        except Exception:
            self.assertTrue(False)

        try:
            test_data_set = np.ones((200,))
            X_result = 3*np.ones((200,))
            X = preprocess.invert_center_scale(test_data_set, 1, 2)
            self.assertTrue(X.all() == X_result.all())
        except Exception:
            self.assertTrue(False)

        try:
            test_data_set = np.ones((200,1))
            X_result = 2*np.ones((200,))
            X = preprocess.invert_center_scale(test_data_set, 0, 2)
            self.assertTrue(X.all() == X_result.all())
        except Exception:
            self.assertTrue(False)

        try:
            test_data_set = np.ones((200,1))
            X_result = 3*np.ones((200,))
            X = preprocess.invert_center_scale(test_data_set, 1, 2)
            self.assertTrue(X.all() == X_result.all())
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
            self.assertTrue(X_removed.all() == test_data_set.all())
            self.assertTrue(idx_removed == idx_removed_check)
            self.assertTrue(idx_retained == idx_retained_check)
        except Exception:
            self.assertTrue(False)

        try:
            # Inject a constant column that has values close to zero:
            close_to_zero_column = -10**(-14)*np.ones((100,1))
            close_to_zero_column[20:30,:] = -10**(-13)
            close_to_zero_column[80:85,:] = -10**(-15)
            test_data_set_constant = np.hstack((test_data_set[:,0:3], close_to_zero_column, test_data_set[:,3::]))
            idx_removed_check = [3]
            idx_retained_check = [0,1,2,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
            (X_removed, idx_removed, idx_retained) = preprocess.remove_constant_vars(test_data_set_constant)
            self.assertTrue(X_removed.all() == test_data_set.all())
            self.assertTrue(idx_removed == idx_removed_check)
            self.assertTrue(idx_retained == idx_retained_check)
        except Exception:
            self.assertTrue(False)

        try:
            # Inject a constant column that has values close to zero:
            close_to_zero_column = -10**(-14)*np.ones((100,1))
            close_to_zero_column[20:30,:] = 10**(-13)
            close_to_zero_column[80:85,:] = 10**(-15)
            test_data_set_constant = np.hstack((test_data_set[:,0:3], close_to_zero_column, test_data_set[:,3::]))
            idx_removed_check = [3]
            idx_retained_check = [0,1,2,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
            (X_removed, idx_removed, idx_retained) = preprocess.remove_constant_vars(test_data_set_constant)
            self.assertTrue(X_removed.all() == test_data_set.all())
            self.assertTrue(idx_removed == idx_removed_check)
            self.assertTrue(idx_retained == idx_retained_check)
        except Exception:
            self.assertTrue(False)

    def test_analyze_centers_change(self):

        test_data_set = np.random.rand(100,20)
        idx_X_r = np.array([1,5,68,9,2,3,6,43,56])

        try:
            (normalized_C, normalized_C_r, center_movement_percentage, plt) = preprocess.analyze_centers_change(test_data_set, idx_X_r, variable_names=[], plot_variables=[], legend_label=[], title=None, save_filename=None)
            plt.close()
        except Exception:
            self.assertTrue(False)

        try:
            (normalized_C, normalized_C_r, center_movement_percentage, plt) = preprocess.analyze_centers_change(test_data_set, idx_X_r, variable_names=[], plot_variables=[1,4,5], legend_label=[], title=None, save_filename=None)
            plt.close()
        except Exception:
            self.assertTrue(False)

    def test_PreProcessing(self):

        test_data_set = np.random.rand(100,20)

        # Inject two constant columns:
        test_data_set_constant = np.hstack((test_data_set[:,0:3], 2.4*np.ones((100,1)), test_data_set[:,3:15], -8.1*np.ones((100,1)), test_data_set[:,15::]))
        idx_removed_check = [3,16]
        idx_retained_check = [0,1,2,4,5,6,7,8,9,10,11,12,13,14,15,17,18,19,20,21]

        try:
            preprocessed = PreProcessing(test_data_set_constant, scaling='none', nocenter=True)
            self.assertTrue(preprocessed.X_removed.all() == test_data_set.all())
            self.assertTrue(preprocessed.idx_removed == idx_removed_check)
            self.assertTrue(preprocessed.idx_retained == idx_retained_check)
            self.assertTrue(np.shape(preprocessed.X_cs) == (100,20))
        except Exception:
            self.assertTrue(False)
