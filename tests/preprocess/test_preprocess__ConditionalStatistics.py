import unittest
import numpy as np
from PCAfold import preprocess
from PCAfold import reduction
from PCAfold import analysis

class Preprocess(unittest.TestCase):

    def test_preprocess__ConditionalStatistics__allowed_calls(self):

        X = np.random.rand(100,20)
        cond_variable = np.random.rand(100,)

        try:
            cond = preprocess.ConditionalStatistics(X, cond_variable, k=2)
            cond.idx
            cond.borders
            cond.centroids
            cond.conditional_mean
            cond.conditional_minimum
            cond.conditional_maximum
            cond.conditional_standard_deviation
        except:
            self.assertTrue(False)

        cond_variable = np.random.rand(100,1)

        try:
            cond = preprocess.ConditionalStatistics(X, cond_variable, k=2)
        except:
            self.assertTrue(False)

        split_values = [0.1,0.5]

        try:
            cond = preprocess.ConditionalStatistics(X, cond_variable, k=2, split_values=split_values)
        except:
            self.assertTrue(False)

# ------------------------------------------------------------------------------

    def test_preprocess__ConditionalStatistics__not_allowed_calls(self):

        X = np.random.rand(100,20)
        cond_variable = np.random.rand(100,)
        cond = preprocess.ConditionalStatistics(X, cond_variable, k=2)

        with self.assertRaises(AttributeError):
            cond.idx = []

        with self.assertRaises(AttributeError):
            cond.borders = []

        with self.assertRaises(AttributeError):
            cond.centroids = []

        with self.assertRaises(AttributeError):
            cond.conditional_mean = []

        with self.assertRaises(AttributeError):
            cond.conditional_minimum = []

        with self.assertRaises(AttributeError):
            cond.conditional_maximum = []

        with self.assertRaises(AttributeError):
            cond.conditional_standard_deviation = []

        with self.assertRaises(ValueError):
            cond = preprocess.ConditionalStatistics([1,2,3], cond_variable, k=2)

        with self.assertRaises(ValueError):
            cond = preprocess.ConditionalStatistics(X, X[:,0:5], k=2)

        with self.assertRaises(ValueError):
            cond = preprocess.ConditionalStatistics(X, cond_variable, k=0)

        with self.assertRaises(ValueError):
            cond = preprocess.ConditionalStatistics(X, cond_variable, k=-1)

        with self.assertRaises(ValueError):
            cond = preprocess.ConditionalStatistics(X, cond_variable, split_values=2)

# ------------------------------------------------------------------------------

    def test_preprocess__ConditionalStatistics__computation(self):

        X = np.array([[1],[2],[3],[4],[5]])
        cond_variable = X

        try:
            cond = preprocess.ConditionalStatistics(X, cond_variable, k=4)
            self.assertTrue(np.array_equal(np.array([1.5, 2.5, 3.5, 4.5]), cond.centroids))
            self.assertTrue(np.array_equal(np.array([[1.],[2.],[3.],[4.5]]), cond.conditional_mean))
            self.assertTrue(np.array_equal(np.array([[1.],[2.],[3.],[4.]]), cond.conditional_minimum))
            self.assertTrue(np.array_equal(np.array([[1.],[2.],[3.],[5.]]), cond.conditional_maximum))
        except:
            self.assertTrue(False)

# ------------------------------------------------------------------------------
