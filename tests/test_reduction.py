import unittest
import numpy as np
from PCAfold import preprocess
from PCAfold import reduction
from PCAfold import PCA


class TestReduction(unittest.TestCase):

    def test_PCA_class_initialization(self):

        test_data_set = np.random.rand(100,20)
        test_data_set_constant = np.random.rand(100,20)
        test_data_set_constant[:,10] = np.ones((100,))
        test_data_set_constant[:,5] = np.ones((100,))

        # Instantiations that should work:
        try:
            pca = PCA(test_data_set, scaling='auto')
        except Exception:
            self.assertTrue(False)

        try:
            pca = PCA(test_data_set, scaling='auto')
        except Exception:
            self.assertTrue(False)

        try:
            pca = PCA(test_data_set, scaling='std')
        except Exception:
            self.assertTrue(False)

        try:
            pca = PCA(test_data_set, scaling='none')
        except Exception:
            self.assertTrue(False)

        try:
            pca = PCA(test_data_set, scaling='auto', neta=2)
        except Exception:
            self.assertTrue(False)

        try:
            pca = PCA(test_data_set, scaling='auto', neta=3, nocenter=True)
        except Exception:
            self.assertTrue(False)

        try:
            pca = PCA(test_data_set, scaling='pareto', neta=2, nocenter=True)
        except Exception:
            self.assertTrue(False)

        try:
            pca = PCA(test_data_set, scaling='auto', neta=2, useXTXeig=False)
        except Exception:
            self.assertTrue(False)

        try:
            pca = PCA(test_data_set, scaling='range', neta=2, useXTXeig=False, nocenter=True)
        except Exception:
            self.assertTrue(False)

        # Instantiations that should not work:
        with self.assertRaises(ValueError):
            pca = PCA(test_data_set, scaling='none', neta=-1)

        with self.assertRaises(ValueError):
            pca = PCA(test_data_set, scaling='auto', neta=30)

        with self.assertRaises(ValueError):
            pca = PCA(test_data_set, scaling='auto', neta=3, useXTXeig=1)

        with self.assertRaises(ValueError):
            pca = PCA(test_data_set, scaling='auto', nocenter=1)

        with self.assertRaises(ValueError):
            pca = PCA(test_data_set, scaling=False)

        with self.assertRaises(ValueError):
            pca = PCA(test_data_set, scaling='none', neta=True)

        with self.assertRaises(ValueError):
            pca = PCA(test_data_set, scaling='none', neta=5, nocenter='False')

        with self.assertRaises(ValueError):
            pca = PCA(test_data_set, scaling='auto', neta=3, useXTXeig='True')

        with self.assertRaises(ValueError):
            pca = PCA(test_data_set_constant, scaling='auto', neta=2)

        with self.assertRaises(ValueError):
            pca = PCA(test_data_set_constant)

        with self.assertRaises(ValueError):
            pca = PCA(test_data_set_constant, scaling='range', neta=5)

        try:
            (X_removed, idx_removed, idx_retained) = preprocess.remove_constant_vars(test_data_set_constant)
        except Exception:
            self.assertTrue(False)

        try:
            pca = PCA(X_removed, scaling='range', neta=2)
        except Exception:
            self.assertTrue(False)

    def test_x2eta(self):

        test_data_set = np.random.rand(10,2)

        pca = PCA(test_data_set, scaling='auto')

        # Instantiations that should work:
        try:
            pca.x2eta(test_data_set)
        except Exception:
            self.assertTrue(False)

        try:
            scores = pca.x2eta(test_data_set)
        except Exception:
            self.assertTrue(False)

        try:
            x = pca.eta2x(scores)
        except Exception:
            self.assertTrue(False)

        self.assertTrue(test_data_set.all() == x.all())
