import unittest
import numpy as np
from PCAfold import preprocess
from PCAfold import reduction
from PCAfold import PCA, LPCA
from PCAfold import DataSampler
from scipy import linalg as lg

class TestReduction(unittest.TestCase):

################################################################################
#
# Test PCA class
#
################################################################################

    def test_PCA_with_eigendec_vs_SVD(self):

        tol = 10 * np.finfo(float).eps

        # create random dataset with zero mean
        n_observations = 100
        PHI = np.vstack(
            (np.sin(np.linspace(0, np.pi, n_observations)).T, np.cos(np.linspace(0, 2 * np.pi, n_observations)),
             np.linspace(0, np.pi, n_observations)))
        PHI, cntr, scl = preprocess.center_scale(PHI.T, 'NONE')

        # create random means for the dataset for comparison with PCA X_center
        xbar = np.random.rand(1, PHI.shape[1])

        # svd on PHI to get Q and L for comparison with PCA Q and L
        U, s, V = lg.svd(PHI)
        L = s * s / np.sum(s * s)
        isort = np.argsort(-np.diagonal(np.diag(L)))  # descending order
        L = L[isort]
        Q = V.T[:, isort]

        # checking both methods for PCA:
        pca = PCA(PHI + xbar, 'NONE', use_eigendec=False)
        pca2 = PCA(PHI + xbar, 'NONE', use_eigendec=True)

        # comparing mean(centering), centered data, Q, and L

        if np.any(xbar - pca.X_center > tol) or np.any(xbar - pca2.X_center > tol):
            self.assertTrue(False)

        if np.any(PHI - pca.X_cs > tol) or np.any(PHI - pca2.X_cs > tol):
            self.assertTrue(False)

        if np.any(Q - pca.A > tol) or np.any(Q - pca2.A > tol):
            self.assertTrue(False)

        if np.any(L - pca.L > tol) or np.any(L - pca2.L > tol):
            self.assertTrue(False)

        # Check if feed eta's to PCA, return same eta's when do transform
        eta = pca.transform(PHI + xbar)  # dataset as example of eta's

        # both methods of PCA:
        pca = PCA(eta, 'NONE', use_eigendec=False)
        pca2 = PCA(eta, 'NONE', use_eigendec=True)

        # transform transformation:
        eta_new = pca.transform(eta)
        eta_new2 = pca2.transform(eta)

        # transformation can have different direction -> check sign is the same before compare eta's
        (n_observations, n_variables) = np.shape(PHI)
        for i in range(n_variables):
            if np.sign(eta[0, i]) != np.sign(eta_new[0, i]):
                eta_new[:, i] *= -1
            if np.sign(eta[0, i]) != np.sign(eta_new2[0, i]):
                eta_new2[:, i] *= -1

        # checking eta's are the same from transformation of eta
        if np.any(eta - eta_new > tol) or np.any(eta - eta_new2 > tol):
            self.assertTrue(False)

# Test if 10 PCA class attributes cannot be set by the user after `PCA` object has been created:
    def test_PCA_not_allowed_attribute_setting(self):

        X = np.random.rand(100,20)
        pca = PCA(X, scaling='auto', n_components=10)

        with self.assertRaises(AttributeError):
            pca.X_cs = 1
        with self.assertRaises(AttributeError):
            pca.X_center = 1
        with self.assertRaises(AttributeError):
            pca.X_scale = 1
        with self.assertRaises(AttributeError):
            pca.S = 1
        with self.assertRaises(AttributeError):
            pca.A = 1
        with self.assertRaises(AttributeError):
            pca.L = 1
        with self.assertRaises(AttributeError):
            pca.loadings = 1
        with self.assertRaises(AttributeError):
            pca.scaling = 1
        with self.assertRaises(AttributeError):
            pca.n_variables = 1
        with self.assertRaises(AttributeError):
            pca.n_components_init = 1

# Test if all 11 available PCA class attributes can be accessed without error:
    def test_PCA_class_getting_attributes(self):

        X = np.random.rand(100,20)
        pca = PCA(X, scaling='auto', n_components=10)

        try:
            pca.X_cs
            pca.X_center
            pca.X_scale
            pca.S
            pca.A
            pca.L
            pca.loadings
            pca.scaling
            pca.n_variables
            pca.n_components_init
            pca.n_components
        except Exception:
            self.assertTrue(False)

# Test n_components PCA class attribute - the only attribute that is allowed to be set
    def test_PCA_n_components_attribute(self):

        X = np.random.rand(100,20)

        try:
            pca = PCA(X, scaling='auto', n_components=2)
        except Exception:
            self.assertTrue(False)

        try:
            pca.n_components
        except Exception:
            self.assertTrue(False)

        try:
            pca = PCA(X, scaling='auto', n_components=2)
            pca.n_components = 0
            current_n = pca.n_components
            self.assertTrue(current_n == 20)
        except Exception:
            self.assertTrue(False)

        try:
            pca = PCA(X, scaling='auto', n_components=2)
            pca.n_components = 10
            current_n = pca.n_components
            self.assertTrue(current_n == 10)
        except Exception:
            self.assertTrue(False)

        try:
            pca = PCA(X, scaling='auto', n_components=2)
            pca.n_components = 10
            current_n = pca.n_components
            self.assertTrue(current_n == 10)
            pca.n_components = pca.n_components_init
            current_n = pca.n_components
            self.assertTrue(current_n == 2)
        except Exception:
            self.assertTrue(False)

    def test_PCA_n_components_attribute_not_allowed(self):

        X = np.random.rand(100,20)

        with self.assertRaises(ValueError):
            pca = PCA(X, scaling='auto', n_components=-1)
        with self.assertRaises(ValueError):
            pca = PCA(X, scaling='auto', n_components=1.5)
        with self.assertRaises(ValueError):
            pca = PCA(X, scaling='auto', n_components=True)
        with self.assertRaises(ValueError):
            pca = PCA(X, scaling='auto', n_components='PC')

        try:
            pca = PCA(X, scaling='auto', n_components=10)
        except Exception:
            self.assertTrue(False)

        with self.assertRaises(ValueError):
            pca.n_components = -1

        with self.assertRaises(ValueError):
            pca.n_components = 21

        with self.assertRaises(ValueError):
            pca.n_components = True

        with self.assertRaises(ValueError):
            pca.n_components = 1.5

        with self.assertRaises(ValueError):
            pca.n_components = 'PC'

    def test_PCA_allowed_initializations(self):

        test_data_set = np.random.rand(100,20)
        test_data_set_constant = np.random.rand(100,20)
        test_data_set_constant[:,10] = np.ones((100,))
        test_data_set_constant[:,5] = np.ones((100,))

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
            pca = PCA(test_data_set, scaling='auto', n_components=2)
        except Exception:
            self.assertTrue(False)

        try:
            pca = PCA(test_data_set, scaling='auto', n_components=3, nocenter=True)
        except Exception:
            self.assertTrue(False)

        try:
            pca = PCA(test_data_set, scaling='pareto', n_components=2, nocenter=True)
        except Exception:
            self.assertTrue(False)

        try:
            pca = PCA(test_data_set, scaling='auto', n_components=2, use_eigendec=False)
        except Exception:
            self.assertTrue(False)

        try:
            pca = PCA(test_data_set, scaling='range', n_components=2, use_eigendec=False, nocenter=True)
        except Exception:
            self.assertTrue(False)

        try:
            (X_removed, idx_removed, idx_retained) = preprocess.remove_constant_vars(test_data_set_constant)
        except Exception:
            self.assertTrue(False)

        try:
            pca = PCA(X_removed, scaling='range', n_components=2)
        except Exception:
            self.assertTrue(False)

    def test_PCA_not_allowed_initializations(self):

        test_data_set = np.random.rand(100,20)
        test_data_set_constant = np.random.rand(100,20)
        test_data_set_constant[:,10] = np.ones((100,))
        test_data_set_constant[:,5] = np.ones((100,))

        with self.assertRaises(ValueError):
            pca = PCA(test_data_set, scaling='none', n_components=-1)

        with self.assertRaises(ValueError):
            pca = PCA(test_data_set, scaling='auto', n_components=30)

        with self.assertRaises(ValueError):
            pca = PCA(test_data_set, scaling='auto', n_components=3, use_eigendec=1)

        with self.assertRaises(ValueError):
            pca = PCA(test_data_set, scaling='auto', nocenter=1)

        with self.assertRaises(ValueError):
            pca = PCA(test_data_set, scaling=False)

        with self.assertRaises(ValueError):
            pca = PCA(test_data_set, scaling='none', n_components=True)

        with self.assertRaises(ValueError):
            pca = PCA(test_data_set, scaling='none', n_components=5, nocenter='False')

        with self.assertRaises(ValueError):
            pca = PCA(test_data_set, scaling='auto', n_components=3, use_eigendec='True')

        with self.assertRaises(ValueError):
            pca = PCA(test_data_set_constant, scaling='auto', n_components=2)

        with self.assertRaises(ValueError):
            pca = PCA(test_data_set_constant)

        with self.assertRaises(ValueError):
            pca = PCA(test_data_set_constant, scaling='range', n_components=5)

    def test_transform_allowed_calls(self):

        test_data_set = np.random.rand(10,2)

        pca = PCA(test_data_set, scaling='auto')

        try:
            pca.transform(test_data_set)
        except Exception:
            self.assertTrue(False)

        try:
            scores = pca.transform(test_data_set)
        except Exception:
            self.assertTrue(False)

        try:
            x = pca.reconstruct(scores)
        except Exception:
            self.assertTrue(False)

        try:
            scores = pca.transform(test_data_set)
            x = pca.reconstruct(scores)
            difference = abs(test_data_set - x)
            comparison = difference < 10**(-14)
            self.assertTrue(comparison.all())
        except Exception:
            self.assertTrue(False)

    def test_transform_not_allowed_calls(self):

        test_data_set = np.random.rand(10,2)
        test_data_set_2 = np.random.rand(10,3)

        pca = PCA(test_data_set, scaling='auto')

        with self.assertRaises(ValueError):
            pca.transform(test_data_set_2)

    def test_reconstruct_allowed_calls(self):

        X = np.random.rand(100,10)

        try:
            pca_X = PCA(X, scaling='auto')
            principal_components = pca_X.transform(X)
            X_rec = pca_X.reconstruct(principal_components)
        except Exception:
            self.assertTrue(False)

        try:
            pca_X = PCA(X, scaling='auto', n_components=5)
            principal_components = pca_X.transform(X)
            X_rec = pca_X.reconstruct(principal_components)
        except Exception:
            self.assertTrue(False)

        try:
            pca_X = PCA(X, scaling='auto', n_components=2)
            principal_components = pca_X.transform(X)
            X_rec = pca_X.reconstruct(principal_components)
        except Exception:
            self.assertTrue(False)

        try:
            pca_X = PCA(X, scaling='auto', n_components=2, nocenter=True)
            principal_components = pca_X.transform(X)
            X_rec = pca_X.reconstruct(principal_components)
        except Exception:
            self.assertTrue(False)

        try:
            pca_X = PCA(X, scaling='auto', n_components=2)
            principal_components = pca_X.transform(X, nocenter=True)
            X_rec = pca_X.reconstruct(principal_components, nocenter=True)
        except Exception:
            self.assertTrue(False)

        try:
            pca_X = PCA(X, scaling='auto', n_components=2)
            X_2 = np.random.rand(200,10)
            principal_components = pca_X.transform(X_2, nocenter=True)
            X_rec = pca_X.reconstruct(principal_components, nocenter=True)
        except Exception:
            self.assertTrue(False)

        try:
            pca_X = PCA(X, scaling='auto', n_components=2)
            X_2 = np.random.rand(200,10)
            principal_components = pca_X.transform(X_2)
            X_rec = pca_X.reconstruct(principal_components)
        except Exception:
            self.assertTrue(False)

    def test_reconstruct_not_allowed_calls(self):

        X = np.random.rand(100,10)
        fake_PCs = np.random.rand(100,11)

        pca = PCA(X, scaling='auto')
        with self.assertRaises(ValueError):
            X_rec = pca.reconstruct(fake_PCs)

        pca = PCA(X, scaling='auto', n_components=4)
        with self.assertRaises(ValueError):
            X_rec = pca.reconstruct(fake_PCs)

    def test_transform_reconstruct_on_all_available_scalings(self):

        X = np.random.rand(100,10)

        try:
            pca_X = PCA(X, scaling='none', n_components=0)
            principal_components = pca_X.transform(X)
            X_rec = pca_X.reconstruct(principal_components)
            difference = abs(X - X_rec)
            comparison = difference < 10**(-14)
            self.assertTrue(comparison.all())
        except Exception:
            self.assertTrue(False)

        try:
            pca_X = PCA(X, scaling='auto', n_components=0)
            principal_components = pca_X.transform(X)
            X_rec = pca_X.reconstruct(principal_components)
            difference = abs(X - X_rec)
            comparison = difference < 10**(-14)
            self.assertTrue(comparison.all())
        except Exception:
            self.assertTrue(False)

        try:
            pca_X = PCA(X, scaling='range', n_components=0)
            principal_components = pca_X.transform(X)
            X_rec = pca_X.reconstruct(principal_components)
            difference = abs(X - X_rec)
            comparison = difference < 10**(-14)
            self.assertTrue(comparison.all())
        except Exception:
            self.assertTrue(False)

        try:
            pca_X = PCA(X, scaling='vast', n_components=0)
            principal_components = pca_X.transform(X)
            X_rec = pca_X.reconstruct(principal_components)
            difference = abs(X - X_rec)
            comparison = difference < 10**(-14)
            self.assertTrue(comparison.all())
        except Exception:
            self.assertTrue(False)

        try:
            pca_X = PCA(X, scaling='pareto', n_components=0)
            principal_components = pca_X.transform(X)
            X_rec = pca_X.reconstruct(principal_components)
            difference = abs(X - X_rec)
            comparison = difference < 10**(-14)
            self.assertTrue(comparison.all())
        except Exception:
            self.assertTrue(False)

        try:
            pca_X = PCA(X, scaling='max', n_components=0)
            principal_components = pca_X.transform(X)
            X_rec = pca_X.reconstruct(principal_components)
            difference = abs(X - X_rec)
            comparison = difference < 10**(-14)
            self.assertTrue(comparison.all())
        except Exception:
            self.assertTrue(False)

        try:
            pca_X = PCA(X, scaling='level', n_components=0)
            principal_components = pca_X.transform(X)
            X_rec = pca_X.reconstruct(principal_components)
            difference = abs(X - X_rec)
            comparison = difference < 10**(-14)
            self.assertTrue(comparison.all())
        except Exception:
            self.assertTrue(False)

        try:
            pca_X = PCA(X, scaling='-1to1', n_components=0)
            principal_components = pca_X.transform(X)
            X_rec = pca_X.reconstruct(principal_components)
            difference = abs(X - X_rec)
            comparison = difference < 10**(-14)
            self.assertTrue(comparison.all())
        except Exception:
            self.assertTrue(False)

        try:
            pca_X = PCA(X, scaling='poisson', n_components=0)
            principal_components = pca_X.transform(X)
            X_rec = pca_X.reconstruct(principal_components)
            difference = abs(X - X_rec)
            comparison = difference < 10**(-14)
            self.assertTrue(comparison.all())
        except Exception:
            self.assertTrue(False)

        try:
            pca_X = PCA(X, scaling='vast_2', n_components=0)
            principal_components = pca_X.transform(X)
            X_rec = pca_X.reconstruct(principal_components)
            difference = abs(X - X_rec)
            comparison = difference < 10**(-14)
            self.assertTrue(comparison.all())
        except Exception:
            self.assertTrue(False)

        try:
            pca_X = PCA(X, scaling='vast_3', n_components=0)
            principal_components = pca_X.transform(X)
            X_rec = pca_X.reconstruct(principal_components)
            difference = abs(X - X_rec)
            comparison = difference < 10**(-14)
            self.assertTrue(comparison.all())
        except Exception:
            self.assertTrue(False)

        try:
            pca_X = PCA(X, scaling='vast_4', n_components=0)
            principal_components = pca_X.transform(X)
            X_rec = pca_X.reconstruct(principal_components)
            difference = abs(X - X_rec)
            comparison = difference < 10**(-14)
            self.assertTrue(comparison.all())
        except Exception:
            self.assertTrue(False)

    def test_transform_reconstruct_on_all_available_scalings_with_no_centering(self):

        X = np.random.rand(100,10)

        try:
            pca_X = PCA(X, scaling='none', n_components=0)
            principal_components = pca_X.transform(X, nocenter=True)
            X_rec = pca_X.reconstruct(principal_components, nocenter=True)
            difference = abs(X - X_rec)
            comparison = difference < 10**(-14)
            self.assertTrue(comparison.all())
        except Exception:
            self.assertTrue(False)

        try:
            pca_X = PCA(X, scaling='auto', n_components=0)
            principal_components = pca_X.transform(X, nocenter=True)
            X_rec = pca_X.reconstruct(principal_components, nocenter=True)
            difference = abs(X - X_rec)
            comparison = difference < 10**(-14)
            self.assertTrue(comparison.all())
        except Exception:
            self.assertTrue(False)

        try:
            pca_X = PCA(X, scaling='range', n_components=0)
            principal_components = pca_X.transform(X, nocenter=True)
            X_rec = pca_X.reconstruct(principal_components, nocenter=True)
            difference = abs(X - X_rec)
            comparison = difference < 10**(-14)
            self.assertTrue(comparison.all())
        except Exception:
            self.assertTrue(False)

        try:
            pca_X = PCA(X, scaling='vast', n_components=0)
            principal_components = pca_X.transform(X, nocenter=True)
            X_rec = pca_X.reconstruct(principal_components, nocenter=True)
            difference = abs(X - X_rec)
            comparison = difference < 10**(-14)
            self.assertTrue(comparison.all())
        except Exception:
            self.assertTrue(False)

        try:
            pca_X = PCA(X, scaling='pareto', n_components=0)
            principal_components = pca_X.transform(X, nocenter=True)
            X_rec = pca_X.reconstruct(principal_components, nocenter=True)
            difference = abs(X - X_rec)
            comparison = difference < 10**(-14)
            self.assertTrue(comparison.all())
        except Exception:
            self.assertTrue(False)

        try:
            pca_X = PCA(X, scaling='max', n_components=0)
            principal_components = pca_X.transform(X, nocenter=True)
            X_rec = pca_X.reconstruct(principal_components, nocenter=True)
            difference = abs(X - X_rec)
            comparison = difference < 10**(-14)
            self.assertTrue(comparison.all())
        except Exception:
            self.assertTrue(False)

        try:
            pca_X = PCA(X, scaling='level', n_components=0)
            principal_components = pca_X.transform(X, nocenter=True)
            X_rec = pca_X.reconstruct(principal_components, nocenter=True)
            difference = abs(X - X_rec)
            comparison = difference < 10**(-14)
            self.assertTrue(comparison.all())
        except Exception:
            self.assertTrue(False)

        try:
            pca_X = PCA(X, scaling='-1to1', n_components=0)
            principal_components = pca_X.transform(X, nocenter=True)
            X_rec = pca_X.reconstruct(principal_components, nocenter=True)
            difference = abs(X - X_rec)
            comparison = difference < 10**(-14)
            self.assertTrue(comparison.all())
        except Exception:
            self.assertTrue(False)

        try:
            pca_X = PCA(X, scaling='poisson', n_components=0)
            principal_components = pca_X.transform(X, nocenter=True)
            X_rec = pca_X.reconstruct(principal_components, nocenter=True)
            difference = abs(X - X_rec)
            comparison = difference < 10**(-14)
            self.assertTrue(comparison.all())
        except Exception:
            self.assertTrue(False)

        try:
            pca_X = PCA(X, scaling='vast_2', n_components=0)
            principal_components = pca_X.transform(X, nocenter=True)
            X_rec = pca_X.reconstruct(principal_components, nocenter=True)
            difference = abs(X - X_rec)
            comparison = difference < 10**(-14)
            self.assertTrue(comparison.all())
        except Exception:
            self.assertTrue(False)

        try:
            pca_X = PCA(X, scaling='vast_3', n_components=0)
            principal_components = pca_X.transform(X, nocenter=True)
            X_rec = pca_X.reconstruct(principal_components, nocenter=True)
            difference = abs(X - X_rec)
            comparison = difference < 10**(-14)
            self.assertTrue(comparison.all())
        except Exception:
            self.assertTrue(False)

        try:
            pca_X = PCA(X, scaling='vast_4', n_components=0)
            principal_components = pca_X.transform(X, nocenter=True)
            X_rec = pca_X.reconstruct(principal_components, nocenter=True)
            difference = abs(X - X_rec)
            comparison = difference < 10**(-14)
            self.assertTrue(comparison.all())
        except Exception:
            self.assertTrue(False)

    def test_transform_reconstruct_on_all_available_scalings_using_different_X(self):

        X_init = np.random.rand(100,10)
        X = np.random.rand(60,10)

        try:
            pca_X = PCA(X_init, scaling='none', n_components=0)
            principal_components = pca_X.transform(X)
            X_rec = pca_X.reconstruct(principal_components)
            difference = abs(X - X_rec)
            comparison = difference < 10**(-14)
            self.assertTrue(comparison.all())
        except Exception:
            self.assertTrue(False)

        try:
            pca_X = PCA(X_init, scaling='auto', n_components=0)
            principal_components = pca_X.transform(X)
            X_rec = pca_X.reconstruct(principal_components)
            difference = abs(X - X_rec)
            comparison = difference < 10**(-14)
            self.assertTrue(comparison.all())
        except Exception:
            self.assertTrue(False)

        try:
            pca_X = PCA(X_init, scaling='range', n_components=0)
            principal_components = pca_X.transform(X)
            X_rec = pca_X.reconstruct(principal_components)
            difference = abs(X - X_rec)
            comparison = difference < 10**(-14)
            self.assertTrue(comparison.all())
        except Exception:
            self.assertTrue(False)

        try:
            pca_X = PCA(X_init, scaling='vast', n_components=0)
            principal_components = pca_X.transform(X)
            X_rec = pca_X.reconstruct(principal_components)
            difference = abs(X - X_rec)
            comparison = difference < 10**(-14)
            self.assertTrue(comparison.all())
        except Exception:
            self.assertTrue(False)

        try:
            pca_X = PCA(X_init, scaling='pareto', n_components=0)
            principal_components = pca_X.transform(X)
            X_rec = pca_X.reconstruct(principal_components)
            difference = abs(X - X_rec)
            comparison = difference < 10**(-14)
            self.assertTrue(comparison.all())
        except Exception:
            self.assertTrue(False)

        try:
            pca_X = PCA(X_init, scaling='max', n_components=0)
            principal_components = pca_X.transform(X)
            X_rec = pca_X.reconstruct(principal_components)
            difference = abs(X - X_rec)
            comparison = difference < 10**(-14)
            self.assertTrue(comparison.all())
        except Exception:
            self.assertTrue(False)

        try:
            pca_X = PCA(X_init, scaling='level', n_components=0)
            principal_components = pca_X.transform(X)
            X_rec = pca_X.reconstruct(principal_components)
            difference = abs(X - X_rec)
            comparison = difference < 10**(-14)
            self.assertTrue(comparison.all())
        except Exception:
            self.assertTrue(False)

        try:
            pca_X = PCA(X, scaling='-1to1', n_components=0)
            principal_components = pca_X.transform(X)
            X_rec = pca_X.reconstruct(principal_components)
            difference = abs(X - X_rec)
            comparison = difference < 10**(-14)
            self.assertTrue(comparison.all())
        except Exception:
            self.assertTrue(False)

        try:
            pca_X = PCA(X_init, scaling='poisson', n_components=0)
            principal_components = pca_X.transform(X)
            X_rec = pca_X.reconstruct(principal_components)
            difference = abs(X - X_rec)
            comparison = difference < 10**(-14)
            self.assertTrue(comparison.all())
        except Exception:
            self.assertTrue(False)

        try:
            pca_X = PCA(X_init, scaling='vast_2', n_components=0)
            principal_components = pca_X.transform(X)
            X_rec = pca_X.reconstruct(principal_components)
            difference = abs(X - X_rec)
            comparison = difference < 10**(-14)
            self.assertTrue(comparison.all())
        except Exception:
            self.assertTrue(False)

        try:
            pca_X = PCA(X_init, scaling='vast_3', n_components=0)
            principal_components = pca_X.transform(X)
            X_rec = pca_X.reconstruct(principal_components)
            difference = abs(X - X_rec)
            comparison = difference < 10**(-14)
            self.assertTrue(comparison.all())
        except Exception:
            self.assertTrue(False)

        try:
            pca_X = PCA(X_init, scaling='vast_4', n_components=0)
            principal_components = pca_X.transform(X)
            X_rec = pca_X.reconstruct(principal_components)
            difference = abs(X - X_rec)
            comparison = difference < 10**(-14)
            self.assertTrue(comparison.all())
        except Exception:
            self.assertTrue(False)

    def test_transform_reconstruct_on_all_available_scalings_using_different_X_with_no_centering(self):

        X_init = np.random.rand(100,10)
        X = np.random.rand(60,10)

        try:
            pca_X = PCA(X_init, scaling='none', n_components=0)
            principal_components = pca_X.transform(X, nocenter=True)
            X_rec = pca_X.reconstruct(principal_components, nocenter=True)
            difference = abs(X - X_rec)
            comparison = difference < 10**(-14)
            self.assertTrue(comparison.all())
        except Exception:
            self.assertTrue(False)

        try:
            pca_X = PCA(X_init, scaling='auto', n_components=0)
            principal_components = pca_X.transform(X, nocenter=True)
            X_rec = pca_X.reconstruct(principal_components, nocenter=True)
            difference = abs(X - X_rec)
            comparison = difference < 10**(-14)
            self.assertTrue(comparison.all())
        except Exception:
            self.assertTrue(False)

        try:
            pca_X = PCA(X_init, scaling='range', n_components=0)
            principal_components = pca_X.transform(X, nocenter=True)
            X_rec = pca_X.reconstruct(principal_components, nocenter=True)
            difference = abs(X - X_rec)
            comparison = difference < 10**(-14)
            self.assertTrue(comparison.all())
        except Exception:
            self.assertTrue(False)

        try:
            pca_X = PCA(X_init, scaling='vast', n_components=0)
            principal_components = pca_X.transform(X, nocenter=True)
            X_rec = pca_X.reconstruct(principal_components, nocenter=True)
            difference = abs(X - X_rec)
            comparison = difference < 10**(-14)
            self.assertTrue(comparison.all())
        except Exception:
            self.assertTrue(False)

        try:
            pca_X = PCA(X_init, scaling='pareto', n_components=0)
            principal_components = pca_X.transform(X, nocenter=True)
            X_rec = pca_X.reconstruct(principal_components, nocenter=True)
            difference = abs(X - X_rec)
            comparison = difference < 10**(-14)
            self.assertTrue(comparison.all())
        except Exception:
            self.assertTrue(False)

        try:
            pca_X = PCA(X_init, scaling='max', n_components=0)
            principal_components = pca_X.transform(X, nocenter=True)
            X_rec = pca_X.reconstruct(principal_components, nocenter=True)
            difference = abs(X - X_rec)
            comparison = difference < 10**(-14)
            self.assertTrue(comparison.all())
        except Exception:
            self.assertTrue(False)

        try:
            pca_X = PCA(X_init, scaling='level', n_components=0)
            principal_components = pca_X.transform(X, nocenter=True)
            X_rec = pca_X.reconstruct(principal_components, nocenter=True)
            difference = abs(X - X_rec)
            comparison = difference < 10**(-14)
            self.assertTrue(comparison.all())
        except Exception:
            self.assertTrue(False)

        try:
            pca_X = PCA(X, scaling='-1to1', n_components=0)
            principal_components = pca_X.transform(X, nocenter=True)
            X_rec = pca_X.reconstruct(principal_components, nocenter=True)
            difference = abs(X - X_rec)
            comparison = difference < 10**(-14)
            self.assertTrue(comparison.all())
        except Exception:
            self.assertTrue(False)

        try:
            pca_X = PCA(X_init, scaling='poisson', n_components=0)
            principal_components = pca_X.transform(X, nocenter=True)
            X_rec = pca_X.reconstruct(principal_components, nocenter=True)
            difference = abs(X - X_rec)
            comparison = difference < 10**(-14)
            self.assertTrue(comparison.all())
        except Exception:
            self.assertTrue(False)

        try:
            pca_X = PCA(X_init, scaling='vast_2', n_components=0)
            principal_components = pca_X.transform(X)
            X_rec = pca_X.reconstruct(principal_components)
            difference = abs(X - X_rec)
            comparison = difference < 10**(-14)
            self.assertTrue(comparison.all())
        except Exception:
            self.assertTrue(False)

        try:
            pca_X = PCA(X_init, scaling='vast_3', n_components=0)
            principal_components = pca_X.transform(X, nocenter=True)
            X_rec = pca_X.reconstruct(principal_components, nocenter=True)
            difference = abs(X - X_rec)
            comparison = difference < 10**(-14)
            self.assertTrue(comparison.all())
        except Exception:
            self.assertTrue(False)

        try:
            pca_X = PCA(X_init, scaling='vast_4', n_components=0)
            principal_components = pca_X.transform(X, nocenter=True)
            X_rec = pca_X.reconstruct(principal_components, nocenter=True)
            difference = abs(X - X_rec)
            comparison = difference < 10**(-14)
            self.assertTrue(comparison.all())
        except Exception:
            self.assertTrue(False)

    def test_u_scores_allowed_calls(self):

        X = np.random.rand(100,10)

        try:
            pca = PCA(X, scaling='auto')
            u_scores = pca.u_scores(X)
            (n_obs, n_u_scores) = np.shape(u_scores)
            self.assertTrue(n_u_scores==10)
        except Exception:
            self.assertTrue(False)

        try:
            pca = PCA(X, scaling='auto', n_components=4)
            u_scores = pca.u_scores(X)
            (n_obs, n_u_scores) = np.shape(u_scores)
            self.assertTrue(n_u_scores==4)
        except Exception:
            self.assertTrue(False)

        try:
            pca = PCA(X, scaling='auto', n_components=1)
            u_scores = pca.u_scores(X)
            (n_obs, n_u_scores) = np.shape(u_scores)
            self.assertTrue(n_u_scores==1)
        except Exception:
            self.assertTrue(False)

        try:
            pca = PCA(X, scaling='pareto', n_components=10)
            u_scores = pca.u_scores(X)
            (n_obs, n_u_scores) = np.shape(u_scores)
            self.assertTrue(n_u_scores==10)
        except Exception:
            self.assertTrue(False)

        try:
            pca = PCA(X, scaling='auto', n_components=4)
            X_new = np.random.rand(50,10)
            u_scores = pca.u_scores(X_new)
            (n_obs, n_u_scores) = np.shape(u_scores)
            self.assertTrue(n_u_scores==4)
        except Exception:
            self.assertTrue(False)

    def test_u_scores_not_allowed_calls(self):

        X = np.random.rand(20,4)
        X_2 = np.random.rand(20,3)
        X_3 = np.random.rand(20,5)

        pca = PCA(X, scaling='auto')

        with self.assertRaises(ValueError):
            u_scores = pca.u_scores(X_2)

        with self.assertRaises(ValueError):
            u_scores = pca.u_scores(X_3)

    def test_w_scores_allowed_calls(self):

        X = np.random.rand(100,10)

        pca = PCA(X, scaling='auto')

        try:
            w_scores = pca.w_scores(X)
        except Exception:
            self.assertTrue(False)

        try:
            pca.n_components = 5
            w_scores = pca.w_scores(X)
            (n_observations, n_w_scores) = np.shape(w_scores)
            self.assertTrue(n_w_scores == 5)
        except Exception:
            self.assertTrue(False)

        try:
            pca.n_components = 0
            w_scores = pca.w_scores(X)
            (n_observations, n_w_scores) = np.shape(w_scores)
            self.assertTrue(n_w_scores == 10)
        except Exception:
            self.assertTrue(False)

    def test_w_scores_not_allowed_calls(self):

        X = np.random.rand(20,4)
        X_2 = np.random.rand(20,3)
        X_3 = np.random.rand(20,5)

        pca = PCA(X, scaling='auto')

        with self.assertRaises(ValueError):
            u_scores = pca.w_scores(X_2)

        with self.assertRaises(ValueError):
            u_scores = pca.w_scores(X_3)

    def test_calculate_r2_allowed_calls(self):

        test_data_set = np.random.rand(100,20)
        r2_test = np.ones((20,))

        try:
            pca_X = PCA(test_data_set, scaling='auto', n_components=20, use_eigendec=True, nocenter=False)
            r2_values = pca_X.calculate_r2(test_data_set)
            comparison = r2_values == r2_test
            self.assertTrue(comparison.all())
        except Exception:
            self.assertTrue(False)

    def test_r2_convergence_allowed_calls(self):

        X = np.random.rand(100,3)

        pca = PCA(X, scaling='auto')

        try:
            r2 = pca.r2_convergence(X, 3, variable_names=[], print_width=10, verbose=False, save_filename=None)
        except Exception:
            self.assertTrue(False)

        try:
            r2 = pca.r2_convergence(X, 3, variable_names=['a', 'b', 'c'], print_width=10, verbose=False, save_filename=None)
        except Exception:
            self.assertTrue(False)

        try:
            r2 = pca.r2_convergence(X, 1, variable_names=[], print_width=10, verbose=False, save_filename=None)
        except Exception:
            self.assertTrue(False)

        try:
            r2 = pca.r2_convergence(X, 1, variable_names=['a', 'b', 'c'], print_width=10, verbose=False, save_filename=None)
        except Exception:
            self.assertTrue(False)

    def test_r2_convergence_not_allowed_calls(self):
        pass

    def test_set_retained_eigenvalues_allowed_calls(self):

        X = np.random.rand(100,10)

        pca = PCA(X, scaling='auto')

        # This one is commented out since it requires user input:
        # try:
        #     pca.set_retained_eigenvalues(method='SCREE GRAPH')
        # except Exception:
        #     self.assertTrue(False)

        try:
            pca_new = pca.set_retained_eigenvalues(method='TOTAL VARIANCE', option=0.5)
        except Exception:
            self.assertTrue(False)

        try:
            pca_new = pca.set_retained_eigenvalues(method='INDIVIDUAL VARIANCE', option=0.5)
        except Exception:
            self.assertTrue(False)

        try:
            pca_new = pca.set_retained_eigenvalues(method='BROKEN STICK')
        except Exception:
            self.assertTrue(False)

    def test_set_retained_eigenvalues_not_allowed_calls(self):

        X = np.random.rand(100,10)

        pca = PCA(X, scaling='auto')

        with self.assertRaises(ValueError):
            pca.set_retained_eigenvalues(method='Method')
        with self.assertRaises(ValueError):
            pca.set_retained_eigenvalues(method='TOTAL VARIANCE', option=1.1)
        with self.assertRaises(ValueError):
            pca.set_retained_eigenvalues(method='TOTAL VARIANCE', option=-0.1)
        with self.assertRaises(ValueError):
            pca.set_retained_eigenvalues(method='INDIVIDUAL VARIANCE', option=1.1)
        with self.assertRaises(ValueError):
            pca.set_retained_eigenvalues(method='INDIVIDUAL VARIANCE', option=-0.1)

    def test_principal_variables_allowed_calls(self):

        X = np.random.rand(100,10)

        pca = PCA(X, scaling='auto')

        try:
            principal_variables_indices = pca.principal_variables(method='B2')
        except Exception:
            self.assertTrue(False)

        try:
            principal_variables_indices = pca.principal_variables(method='B4')
        except Exception:
            self.assertTrue(False)

        try:
            principal_variables_indices = pca.principal_variables(method='M2', x=X)
        except Exception:
            self.assertTrue(False)

    def test_principal_variables_not_allowed_calls(self):

        X = np.random.rand(100,10)

        pca = PCA(X, scaling='auto')

        with self.assertRaises(ValueError):
            pca.principal_variables(method='M2')
        with self.assertRaises(ValueError):
            pca.principal_variables(method='Method')

    def test_data_consistency_check_allowed_calls(self):

        X = np.random.rand(100,20)
        pca_X = PCA(X, scaling='auto', n_components=10)

        try:
            X_1 = np.random.rand(50,20)
            is_consistent = pca_X.data_consistency_check(X_1)
            self.assertTrue(is_consistent==True)
        except Exception:
            self.assertTrue(False)

        try:
            X_2 = np.random.rand(100,10)
            is_consistent = pca_X.data_consistency_check(X_2)
            self.assertTrue(is_consistent==False)
        except Exception:
            self.assertTrue(False)

        X_3 = np.random.rand(100,10)
        with self.assertRaises(ValueError):
            is_consistent = pca_X.data_consistency_check(X_3, errors_are_fatal=True)

        try:
            X_4 = np.random.rand(80,20)
            is_consistent = pca_X.data_consistency_check(X_4, errors_are_fatal=True)
            self.assertTrue(is_consistent==True)
        except Exception:
            self.assertTrue(False)

    def test_data_consistency_check_not_allowed_calls(self):

        X = np.random.rand(100,20)
        pca_X = PCA(X, scaling='auto', n_components=10)

        with self.assertRaises(ValueError):
            is_consistent = pca_X.data_consistency_check(X, errors_are_fatal=1)

        with self.assertRaises(ValueError):
            is_consistent = pca_X.data_consistency_check(X, errors_are_fatal=0)

    def test_simulate_chemical_source_term_handling(self):

        X = np.random.rand(200,10)
        X_source = np.random.rand(200,10)

        pca = PCA(X, scaling='auto')

        try:
            PC_source = pca.transform(X_source, nocenter=True)
            PC_source_rec = pca.reconstruct(PC_source, nocenter=True)

            difference = abs(X_source - PC_source_rec)
            comparison = difference < 10**(-14)
            self.assertTrue(comparison.all())
        except Exception:
            self.assertTrue(False)

################################################################################
#
# Test LPCA class
#
################################################################################

    def test_LPCA__allowed_calls(self):

        X = np.random.rand(100,10)
        idx = np.zeros((100,))
        idx[50:80] = 1
        idx = idx.astype(int)

        try:
            lpca_X = LPCA(X, idx, scaling='none', n_components=2)
            A = lpca_X.A
            L = lpca_X.L
            Z = lpca_X.principal_components
            A_k1 = lpca_X.A[0]
            L_k1 = lpca_X.L[0]
            Z_k1 = lpca_X.principal_components[0]
            A1_k1 = lpca_X.A[0][:,0]
            L1_k1 = lpca_X.L[0][0]
            Z1_k1 = lpca_X.principal_components[0][:,0]
        except Exception:
            self.assertTrue(False)

        idx = np.zeros((100,1))
        idx[50:80] = 1
        idx = idx.astype(int)

        try:
            lpca_X = LPCA(X, idx, scaling='none', n_components=2)
            A = lpca_X.A
            L = lpca_X.L
            Z = lpca_X.principal_components
            A_k1 = lpca_X.A[0]
            L_k1 = lpca_X.L[0]
            Z_k1 = lpca_X.principal_components[0]
            A1_k1 = lpca_X.A[0][:,0]
            L1_k1 = lpca_X.L[0][0]
            Z1_k1 = lpca_X.principal_components[0][:,0]
        except Exception:
            self.assertTrue(False)

    def test_LPCA_equivalent_to_PCA_for_one_cluster(self):

        X = np.random.rand(100,10)
        idx = np.zeros((100,))
        idx = idx.astype(int)

        try:
            lpca_X = LPCA(X, idx, scaling='none', n_components=2)
            pca_X = PCA(X, scaling='none', n_components=2)
            lpca_A = lpca_X.A[0]
            pca_A = pca_X.A[:,0:2]
            self.assertTrue(np.array_equal(lpca_A, pca_A))
            lpca_L = lpca_X.L[0]
            pca_L = pca_X.L[0:2]
            self.assertTrue(np.array_equal(lpca_L, pca_L))
            lpca_Z = lpca_X.principal_components[0]
            pca_Z = pca_X.transform(X)
            self.assertTrue(np.array_equal(lpca_Z, pca_Z))
        except:
            self.assertTrue(False)

        try:
            lpca_X = LPCA(X, idx, scaling='range', n_components=2)
            pca_X = PCA(X, scaling='range', n_components=2)
            lpca_A = lpca_X.A[0]
            pca_A = pca_X.A[:,0:2]
            self.assertTrue(np.array_equal(lpca_A, pca_A))
            lpca_L = lpca_X.L[0]
            pca_L = pca_X.L[0:2]
            self.assertTrue(np.array_equal(lpca_L, pca_L))
            lpca_Z = lpca_X.principal_components[0]
            pca_Z = pca_X.transform(X)
            self.assertTrue(np.array_equal(lpca_Z, pca_Z))
        except:
            self.assertTrue(False)

        try:
            lpca_X = LPCA(X, idx, scaling='auto', n_components=0)
            pca_X = PCA(X, scaling='auto', n_components=0)
            lpca_A = lpca_X.A[0]
            pca_A = pca_X.A
            self.assertTrue(np.array_equal(lpca_A, pca_A))
            lpca_L = lpca_X.L[0]
            pca_L = pca_X.L
            self.assertTrue(np.array_equal(lpca_L, pca_L))
            lpca_Z = lpca_X.principal_components[0]
            pca_Z = pca_X.transform(X)
            self.assertTrue(np.array_equal(lpca_Z, pca_Z))
        except:
            self.assertTrue(False)

        X = np.random.rand(100,10)
        idx = np.zeros((100,1))
        idx = idx.astype(int)

        try:
            lpca_X = LPCA(X, idx, scaling='none', n_components=2)
            pca_X = PCA(X, scaling='none', n_components=2)
            lpca_A = lpca_X.A[0]
            pca_A = pca_X.A[:,0:2]
            self.assertTrue(np.array_equal(lpca_A, pca_A))
            lpca_L = lpca_X.L[0]
            pca_L = pca_X.L[0:2]
            self.assertTrue(np.array_equal(lpca_L, pca_L))
            lpca_Z = lpca_X.principal_components[0]
            pca_Z = pca_X.transform(X)
            self.assertTrue(np.array_equal(lpca_Z, pca_Z))
        except:
            self.assertTrue(False)

        try:
            lpca_X = LPCA(X, idx, scaling='range', n_components=2)
            pca_X = PCA(X, scaling='range', n_components=2)
            lpca_A = lpca_X.A[0]
            pca_A = pca_X.A[:,0:2]
            self.assertTrue(np.array_equal(lpca_A, pca_A))
            lpca_L = lpca_X.L[0]
            pca_L = pca_X.L[0:2]
            self.assertTrue(np.array_equal(lpca_L, pca_L))
            lpca_Z = lpca_X.principal_components[0]
            pca_Z = pca_X.transform(X)
            self.assertTrue(np.array_equal(lpca_Z, pca_Z))
        except:
            self.assertTrue(False)

        try:
            lpca_X = LPCA(X, idx, scaling='auto', n_components=0)
            pca_X = PCA(X, scaling='auto', n_components=0)
            lpca_A = lpca_X.A[0]
            pca_A = pca_X.A
            self.assertTrue(np.array_equal(lpca_A, pca_A))
            lpca_L = lpca_X.L[0]
            pca_L = pca_X.L
            self.assertTrue(np.array_equal(lpca_L, pca_L))
            lpca_Z = lpca_X.principal_components[0]
            pca_Z = pca_X.transform(X)
            self.assertTrue(np.array_equal(lpca_Z, pca_Z))
        except:
            self.assertTrue(False)

    def test_LPCA__not_allowed_calls(self):

        X = np.random.rand(100,10)
        idx = np.zeros((100,))
        idx[50:80] = 1
        idx = idx.astype(int)
        lpca_X = LPCA(X, idx, scaling='none', n_components=5)

        with self.assertRaises(IndexError):
            A = lpca_X.A[2]

        with self.assertRaises(IndexError):
            L = lpca_X.L[2]

        with self.assertRaises(IndexError):
            Z = lpca_X.principal_components[2]

        with self.assertRaises(IndexError):
            A = lpca_X.A[0][:,8]

        with self.assertRaises(IndexError):
            L = lpca_X.L[0][8]

        with self.assertRaises(IndexError):
            Z = lpca_X.principal_components[0][:,8]

        with self.assertRaises(ValueError):
            lpca_X = LPCA([1,2,3], idx, scaling='none', n_components=5)

        with self.assertRaises(ValueError):
            lpca_X = LPCA(X, [1,2,3], scaling='none', n_components=5)

        with self.assertRaises(ValueError):
            lpca_X = LPCA(X, idx, scaling=1, n_components=5)

        with self.assertRaises(ValueError):
            lpca_X = LPCA(X, idx, scaling='none', n_components=20)

        X = np.random.rand(100,10)
        idx = np.zeros((200,))
        idx[50:80] = 1
        idx = idx.astype(int)

        with self.assertRaises(ValueError):
            lpca_X = LPCA(X, idx)

        X = np.random.rand(100,10)
        idx = np.zeros((100,2))
        idx = idx.astype(int)

        with self.assertRaises(ValueError):
            lpca_X = LPCA(X, idx)

    def test_LPCA__not_allowed_attribute_set(self):

        X = np.random.rand(100,5)
        idx = np.zeros((100,))
        idx[50:80] = 1
        idx = idx.astype(int)
        lpca_X = LPCA(X, idx)

        with self.assertRaises(AttributeError):
            lpca_X.A = 1

        with self.assertRaises(AttributeError):
            lpca_X.L = 1

        with self.assertRaises(AttributeError):
            lpca_X.principal_components = 1

    def test_LPCA_local_correlation__allowed_calls(self):

        X = np.random.rand(100,5)
        idx = np.zeros((100,))
        idx[50:80] = 1
        idx = idx.astype(int)

        try:
            lpca = LPCA(X, idx)
            (local_correlations, weighted, unweighted) = lpca.local_correlation(X[:,0])
        except:
            self.assertTrue(False)

        try:
            lpca = LPCA(X, idx)
            (local_correlations, weighted, unweighted) = lpca.local_correlation(X[:,0:1])
        except:
            self.assertTrue(False)

        try:
            lpca = LPCA(X, idx)
            (local_correlations, weighted, unweighted) = lpca.local_correlation(X[:,0], index=1, metric='pearson')
        except:
            self.assertTrue(False)

        try:
            lpca = LPCA(X, idx, n_components=2)
            (local_correlations, weighted, unweighted) = lpca.local_correlation(X[:,0], index=1, metric='pearson')
        except:
            self.assertTrue(False)

        try:
            lpca = LPCA(X, idx, n_components=0)
            (local_correlations, weighted, unweighted) = lpca.local_correlation(X[:,0], index=4, metric='pearson')
        except:
            self.assertTrue(False)

    def test_LPCA_local_correlation__not_allowed_calls(self):

        X = np.random.rand(100,5)
        idx = np.zeros((100,))
        idx[50:80] = 1
        idx = idx.astype(int)
        lpca = LPCA(X, idx, n_components=2)

        with self.assertRaises(ValueError):
            (local_correlations, weighted, unweighted) = lpca.local_correlation(X[:,0:2])

        with self.assertRaises(ValueError):
            (local_correlations, weighted, unweighted) = lpca.local_correlation(X[0:50,0])

        with self.assertRaises(ValueError):
            (local_correlations, weighted, unweighted) = lpca.local_correlation(X[:,0], index=-1)

        with self.assertRaises(ValueError):
            (local_correlations, weighted, unweighted) = lpca.local_correlation(X[:,0], index=2)

        with self.assertRaises(ValueError):
            (local_correlations, weighted, unweighted) = lpca.local_correlation(X[:,0], metric='none')

        with self.assertRaises(ValueError):
            (local_correlations, weighted, unweighted) = lpca.local_correlation(X[:,0], verbose=1)

################################################################################
#
# Test PCA on sampled data sets functionalities of the `reduction` module
#
################################################################################

    def test_pca_on_sampled_data_set_allowed_calls(self):

        X = np.random.rand(200,20)
        idx_X_r = np.arange(91,151,1)

        try:
            (eigenvalues, eigenvectors, pc_scores, pc_sources, C, D, C_r, D_r) = reduction.pca_on_sampled_data_set(X, idx_X_r, 'auto', 2, 1, X_source=[])
            (eigenvalues, eigenvectors, pc_scores, pc_sources, C, D, C_r, D_r) = reduction.pca_on_sampled_data_set(X, idx_X_r, 'auto', 2, 2, X_source=[])
            (eigenvalues, eigenvectors, pc_scores, pc_sources, C, D, C_r, D_r) = reduction.pca_on_sampled_data_set(X, idx_X_r, 'auto', 2, 3, X_source=[])
            (eigenvalues, eigenvectors, pc_scores, pc_sources, C, D, C_r, D_r) = reduction.pca_on_sampled_data_set(X, idx_X_r, 'auto', 2, 4, X_source=[])
        except Exception:
            self.assertTrue(False)

        try:
            (eigenvalues, eigenvectors, pc_scores, pc_sources, C, D, C_r, D_r) = reduction.pca_on_sampled_data_set(X, idx_X_r, 'range', 2, 1, X_source=[])
            (eigenvalues, eigenvectors, pc_scores, pc_sources, C, D, C_r, D_r) = reduction.pca_on_sampled_data_set(X, idx_X_r, 'range', 2, 2, X_source=[])
            (eigenvalues, eigenvectors, pc_scores, pc_sources, C, D, C_r, D_r) = reduction.pca_on_sampled_data_set(X, idx_X_r, 'range', 2, 3, X_source=[])
            (eigenvalues, eigenvectors, pc_scores, pc_sources, C, D, C_r, D_r) = reduction.pca_on_sampled_data_set(X, idx_X_r, 'range', 2, 4, X_source=[])
        except Exception:
            self.assertTrue(False)

        try:
            (eigenvalues, eigenvectors, pc_scores, pc_sources, C, D, C_r, D_r) = reduction.pca_on_sampled_data_set(X, idx_X_r, 'auto', 1, 1, X_source=[])
            (eigenvalues, eigenvectors, pc_scores, pc_sources, C, D, C_r, D_r) = reduction.pca_on_sampled_data_set(X, idx_X_r, 'auto', 1, 2, X_source=[])
            (eigenvalues, eigenvectors, pc_scores, pc_sources, C, D, C_r, D_r) = reduction.pca_on_sampled_data_set(X, idx_X_r, 'auto', 1, 3, X_source=[])
            (eigenvalues, eigenvectors, pc_scores, pc_sources, C, D, C_r, D_r) = reduction.pca_on_sampled_data_set(X, idx_X_r, 'auto', 1, 4, X_source=[])
        except Exception:
            self.assertTrue(False)

        X_source = np.random.rand(200,20)
        try:
            (eigenvalues, eigenvectors, pc_scores, pc_sources, C, D, C_r, D_r) = reduction.pca_on_sampled_data_set(X, idx_X_r, 'auto', 1, 1, X_source=X_source)
            (eigenvalues, eigenvectors, pc_scores, pc_sources, C, D, C_r, D_r) = reduction.pca_on_sampled_data_set(X, idx_X_r, 'auto', 1, 2, X_source=X_source)
            (eigenvalues, eigenvectors, pc_scores, pc_sources, C, D, C_r, D_r) = reduction.pca_on_sampled_data_set(X, idx_X_r, 'auto', 1, 3, X_source=X_source)
            (eigenvalues, eigenvectors, pc_scores, pc_sources, C, D, C_r, D_r) = reduction.pca_on_sampled_data_set(X, idx_X_r, 'auto', 1, 4, X_source=X_source)
        except Exception:
            self.assertTrue(False)

        try:
            (eigenvalues, eigenvectors, pc_scores, pc_sources, C, D, C_r, D_r) = reduction.pca_on_sampled_data_set(X, idx_X_r, 'pareto', 10, 1, X_source=X_source)
            (eigenvalues, eigenvectors, pc_scores, pc_sources, C, D, C_r, D_r) = reduction.pca_on_sampled_data_set(X, idx_X_r, 'pareto', 10, 2, X_source=X_source)
            (eigenvalues, eigenvectors, pc_scores, pc_sources, C, D, C_r, D_r) = reduction.pca_on_sampled_data_set(X, idx_X_r, 'pareto', 10, 3, X_source=X_source)
            (eigenvalues, eigenvectors, pc_scores, pc_sources, C, D, C_r, D_r) = reduction.pca_on_sampled_data_set(X, idx_X_r, 'pareto', 10, 4, X_source=X_source)
        except Exception:
            self.assertTrue(False)

    def test_pca_on_sampled_data_set_not_allowed_calls(self):

        X = np.random.rand(200,20)
        idx_X_r = np.arange(91,151,1)

        with self.assertRaises(ValueError):
            (eigenvalues, eigenvectors, pc_scores, pc_sources, C, D, C_r, D_r) = reduction.pca_on_sampled_data_set(X, idx_X_r, 'auto', 2, 5, X_source=[])

        with self.assertRaises(ValueError):
            (eigenvalues, eigenvectors, pc_scores, pc_sources, C, D, C_r, D_r) = reduction.pca_on_sampled_data_set(X, idx_X_r, 'auto', 2, 25, X_source=[])

    def test_equilibrate_cluster_populations_allowed_calls(self):

        X = np.random.rand(200,20)
        idx = np.zeros((200,))
        idx[20:60,] = 1
        idx[150:190] = 2

        idx = idx.astype(int)

        try:
            (eigenvalues, eigenvectors_matrix, pc_scores_matrix, pc_sources_matrix, idx_train, C_r, D_r) = reduction.equilibrate_cluster_populations(X, idx, 'auto', 2, 1, X_source=[], n_iterations=10, stop_iter=0, random_seed=None, verbose=False)
            (eigenvalues, eigenvectors_matrix, pc_scores_matrix, pc_sources_matrix, idx_train, C_r, D_r) = reduction.equilibrate_cluster_populations(X, idx, 'auto', 2, 2, X_source=[], n_iterations=10, stop_iter=0, random_seed=None, verbose=False)
            (eigenvalues, eigenvectors_matrix, pc_scores_matrix, pc_sources_matrix, idx_train, C_r, D_r) = reduction.equilibrate_cluster_populations(X, idx, 'auto', 2, 3, X_source=[], n_iterations=10, stop_iter=0, random_seed=None, verbose=False)
            (eigenvalues, eigenvectors_matrix, pc_scores_matrix, pc_sources_matrix, idx_train, C_r, D_r) = reduction.equilibrate_cluster_populations(X, idx, 'auto', 2, 4, X_source=[], n_iterations=10, stop_iter=0, random_seed=None, verbose=False)
        except Exception:
            self.assertTrue(False)

        try:
            (eigenvalues, eigenvectors_matrix, pc_scores_matrix, pc_sources_matrix, idx_train, C_r, D_r) = reduction.equilibrate_cluster_populations(X, idx, 'auto', 2, 1, X_source=[], n_iterations=1, stop_iter=0, random_seed=None, verbose=False)
            (eigenvalues, eigenvectors_matrix, pc_scores_matrix, pc_sources_matrix, idx_train, C_r, D_r) = reduction.equilibrate_cluster_populations(X, idx, 'auto', 2, 2, X_source=[], n_iterations=1, stop_iter=0, random_seed=None, verbose=False)
            (eigenvalues, eigenvectors_matrix, pc_scores_matrix, pc_sources_matrix, idx_train, C_r, D_r) = reduction.equilibrate_cluster_populations(X, idx, 'auto', 2, 3, X_source=[], n_iterations=1, stop_iter=0, random_seed=None, verbose=False)
            (eigenvalues, eigenvectors_matrix, pc_scores_matrix, pc_sources_matrix, idx_train, C_r, D_r) = reduction.equilibrate_cluster_populations(X, idx, 'auto', 2, 4, X_source=[], n_iterations=1, stop_iter=0, random_seed=None, verbose=False)
        except Exception:
            self.assertTrue(False)

        try:
            (eigenvalues, eigenvectors_matrix, pc_scores_matrix, pc_sources_matrix, idx_train, C_r, D_r) = reduction.equilibrate_cluster_populations(X, idx, 'range', 2, 1, X_source=[], n_iterations=1, stop_iter=0, random_seed=100, verbose=False)
            (eigenvalues, eigenvectors_matrix, pc_scores_matrix, pc_sources_matrix, idx_train, C_r, D_r) = reduction.equilibrate_cluster_populations(X, idx, 'range', 2, 2, X_source=[], n_iterations=1, stop_iter=0, random_seed=100, verbose=False)
            (eigenvalues, eigenvectors_matrix, pc_scores_matrix, pc_sources_matrix, idx_train, C_r, D_r) = reduction.equilibrate_cluster_populations(X, idx, 'range', 2, 3, X_source=[], n_iterations=1, stop_iter=0, random_seed=100, verbose=False)
            (eigenvalues, eigenvectors_matrix, pc_scores_matrix, pc_sources_matrix, idx_train, C_r, D_r) = reduction.equilibrate_cluster_populations(X, idx, 'range', 2, 4, X_source=[], n_iterations=1, stop_iter=0, random_seed=100, verbose=False)
        except Exception:
            self.assertTrue(False)

        X_source = np.random.rand(200,20)

    def test_analyze_centers_change(self):

        test_data_set = np.random.rand(100,20)
        idx_X_r = np.array([1,5,68,9,2,3,6,43,56])

        try:
            (normalized_C, normalized_C_r, center_movement_percentage, plt) = reduction.analyze_centers_change(test_data_set, idx_X_r, variable_names=[], plot_variables=[], legend_label=[], title=None, save_filename=None)
            plt.close()
        except Exception:
            self.assertTrue(False)

        try:
            (normalized_C, normalized_C_r, center_movement_percentage, plt) = reduction.analyze_centers_change(test_data_set, idx_X_r, variable_names=[], plot_variables=[1,4,5], legend_label=[], title=None, save_filename=None)
            plt.close()
        except Exception:
            self.assertTrue(False)

    def test_analyze_eigenvector_weights_change_allowed_calls(self):

        X = np.random.rand(200,20)
        idx = np.zeros((200,))
        idx[20:60,] = 1
        idx[150:190] = 2

        idx = idx.astype(int)

        try:
            (eigenvalues, eigenvectors_matrix, pc_scores_matrix, pc_sources_matrix, idx_train, C_r, D_r) = reduction.equilibrate_cluster_populations(X, idx, 'auto', 20, 1, X_source=[], n_iterations=20, stop_iter=0, random_seed=None, verbose=False)
            plt = reduction.analyze_eigenvector_weights_change(eigenvectors_matrix[:,0,:], variable_names=[], plot_variables=[], normalize=False, zero_norm=False, legend_label=[], title=None, save_filename=None)
            plt.close()
            plt = reduction.analyze_eigenvector_weights_change(eigenvectors_matrix[:,0,:], variable_names=[], plot_variables=[2,5,10], normalize=False, zero_norm=False, legend_label=[], title=None, save_filename=None)
            plt.close()
            plt = reduction.analyze_eigenvector_weights_change(eigenvectors_matrix[:,1,:], variable_names=[], plot_variables=[2,5,10], normalize=False, zero_norm=False, legend_label=[], title=None, save_filename=None)
            plt.close()
            plt = reduction.analyze_eigenvector_weights_change(eigenvectors_matrix[:,0,:], variable_names=[], plot_variables=[2,5,10], normalize=True, zero_norm=False, legend_label=[], title=None, save_filename=None)
            plt.close()
            plt = reduction.analyze_eigenvector_weights_change(eigenvectors_matrix[:,1,:], variable_names=[], plot_variables=[2,5,10], normalize=True, zero_norm=True, legend_label=[], title=None, save_filename=None)
            plt.close()
            plt = reduction.analyze_eigenvector_weights_change(eigenvectors_matrix[:,15,:], variable_names=[], plot_variables=[2,5,10], normalize=True, zero_norm=True, legend_label=[], title=None, save_filename=None)
            plt.close()
        except Exception:
            self.assertTrue(False)

        try:
            (eigenvalues, eigenvectors_matrix, pc_scores_matrix, pc_sources_matrix, idx_train, C_r, D_r) = reduction.equilibrate_cluster_populations(X, idx, 'auto', 20, 1, X_source=[], n_iterations=2, stop_iter=0, random_seed=None, verbose=False)
            plt = reduction.analyze_eigenvector_weights_change(eigenvectors_matrix[:,0,:], variable_names=[], plot_variables=[], normalize=False, zero_norm=False, legend_label=[], title=None, save_filename=None)
            plt.close()
            plt = reduction.analyze_eigenvector_weights_change(eigenvectors_matrix[:,0,:], variable_names=[], plot_variables=[2,5,10], normalize=False, zero_norm=False, legend_label=[], title=None, save_filename=None)
            plt.close()
            plt = reduction.analyze_eigenvector_weights_change(eigenvectors_matrix[:,1,:], variable_names=[], plot_variables=[2,5,10], normalize=False, zero_norm=False, legend_label=[], title=None, save_filename=None)
            plt.close()
            plt = reduction.analyze_eigenvector_weights_change(eigenvectors_matrix[:,1,:], variable_names=[], plot_variables=[2,5,10], normalize=True, zero_norm=False, legend_label=[], title=None, save_filename=None)
            plt.close()
            plt = reduction.analyze_eigenvector_weights_change(eigenvectors_matrix[:,1,:], variable_names=[], plot_variables=[2,5,10], normalize=True, zero_norm=True, legend_label=[], title=None, save_filename=None)
            plt.close()
            plt = reduction.analyze_eigenvector_weights_change(eigenvectors_matrix[:,15,:], variable_names=[], plot_variables=[2,5,10], normalize=True, zero_norm=True, legend_label=[], title=None, save_filename=None)
            plt.close()
        except Exception:
            self.assertTrue(False)

    def test_analyze_eigenvalue_distribution_allowed_calls(self):

        X = np.random.rand(200,20)
        idx_X_r = np.arange(91,151,1)

        try:
            plt = reduction.analyze_eigenvalue_distribution(X, idx_X_r, 'auto', 1, legend_label=[], title=None, save_filename=None)
            plt.close()
            plt = reduction.analyze_eigenvalue_distribution(X, idx_X_r, 'auto', 2, legend_label=[], title=None, save_filename=None)
            plt.close()
            plt = reduction.analyze_eigenvalue_distribution(X, idx_X_r, 'auto', 3, legend_label=[], title=None, save_filename=None)
            plt.close()
            plt = reduction.analyze_eigenvalue_distribution(X, idx_X_r, 'auto', 4, legend_label=[], title=None, save_filename=None)
            plt.close()
            plt = reduction.analyze_eigenvalue_distribution(X, idx_X_r, 'pareto', 1, legend_label=[], title=None, save_filename=None)
            plt.close()
            plt = reduction.analyze_eigenvalue_distribution(X, idx_X_r, 'pareto', 2, legend_label=[], title=None, save_filename=None)
            plt.close()
            plt = reduction.analyze_eigenvalue_distribution(X, idx_X_r, 'pareto', 3, legend_label=[], title=None, save_filename=None)
            plt.close()
            plt = reduction.analyze_eigenvalue_distribution(X, idx_X_r, 'pareto', 4, legend_label=[], title=None, save_filename=None)
            plt.close()
        except Exception:
            self.assertTrue(False)

################################################################################
#
# Test plotting functionalities of the `reduction` module
#
################################################################################

    def test_plot_2d_manifold_allowed_calls(self):

        X = np.random.rand(100,5)

        try:
            pca_X = PCA(X, scaling='auto', n_components=2)
            principal_components = pca_X.transform(X)
            plt = reduction.plot_2d_manifold(principal_components[:,0], principal_components[:,1], color=None, x_label=None, y_label=None, colorbar_label=None, title=None, save_filename=None)
            plt.close()
        except Exception:
            self.assertTrue(False)

        try:
            pca_X = PCA(X, scaling='auto', n_components=2)
            principal_components = pca_X.transform(X)
            plt = reduction.plot_2d_manifold(principal_components[:,0], principal_components[:,1], color=X[:,0], x_label=None, y_label=None, colorbar_label=None, title=None, save_filename=None)
            plt.close()
        except Exception:
            self.assertTrue(False)

        try:
            pca_X = PCA(X, scaling='auto', n_components=2)
            principal_components = pca_X.transform(X)
            plt = reduction.plot_2d_manifold(principal_components[:,0], principal_components[:,1], color=X[:,0:1], x_label=None, y_label=None, colorbar_label=None, title=None, save_filename=None)
            plt.close()
        except Exception:
            self.assertTrue(False)

        try:
            pca_X = PCA(X, scaling='auto', n_components=2)
            principal_components = pca_X.transform(X)
            plt = reduction.plot_2d_manifold(principal_components[:,0], principal_components[:,1], color='k', x_label=None, y_label=None, colorbar_label=None, title=None, save_filename=None)
            plt.close()
        except Exception:
            self.assertTrue(False)

        try:
            pca_X = PCA(X, scaling='auto', n_components=2)
            principal_components = pca_X.transform(X)
            plt = reduction.plot_2d_manifold(principal_components[:,0], principal_components[:,1], color='k', x_label='$x$', y_label='$y$', colorbar_label='$x_1$', title='Title', save_filename=None)
            plt.close()
        except Exception:
            self.assertTrue(False)

    def test_plot_2d_manifold_not_allowed_calls(self):

        X = np.random.rand(100,10)
        pca_X = PCA(X, scaling='auto', n_components=2)
        principal_components = pca_X.transform(X)
        X_rec = pca_X.reconstruct(principal_components)

        with self.assertRaises(ValueError):
            plt = reduction.plot_2d_manifold(X[:,0:2], X_rec[:,0], color=X[:,0])
            plt.close()

        with self.assertRaises(ValueError):
            plt = reduction.plot_2d_manifold(X[:,0], X_rec[:,0:2], color=X[:,0])
            plt.close()

        with self.assertRaises(ValueError):
            plt = reduction.plot_2d_manifold(X[:,0], X_rec[:,0], color=X[:,0:2])
            plt.close()

        with self.assertRaises(ValueError):
            plt = reduction.plot_2d_manifold([1,2,3], X_rec[:,0], color=X[:,0])
            plt.close()

        with self.assertRaises(ValueError):
            plt = reduction.plot_2d_manifold(X[:,0], [1,2,3], color=X[:,0])
            plt.close()

        with self.assertRaises(ValueError):
            plt = reduction.plot_2d_manifold([1,2,3], [1,2,3], color=[1,2,3])
            plt.close()

        with self.assertRaises(ValueError):
            plt = reduction.plot_2d_manifold(X[:,0], X_rec[:,0], color=[1,2,3])
            plt.close()

        with self.assertRaises(ValueError):
            plt = reduction.plot_2d_manifold(X[0:10,0], X_rec[:,0], color=X[:,0])
            plt.close()

        with self.assertRaises(ValueError):
            plt = reduction.plot_2d_manifold(X[:,0], X_rec[0:10,0], color=X[:,0])
            plt.close()

        with self.assertRaises(ValueError):
            plt = reduction.plot_2d_manifold(X[:,0], X_rec[:,0], color=X[0:10,0])
            plt.close()

    def test_plot_3d_manifold_allowed_calls(self):

        X = np.random.rand(100,5)

        try:
            plt = reduction.plot_3d_manifold(X[:,0], X[:,1], X[:,2], color=None, x_label=None, y_label=None, colorbar_label=None, title=None, save_filename=None)
            plt.close()
        except Exception:
            self.assertTrue(False)

        try:
            plt = reduction.plot_3d_manifold(X[:,0], X[:,1], X[:,2], color=X[:,0], x_label=None, y_label=None, colorbar_label=None, title=None, save_filename=None)
            plt.close()
        except Exception:
            self.assertTrue(False)

        try:
            plt = reduction.plot_3d_manifold(X[:,0], X[:,1], X[:,2], color=X[:,0:1], x_label=None, y_label=None, colorbar_label=None, title=None, save_filename=None)
            plt.close()
        except Exception:
            self.assertTrue(False)

        try:
            plt = reduction.plot_3d_manifold(X[:,0], X[:,1], X[:,2], color='k', x_label=None, y_label=None, colorbar_label=None, title=None, save_filename=None)
            plt.close()
        except Exception:
            self.assertTrue(False)

        try:
            plt = reduction.plot_3d_manifold(X[:,0], X[:,1], X[:,2], color='k', x_label='$x$', y_label='$y$', colorbar_label='$x_1$', title='Title', save_filename=None)
            plt.close()
        except Exception:
            self.assertTrue(False)

    def test_plot_3d_manifold_not_allowed_calls(self):

        X = np.random.rand(100,10)

        with self.assertRaises(ValueError):
            plt = reduction.plot_3d_manifold(X[:,0:2], X[:,0], X[:,0], color=X[:,0])
            plt.close()

        with self.assertRaises(ValueError):
            plt = reduction.plot_3d_manifold(X[:,0], X[:,0:2], X[:,0], color=X[:,0])
            plt.close()

        with self.assertRaises(ValueError):
            plt = reduction.plot_3d_manifold(X[:,0], X[:,0], X[:,0:2], color=X[:,0])
            plt.close()

        with self.assertRaises(ValueError):
            plt = reduction.plot_3d_manifold(X[:,0], X[:,0], X[:,0], color=X[:,0:2])
            plt.close()

        with self.assertRaises(ValueError):
            plt = reduction.plot_3d_manifold([1,2,3], X[:,0], X[:,0], color=X[:,0])
            plt.close()

        with self.assertRaises(ValueError):
            plt = reduction.plot_3d_manifold(X[:,0], [1,2,3], X[:,0], color=X[:,0])
            plt.close()

        with self.assertRaises(ValueError):
            plt = reduction.plot_3d_manifold(X[:,0], X[:,0], [1,2,3], color=X[:,0])
            plt.close()

        with self.assertRaises(ValueError):
            plt = reduction.plot_3d_manifold(X[:,0], X[:,0], X[:,0], color=[1,2,3])
            plt.close()

        with self.assertRaises(ValueError):
            plt = reduction.plot_3d_manifold([1,2,3], [1,2,3], [1,2,3], color=[1,2,3])
            plt.close()

        with self.assertRaises(ValueError):
            plt = reduction.plot_3d_manifold(X[0:10,0], X[:,0], X[:,0], color=X[:,0])
            plt.close()

        with self.assertRaises(ValueError):
            plt = reduction.plot_3d_manifold(X[:,0], X[0:10,0], X[:,0], color=X[:,0])
            plt.close()

        with self.assertRaises(ValueError):
            plt = reduction.plot_3d_manifold(X[:,0], X[:,0], X[0:10,0], color=X[:,0])
            plt.close()

        with self.assertRaises(ValueError):
            plt = reduction.plot_3d_manifold(X[:,0], X[:,0], X[:,0], color=X[0:10,0])
            plt.close()

    def test_plot_parity_allowed_calls(self):

        X = np.random.rand(100,5)

        try:
            pca_X = PCA(X, scaling='auto', n_components=2)
            principal_components = pca_X.transform(X)
            X_rec = pca_X.reconstruct(principal_components)
            plt = reduction.plot_parity(X[:,0], X_rec[:,0], color=None, x_label=None, y_label=None, colorbar_label=None, title=None, save_filename=None)
            plt.close()
        except Exception:
            self.assertTrue(False)

        try:
            pca_X = PCA(X, scaling='auto', n_components=2)
            principal_components = pca_X.transform(X)
            X_rec = pca_X.reconstruct(principal_components)
            plt = reduction.plot_parity(X[:,0], X_rec[:,0], color=X[:,0], x_label=None, y_label=None, colorbar_label=None, title=None, save_filename=None)
            plt.close()
        except Exception:
            self.assertTrue(False)

        try:
            pca_X = PCA(X, scaling='auto', n_components=2)
            principal_components = pca_X.transform(X)
            X_rec = pca_X.reconstruct(principal_components)
            plt = reduction.plot_parity(X[:,0], X_rec[:,0], color=X[:,0:1], x_label=None, y_label=None, colorbar_label=None, title=None, save_filename=None)
            plt.close()
        except Exception:
            self.assertTrue(False)

        try:
            pca_X = PCA(X, scaling='auto', n_components=2)
            principal_components = pca_X.transform(X)
            X_rec = pca_X.reconstruct(principal_components)
            plt = reduction.plot_parity(X[:,0], X_rec[:,0], color='k', x_label=None, y_label=None, colorbar_label=None, title=None, save_filename=None)
            plt.close()
        except Exception:
            self.assertTrue(False)

        try:
            pca_X = PCA(X, scaling='auto', n_components=2)
            principal_components = pca_X.transform(X)
            X_rec = pca_X.reconstruct(principal_components)
            plt = reduction.plot_parity(X[:,0], X_rec[:,0], color='k', x_label='$x$', y_label='$y$', colorbar_label='$x_1$', title='Title', save_filename=None)
            plt.close()
        except Exception:
            self.assertTrue(False)

    def test_plot_parity_not_allowed_calls(self):

        X = np.random.rand(100,10)
        pca_X = PCA(X, scaling='auto', n_components=2)
        principal_components = pca_X.transform(X)
        X_rec = pca_X.reconstruct(principal_components)

        with self.assertRaises(ValueError):
            plt = reduction.plot_parity(X[:,0:2], X_rec[:,0], color=X[:,0])
            plt.close()

        with self.assertRaises(ValueError):
            plt = reduction.plot_parity(X[:,0], X_rec[:,0:2], color=X[:,0])
            plt.close()

        with self.assertRaises(ValueError):
            plt = reduction.plot_parity(X[:,0], X_rec[:,0], color=X[:,0:2])
            plt.close()

        with self.assertRaises(ValueError):
            plt = reduction.plot_parity([1,2,3], X_rec[:,0], color=X[:,0])
            plt.close()

        with self.assertRaises(ValueError):
            plt = reduction.plot_parity(X[:,0], [1,2,3], color=X[:,0])
            plt.close()

        with self.assertRaises(ValueError):
            plt = reduction.plot_parity(X[:,0], X_rec[:,0], color=[1,2,3])
            plt.close()

        with self.assertRaises(ValueError):
            plt = reduction.plot_parity([1,2,3], [1,2,3], color=[1,2,3])
            plt.close()

        with self.assertRaises(ValueError):
            plt = reduction.plot_parity(X[0:10,0], X_rec[:,0], color=X[:,0])
            plt.close()

        with self.assertRaises(ValueError):
            plt = reduction.plot_parity(X[:,0], X_rec[0:10,0], color=X[:,0])
            plt.close()

        with self.assertRaises(ValueError):
            plt = reduction.plot_parity(X[:,0], X_rec[:,0], color=X[0:10,0])
            plt.close()

    def test_plot_eigenvectors_allowed_calls(self):

        X = np.random.rand(100,5)

        try:
            pca_X = PCA(X, scaling='auto', n_components=2)
            plts = reduction.plot_eigenvectors(pca_X.A, eigenvectors_indices=[], variable_names=[], plot_absolute=False, bar_color=None, title=None, save_filename=None)
            for i in range(0, len(plts)):
                plts[i].close()
        except Exception:
            self.assertTrue(False)

        try:
            pca_X = PCA(X, scaling='auto', n_components=2)
            plts = reduction.plot_eigenvectors(pca_X.A[:,0], eigenvectors_indices=[], variable_names=[], plot_absolute=False, bar_color=None, title=None, save_filename=None)
            for i in range(0, len(plts)):
                plts[i].close()
        except Exception:
            self.assertTrue(False)

        try:
            pca_X = PCA(X, scaling='auto', n_components=2)
            plts = reduction.plot_eigenvectors(pca_X.A[:,2:4], eigenvectors_indices=[], variable_names=[], plot_absolute=False, bar_color=None, title=None, save_filename=None)
            for i in range(0, len(plts)):
                plts[i].close()
        except Exception:
            self.assertTrue(False)

        try:
            pca_X = PCA(X, scaling='auto', n_components=2)
            plts = reduction.plot_eigenvectors(pca_X.A[:,0], eigenvectors_indices=[0], variable_names=['a', 'b', 'c', 'd', 'e'], plot_absolute=True, bar_color='r', title='Title', save_filename=None)
            for i in range(0, len(plts)):
                plts[i].close()
        except Exception:
            self.assertTrue(False)

    def test_plot_eigenvectors_comparison_allowed_calls(self):

        X = np.random.rand(100,5)

        try:
            pca_1 = PCA(X, scaling='auto', n_components=2)
            pca_2 = PCA(X, scaling='range', n_components=2)
            pca_3 = PCA(X, scaling='vast', n_components=2)
            plt = reduction.plot_eigenvectors_comparison((pca_1.A[:,0], pca_2.A[:,0], pca_3.A[:,0]), legend_labels=[], variable_names=[], plot_absolute=False, color_map='coolwarm', title=None, save_filename=None)
            plt.close()
        except Exception:
            self.assertTrue(False)

        try:
            pca_1 = PCA(X, scaling='auto', n_components=2)
            pca_2 = PCA(X, scaling='range', n_components=2)
            pca_3 = PCA(X, scaling='vast', n_components=2)
            plt = reduction.plot_eigenvectors_comparison((pca_1.A[:,0], pca_2.A[:,0], pca_3.A[:,0]), legend_labels=['$a$', '$b$', '$c$'], variable_names=['a', 'b', 'c', 'd', 'e'], plot_absolute=True, color_map='viridis', title='Title', save_filename=None)
            plt.close()
        except Exception:
            self.assertTrue(False)

    def test_plot_eigenvalue_distribution_allowed_calls(self):

        X = np.random.rand(100,5)

        try:
            pca_X = PCA(X, scaling='auto', n_components=2)
            plt = reduction.plot_eigenvalue_distribution(pca_X.L, normalized=False, title=None, save_filename=None)
            plt.close()
        except Exception:
            self.assertTrue(False)

        try:
            pca_X = PCA(X, scaling='auto', n_components=2)
            plt = reduction.plot_eigenvalue_distribution(pca_X.L, normalized=True, title=None, save_filename=None)
            plt.close()
        except Exception:
            self.assertTrue(False)

        try:
            pca_X = PCA(X, scaling='auto', n_components=2)
            plt = reduction.plot_eigenvalue_distribution(pca_X.L, normalized=True, title='Title', save_filename=None)
            plt.close()
        except Exception:
            self.assertTrue(False)

    def test_plot_eigenvalue_distribution_comparison_allowed_calls(self):

        X = np.random.rand(100,5)

        try:
            pca_1 = PCA(X, scaling='auto', n_components=2)
            pca_2 = PCA(X, scaling='range', n_components=2)
            pca_3 = PCA(X, scaling='vast', n_components=2)
            plt = reduction.plot_eigenvalue_distribution_comparison((pca_1.L, pca_2.L, pca_3.L), legend_labels=[], normalized=False, color_map='coolwarm', title=None, save_filename=None)
            plt.close()
        except Exception:
            self.assertTrue(False)

        try:
            pca_1 = PCA(X, scaling='auto', n_components=2)
            pca_2 = PCA(X, scaling='range', n_components=2)
            pca_3 = PCA(X, scaling='vast', n_components=2)
            plt = reduction.plot_eigenvalue_distribution_comparison((pca_1.L, pca_2.L, pca_3.L), legend_labels=['Auto', 'Range', 'Vast'], normalized=True, color_map='viridis', title='Title', save_filename=None)
            plt.close()
        except Exception:
            self.assertTrue(False)

    def test_plot_cumulative_variance_allowed_calls(self):

        X = np.random.rand(100,5)

        try:
            pca_X = PCA(X, scaling='auto', n_components=2)
            plt = reduction.plot_cumulative_variance(pca_X.L, n_components=0, title=None, save_filename=None)
            plt.close()
        except Exception:
            self.assertTrue(False)

        try:
            pca_X = PCA(X, scaling='auto', n_components=2)
            plt = reduction.plot_cumulative_variance(pca_X.L, n_components=2, title=None, save_filename=None)
            plt.close()
        except Exception:
            self.assertTrue(False)

        try:
            pca_X = PCA(X, scaling='auto', n_components=2)
            plt = reduction.plot_cumulative_variance(pca_X.L, n_components=3, title='Title', save_filename=None)
            plt.close()
        except Exception:
            self.assertTrue(False)

    def test_calculate_r2_on_all_avaiable_scalings(self):

        test_data_set = np.random.rand(1000,20)
        r2_test = np.ones((20,))

        try:
            pca_X = PCA(test_data_set, scaling='none', n_components=20, use_eigendec=True, nocenter=False)
            r2_values = pca_X.calculate_r2(test_data_set)
            self.assertTrue(np.allclose(r2_values, r2_test))
        except Exception:
            self.assertTrue(False)

        try:
            pca_X = PCA(test_data_set, scaling='auto', n_components=20, use_eigendec=True, nocenter=False)
            r2_values = pca_X.calculate_r2(test_data_set)
            self.assertTrue(np.allclose(r2_values, r2_test))
        except Exception:
            self.assertTrue(False)

        try:
            pca_X = PCA(test_data_set, scaling='pareto', n_components=20, use_eigendec=True, nocenter=False)
            r2_values = pca_X.calculate_r2(test_data_set)
            self.assertTrue(np.allclose(r2_values, r2_test))
        except Exception:
            self.assertTrue(False)

        try:
            pca_X = PCA(test_data_set, scaling='vast', n_components=20, use_eigendec=True, nocenter=False)
            r2_values = pca_X.calculate_r2(test_data_set)
            self.assertTrue(np.allclose(r2_values, r2_test))
        except Exception:
            self.assertTrue(False)

        try:
            pca_X = PCA(test_data_set, scaling='range', n_components=20, use_eigendec=True, nocenter=False)
            r2_values = pca_X.calculate_r2(test_data_set)
            self.assertTrue(np.allclose(r2_values, r2_test))
        except Exception:
            self.assertTrue(False)

        try:
            pca_X = PCA(test_data_set, scaling='-1to1', n_components=20, use_eigendec=True, nocenter=False)
            r2_values = pca_X.calculate_r2(test_data_set)
            self.assertTrue(np.allclose(r2_values, r2_test))
        except Exception:
            self.assertTrue(False)

        try:
            pca_X = PCA(test_data_set, scaling='level', n_components=20, use_eigendec=True, nocenter=False)
            r2_values = pca_X.calculate_r2(test_data_set)
            self.assertTrue(np.allclose(r2_values, r2_test))
        except Exception:
            self.assertTrue(False)

        try:
            pca_X = PCA(test_data_set, scaling='max', n_components=20, use_eigendec=True, nocenter=False)
            r2_values = pca_X.calculate_r2(test_data_set)
            self.assertTrue(np.allclose(r2_values, r2_test))
        except Exception:
            self.assertTrue(False)

        try:
            pca_X = PCA(test_data_set, scaling='poisson', n_components=20, use_eigendec=True, nocenter=False)
            r2_values = pca_X.calculate_r2(test_data_set)
            self.assertTrue(np.allclose(r2_values, r2_test))
        except Exception:
            self.assertTrue(False)

        try:
            pca_X = PCA(test_data_set, scaling='vast_2', n_components=20, use_eigendec=True, nocenter=False)
            r2_values = pca_X.calculate_r2(test_data_set)
            self.assertTrue(np.allclose(r2_values, r2_test))
        except Exception:
            self.assertTrue(False)

        try:
            pca_X = PCA(test_data_set, scaling='vast_3', n_components=20, use_eigendec=True, nocenter=False)
            r2_values = pca_X.calculate_r2(test_data_set)
            self.assertTrue(np.allclose(r2_values, r2_test))
        except Exception:
            self.assertTrue(False)

        try:
            pca_X = PCA(test_data_set, scaling='vast_4', n_components=20, use_eigendec=True, nocenter=False)
            r2_values = pca_X.calculate_r2(test_data_set)
            self.assertTrue(np.allclose(r2_values, r2_test))
        except Exception:
            self.assertTrue(False)

    def test_calculate_r2_on_all_avaiable_scalings_with_no_centering(self):

        test_data_set = np.random.rand(1000,20)
        r2_test = np.ones((20,))

        try:
            pca_X = PCA(test_data_set, scaling='none', n_components=20, use_eigendec=True, nocenter=True)
            r2_values = pca_X.calculate_r2(test_data_set)
            self.assertTrue(np.allclose(r2_values, r2_test))
        except Exception:
            self.assertTrue(False)

        try:
            pca_X = PCA(test_data_set, scaling='auto', n_components=20, use_eigendec=True, nocenter=True)
            r2_values = pca_X.calculate_r2(test_data_set)
            self.assertTrue(np.allclose(r2_values, r2_test))
        except Exception:
            self.assertTrue(False)

        try:
            pca_X = PCA(test_data_set, scaling='pareto', n_components=20, use_eigendec=True, nocenter=True)
            r2_values = pca_X.calculate_r2(test_data_set)
            self.assertTrue(np.allclose(r2_values, r2_test))
        except Exception:
            self.assertTrue(False)

        try:
            pca_X = PCA(test_data_set, scaling='vast', n_components=20, use_eigendec=True, nocenter=True)
            r2_values = pca_X.calculate_r2(test_data_set)
            self.assertTrue(np.allclose(r2_values, r2_test))
        except Exception:
            self.assertTrue(False)

        try:
            pca_X = PCA(test_data_set, scaling='range', n_components=20, use_eigendec=True, nocenter=True)
            r2_values = pca_X.calculate_r2(test_data_set)
            self.assertTrue(np.allclose(r2_values, r2_test))
        except Exception:
            self.assertTrue(False)

        try:
            pca_X = PCA(test_data_set, scaling='-1to1', n_components=20, use_eigendec=True, nocenter=True)
            r2_values = pca_X.calculate_r2(test_data_set)
            self.assertTrue(np.allclose(r2_values, r2_test))
        except Exception:
            self.assertTrue(False)

        try:
            pca_X = PCA(test_data_set, scaling='level', n_components=20, use_eigendec=True, nocenter=True)
            r2_values = pca_X.calculate_r2(test_data_set)
            self.assertTrue(np.allclose(r2_values, r2_test))
        except Exception:
            self.assertTrue(False)

        try:
            pca_X = PCA(test_data_set, scaling='max', n_components=20, use_eigendec=True, nocenter=True)
            r2_values = pca_X.calculate_r2(test_data_set)
            self.assertTrue(np.allclose(r2_values, r2_test))
        except Exception:
            self.assertTrue(False)

        try:
            pca_X = PCA(test_data_set, scaling='poisson', n_components=20, use_eigendec=True, nocenter=True)
            r2_values = pca_X.calculate_r2(test_data_set)
            self.assertTrue(np.allclose(r2_values, r2_test))
        except Exception:
            self.assertTrue(False)

        try:
            pca_X = PCA(test_data_set, scaling='vast_2', n_components=20, use_eigendec=True, nocenter=True)
            r2_values = pca_X.calculate_r2(test_data_set)
            self.assertTrue(np.allclose(r2_values, r2_test))
        except Exception:
            self.assertTrue(False)

        try:
            pca_X = PCA(test_data_set, scaling='vast_3', n_components=20, use_eigendec=True, nocenter=True)
            r2_values = pca_X.calculate_r2(test_data_set)
            self.assertTrue(np.allclose(r2_values, r2_test))
        except Exception:
            self.assertTrue(False)

        try:
            pca_X = PCA(test_data_set, scaling='vast_4', n_components=20, use_eigendec=True, nocenter=True)
            r2_values = pca_X.calculate_r2(test_data_set)
            self.assertTrue(np.allclose(r2_values, r2_test))
        except Exception:
            self.assertTrue(False)
