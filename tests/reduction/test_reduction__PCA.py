import unittest
import numpy as np
from PCAfold import preprocess
from PCAfold import reduction
from scipy import linalg as lg

class TestReduction(unittest.TestCase):

    def test_reduction__PCA__PCA_with_eigendec_vs_SVD(self):

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
        pca = reduction.PCA(PHI + xbar, 'NONE', use_eigendec=False)
        pca2 = reduction.PCA(PHI + xbar, 'NONE', use_eigendec=True)

        # comparing mean(centering), centered data, Q, and L

        if np.any(abs(xbar - pca.X_center) > tol):
            self.assertTrue(False)

        if np.any(abs(PHI - pca.X_cs) > tol):
            self.assertTrue(False)

        if np.any(abs(Q) - abs(pca.A) > tol):
            self.assertTrue(False)

        if np.any(abs(L - pca.L) > tol):
            self.assertTrue(False)

        # Check if feed eta's to PCA, return same eta's when do transform
        eta = pca.transform(PHI + xbar)  # dataset as example of eta's

        # both methods of PCA:
        pca = reduction.PCA(eta, 'NONE', use_eigendec=False)
        pca2 = reduction.PCA(eta, 'NONE', use_eigendec=True)

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
        if np.any(abs(eta - eta_new) > tol):
            self.assertTrue(False)

# ------------------------------------------------------------------------------

# Test if some PCA class attributes cannot be set by the user after `PCA` object has been created:
    def test_reduction__PCA__not_allowed_attribute_setting(self):

        X = np.random.rand(100,20)
        pca = reduction.PCA(X, scaling='auto', n_components=10)

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
            pca.tq = 1
        with self.assertRaises(AttributeError):
            pca.tqj = 1
        with self.assertRaises(AttributeError):
            pca.scaling = 1
        with self.assertRaises(AttributeError):
            pca.n_variables = 1
        with self.assertRaises(AttributeError):
            pca.n_components_init = 1

# ------------------------------------------------------------------------------

# Test if all available PCA class attributes can be accessed without error:
    def test_reduction__PCA__getting_attributes(self):

        X = np.random.rand(100,20)
        pca = reduction.PCA(X, scaling='auto', n_components=10)

        try:
            pca.X_cs
            pca.X_center
            pca.X_scale
            pca.S
            pca.A
            pca.L
            pca.loadings
            pca.tq
            pca.tqj
            pca.scaling
            pca.n_variables
            pca.n_components_init
            pca.n_components
        except Exception:
            self.assertTrue(False)

# ------------------------------------------------------------------------------

# Test n_components PCA class attribute - the only attribute that is allowed to be set
    def test_reduction__PCA__n_components_attribute(self):

        X = np.random.rand(100,20)

        try:
            pca = reduction.PCA(X, scaling='auto', n_components=2)
        except Exception:
            self.assertTrue(False)

        try:
            pca.n_components
        except Exception:
            self.assertTrue(False)

        try:
            pca = reduction.PCA(X, scaling='auto', n_components=2)
            pca.n_components = 0
            current_n = pca.n_components
            self.assertTrue(current_n == 20)
        except Exception:
            self.assertTrue(False)

        try:
            pca = reduction.PCA(X, scaling='auto', n_components=2)
            pca.n_components = 10
            current_n = pca.n_components
            self.assertTrue(current_n == 10)
        except Exception:
            self.assertTrue(False)

        try:
            pca = reduction.PCA(X, scaling='auto', n_components=2)
            pca.n_components = 10
            current_n = pca.n_components
            self.assertTrue(current_n == 10)
            pca.n_components = pca.n_components_init
            current_n = pca.n_components
            self.assertTrue(current_n == 2)
        except Exception:
            self.assertTrue(False)

# ------------------------------------------------------------------------------

    def test_reduction__PCA__n_components_attribute_not_allowed(self):

        X = np.random.rand(100,20)

        with self.assertRaises(ValueError):
            pca = reduction.PCA(X, scaling='auto', n_components=-1)
        with self.assertRaises(ValueError):
            pca = reduction.PCA(X, scaling='auto', n_components=1.5)
        with self.assertRaises(ValueError):
            pca = reduction.PCA(X, scaling='auto', n_components=True)
        with self.assertRaises(ValueError):
            pca = reduction.PCA(X, scaling='auto', n_components='PC')

        try:
            pca = reduction.PCA(X, scaling='auto', n_components=10)
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

# ------------------------------------------------------------------------------

    def test_reduction__PCA__allowed_initializations(self):

        test_data_set = np.random.rand(100,20)
        test_data_set_constant = np.random.rand(100,20)
        test_data_set_constant[:,10] = np.ones((100,))
        test_data_set_constant[:,5] = np.ones((100,))

        try:
            pca = reduction.PCA(test_data_set, scaling='auto')
        except Exception:
            self.assertTrue(False)

        try:
            pca = reduction.PCA(test_data_set, scaling='auto')
        except Exception:
            self.assertTrue(False)

        try:
            pca = reduction.PCA(test_data_set, scaling='std')
        except Exception:
            self.assertTrue(False)

        try:
            pca = reduction.PCA(test_data_set, scaling='none')
        except Exception:
            self.assertTrue(False)

        try:
            pca = reduction.PCA(test_data_set, scaling='auto', n_components=2)
        except Exception:
            self.assertTrue(False)

        try:
            pca = reduction.PCA(test_data_set, scaling='auto', n_components=3, nocenter=True)
        except Exception:
            self.assertTrue(False)

        try:
            pca = reduction.PCA(test_data_set, scaling='pareto', n_components=2, nocenter=True)
        except Exception:
            self.assertTrue(False)

        try:
            pca = reduction.PCA(test_data_set, scaling='auto', n_components=2, use_eigendec=False)
        except Exception:
            self.assertTrue(False)

        try:
            pca = reduction.PCA(test_data_set, scaling='range', n_components=2, use_eigendec=False, nocenter=True)
        except Exception:
            self.assertTrue(False)

        try:
            (X_removed, idx_removed, idx_retained) = preprocess.remove_constant_vars(test_data_set_constant)
        except Exception:
            self.assertTrue(False)

        try:
            pca = reduction.PCA(X_removed, scaling='range', n_components=2)
        except Exception:
            self.assertTrue(False)

# ------------------------------------------------------------------------------

    def test_reduction__PCA__not_allowed_initializations(self):

        test_data_set = np.random.rand(100,20)
        test_data_set_constant = np.random.rand(100,20)
        test_data_set_constant[:,10] = np.ones((100,))
        test_data_set_constant[:,5] = np.ones((100,))

        with self.assertRaises(ValueError):
            pca = reduction.PCA(test_data_set, scaling='none', n_components=-1)

        with self.assertRaises(ValueError):
            pca = reduction.PCA(test_data_set, scaling='auto', n_components=30)

        with self.assertRaises(ValueError):
            pca = reduction.PCA(test_data_set, scaling='auto', n_components=3, use_eigendec=1)

        with self.assertRaises(ValueError):
            pca = reduction.PCA(test_data_set, scaling='auto', nocenter=1)

        with self.assertRaises(ValueError):
            pca = reduction.PCA(test_data_set, scaling=False)

        with self.assertRaises(ValueError):
            pca = reduction.PCA(test_data_set, scaling='none', n_components=True)

        with self.assertRaises(ValueError):
            pca = reduction.PCA(test_data_set, scaling='none', n_components=5, nocenter='False')

        with self.assertRaises(ValueError):
            pca = reduction.PCA(test_data_set, scaling='auto', n_components=3, use_eigendec='True')

        with self.assertRaises(ValueError):
            pca = reduction.PCA(test_data_set_constant, scaling='auto', n_components=2)

        with self.assertRaises(ValueError):
            pca = reduction.PCA(test_data_set_constant)

        with self.assertRaises(ValueError):
            pca = reduction.PCA(test_data_set_constant, scaling='range', n_components=5)

# ------------------------------------------------------------------------------

    def test_reduction__PCA__simulate_chemical_source_term_handling(self):

        X = np.random.rand(200,10)
        X_source = np.random.rand(200,10)

        pca = reduction.PCA(X, scaling='auto')

        try:
            PC_source = pca.transform(X_source, nocenter=True)
            PC_source_rec = pca.reconstruct(PC_source, nocenter=True)

            difference = abs(X_source - PC_source_rec)
            comparison = difference < 10**(-14)
            self.assertTrue(comparison.all())
        except Exception:
            self.assertTrue(False)

# ------------------------------------------------------------------------------

    def test_reduction__PCA__tq_tqj_loadings_computation(self):

        X = np.random.rand(1000,20)

        try:
            pca_X = reduction.PCA(X, scaling='auto', n_components=10)
            tq = pca_X.tq
            tqj = pca_X.tqj
            loadings = pca_X.loadings
            self.assertTrue(np.array_equal(tq, np.sum(tqj, axis=1)))
            difference = abs(tq - np.sum(tqj, axis=1))
            comparison = difference < 10**(-14)
            comparison.all()
        except Exception:
            self.assertTrue(False)

        try:
            pca_X = reduction.PCA(X, scaling='auto')
            tq = pca_X.tq
            tqj = pca_X.tqj
            loadings = pca_X.loadings
            difference = abs(np.ones_like(tq) - tq)
            comparison = difference < 10**(-14)
            comparison.all()
            difference = abs(np.ones_like(tq) - np.sum(loadings**2, axis=1))
            comparison = difference < 10**(-14)
            comparison.all()
        except Exception:
            self.assertTrue(False)

# ------------------------------------------------------------------------------
