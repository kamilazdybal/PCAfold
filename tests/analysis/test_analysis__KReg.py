import unittest
import numpy as np
from PCAfold import preprocess
from PCAfold import reduction
from PCAfold import analysis

class Analysis(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(Analysis, self).__init__(*args, **kwargs)
        self._indepvars = np.array([[1., 2.], [3., 5.], [6., 9.]])
        self._depvars = np.array([[2.], [4.], [1.]])
        self._krmod = analysis.KReg(self._indepvars, self._depvars)
        self._query = np.array([[2., 3.]])

# ------------------------------------------------------------------------------

    def test_analysis__KReg__allowed_class_init(self):

        pass

# ------------------------------------------------------------------------------

    def test_analysis__KReg__not_allowed_class_init(self):

        pass

# ------------------------------------------------------------------------------

    def test_constant_bandwidth(self):
        self.assertTrue(np.all(self._krmod.compute_constant_bandwidth(self._query, 1.5) == np.array([[1.5, 1.5]])))

# ------------------------------------------------------------------------------

    def test_isotropic_bandwidth(self):
        self.assertTrue(np.all(self._krmod.compute_bandwidth_isotropic(self._query, np.array([1.5])) == np.array([[1.5, 1.5]])))

# ------------------------------------------------------------------------------

    def test_anisotropic_bandwidth(self):
        self.assertTrue(
            np.all(self._krmod.compute_bandwidth_anisotropic(self._query, np.array([1.5, 2.0])) == np.array([[1.5, 2.0]])))

# ------------------------------------------------------------------------------

    def test_nearest_neighbors_isotropic_bandwidth(self):
        smallest_distance = np.linalg.norm(self._query - self._indepvars[0, :])
        self.assertTrue(np.all(self._krmod.compute_nearest_neighbors_bandwidth_isotropic(self._query, 1) == np.array(
            [[smallest_distance, smallest_distance]])))

# ------------------------------------------------------------------------------

    def test_nearest_neighbors_anisotropic_bandwidth(self):
        self.assertTrue(np.all(self._krmod.compute_nearest_neighbors_bandwidth_anisotropic(self._query, 2) == np.array([[1., 2.]])))

# ------------------------------------------------------------------------------

    def test_predict_return_depvars(self):
        self.assertTrue(self._krmod.predict(np.array([self._indepvars[0, :]]), 1.e-6) == self._depvars[0])

# ------------------------------------------------------------------------------

    def test_predict(self):
        bw = np.array([[1.5, 2.]])  # test bandwidth
        weight1 = np.exp(np.sum(-(self._query - self._indepvars[0, :]) ** 2 / bw ** 2))
        weight2 = np.exp(np.sum(-(self._query - self._indepvars[1, :]) ** 2 / bw ** 2))
        weight3 = np.exp(np.sum(-(self._query - self._indepvars[2, :]) ** 2 / bw ** 2))
        weightsum = weight1 + weight2 + weight3
        ans = (weight1 * self._depvars[0] + weight2 * self._depvars[1] + weight3 * self._depvars[2]) / weightsum
        self.assertTrue(self._krmod.predict(self._query, bw) == ans)

# ------------------------------------------------------------------------------
