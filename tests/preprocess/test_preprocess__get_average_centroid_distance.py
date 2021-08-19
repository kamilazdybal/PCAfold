import unittest
import numpy as np
from PCAfold import preprocess
from PCAfold import reduction
from PCAfold import analysis

class Preprocess(unittest.TestCase):

    def test_preprocess__get_average_centroid_distance__allowed_calls(self):

        pass

# ------------------------------------------------------------------------------

    def test_preprocess__get_average_centroid_distance__not_allowed_calls(self):

        pass

# ------------------------------------------------------------------------------

    def test_preprocess__get_average_centroid_distance__computation(self):

        phi_1 = np.linspace(0,2*np.pi,100000)
        r_1 = 2
        x_1 = r_1 * np.cos(phi_1)
        y_1 = r_1 * np.sin(phi_1)

        phi_2 = np.linspace(0,2*np.pi,100)
        r_2 = 1
        x_2 = r_2 * np.cos(phi_2) - 5
        y_2 = r_2 * np.sin(phi_2) - 5

        X = np.hstack((np.vstack((x_1[:,None], x_2[:,None])), np.vstack((y_1[:,None], y_2[:,None]))))

        (idx, _) = preprocess.predefined_variable_bins(X[:,0], [-3], verbose=False)

        try:
            average_centroid_distance = preprocess.get_average_centroid_distance(X, idx, weighted=True)
            self.assertTrue(average_centroid_distance > 1.99)
            self.assertTrue(average_centroid_distance < 2.01)
        except Exception:
            self.assertTrue(False)

        try:
            average_centroid_distance = preprocess.get_average_centroid_distance(X, idx, weighted=False)
            self.assertTrue(average_centroid_distance > 1.49)
            self.assertTrue(average_centroid_distance < 1.51)
        except Exception:
            self.assertTrue(False)

# ------------------------------------------------------------------------------
