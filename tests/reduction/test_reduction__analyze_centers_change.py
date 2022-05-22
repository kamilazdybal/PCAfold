import unittest
import numpy as np
from PCAfold import preprocess
from PCAfold import reduction
from PCAfold import analysis

class Reduction(unittest.TestCase):

    def test_reduction__analyze_centers_change__allowed_calls(self):

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


# ------------------------------------------------------------------------------
