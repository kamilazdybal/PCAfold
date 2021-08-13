import unittest
import numpy as np
from PCAfold import preprocess
from PCAfold import reduction
from PCAfold import analysis

class Analysis(unittest.TestCase):

    def test_analysis__plot_normalized_variance__allowed_calls(self):

        X = np.random.rand(100,5)
        pca_X = reduction.PCA(X, n_components=2)
        principal_components = pca_X.transform(X)
        variance_data = analysis.compute_normalized_variance(principal_components, X, depvar_names=['A', 'B', 'C', 'D', 'E'], bandwidth_values=np.logspace(-3, 1, 20), scale_unit_box=True)

        try:
            plt = analysis.plot_normalized_variance(variance_data, plot_variables=[0,1,2], color_map='Blues', figure_size=(10,5), title=None, save_filename=None)
            plt.close()
        except Exception:
            self.assertTrue(False)

        try:
            plt = analysis.plot_normalized_variance(variance_data, plot_variables=[], color_map='Blues', figure_size=(10,5), title='Normalized variance', save_filename=None)
            plt.close()
        except Exception:
            self.assertTrue(False)

        try:
            plt = analysis.plot_normalized_variance(variance_data, plot_variables=[2,3,4], color_map='Blues', figure_size=(15,5), title='Normalized variance', save_filename=None)
            plt.close()
        except Exception:
            self.assertTrue(False)

        try:
            plt = analysis.plot_normalized_variance(variance_data, plot_variables=[], color_map='Reds', figure_size=(10,5), title=None, save_filename=None)
            plt.close()
        except Exception:
            self.assertTrue(False)

# ------------------------------------------------------------------------------

    def test_analysis__plot_normalized_variance__not_allowed_calls(self):

        pass

# ------------------------------------------------------------------------------
