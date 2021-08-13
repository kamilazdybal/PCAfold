import unittest
import numpy as np
from PCAfold import preprocess
from PCAfold import reduction
from PCAfold import analysis

class Analysis(unittest.TestCase):

    def test_analysis__plot_normalized_variance_derivative_comparison__allowed_calls(self):

        X = np.random.rand(100,5)
        Y = np.random.rand(100,5)
        Z = np.random.rand(100,5)
        pca_X = reduction.PCA(X, n_components=2)
        pca_Y = reduction.PCA(Y, n_components=2)
        pca_Z = reduction.PCA(Y, n_components=2)
        principal_components_X = pca_X.transform(X)
        principal_components_Y = pca_Y.transform(Y)
        principal_components_Z = pca_Z.transform(Z)
        variance_data_X = analysis.compute_normalized_variance(principal_components_X, X, depvar_names=['A', 'B', 'C', 'D', 'E'], bandwidth_values=np.logspace(-3, 2, 20), scale_unit_box=True)
        variance_data_Y = analysis.compute_normalized_variance(principal_components_Y, Y, depvar_names=['F', 'G', 'H', 'I', 'J'], bandwidth_values=np.logspace(-3, 2, 20), scale_unit_box=True)
        variance_data_Z = analysis.compute_normalized_variance(principal_components_Z, Z, depvar_names=['K', 'L', 'M', 'N', 'O'], bandwidth_values=np.logspace(-3, 2, 20), scale_unit_box=True)

        try:
            plt = analysis.plot_normalized_variance_derivative_comparison((variance_data_X, variance_data_Y), ([0,1,2], [0,1,2]), ('Blues', 'Reds'), title=None, save_filename=None)
            plt.close()
        except Exception:
            self.assertTrue(False)

        try:
            plt = analysis.plot_normalized_variance_derivative_comparison((variance_data_X, variance_data_Y), ([0,1,2], [0,1,2]), ('Blues', 'Reds'), title='Normalized variance comparison', save_filename=None)
            plt.close()
        except Exception:
            self.assertTrue(False)

        try:
            plt = analysis.plot_normalized_variance_derivative_comparison((variance_data_X, variance_data_Y, variance_data_Z), ([0,1,2], [0,1,2], []), ('Greys', 'Blues', 'Reds'), title='Normalized variance comparison', save_filename=None)
            plt.close()
        except Exception:
            self.assertTrue(False)

        try:
            plt = analysis.plot_normalized_variance_derivative_comparison((variance_data_X, variance_data_Y, variance_data_Z), ([0], [2,3], []), ('Greys', 'Blues', 'Reds'), title=None, save_filename=None)
            plt.close()
        except Exception:
            self.assertTrue(False)

# ------------------------------------------------------------------------------

    def test_analysis__plot_normalized_variance_derivative_comparison__not_allowed_calls(self):

        pass

# ------------------------------------------------------------------------------
