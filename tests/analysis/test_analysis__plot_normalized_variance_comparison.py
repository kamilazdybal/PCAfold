import unittest
import numpy as np
from PCAfold import preprocess
from PCAfold import reduction
from PCAfold import analysis

class Analysis(unittest.TestCase):

    def test_analysis__plot_normalized_variance_comparison__allowed_calls(self):

        X = np.random.rand(100,5)
        Y = np.random.rand(100,5)

        pca_X = reduction.PCA(X, n_components=2)
        pca_Y = reduction.PCA(Y, n_components=2)
        principal_components_X = pca_X.transform(X)
        principal_components_Y = pca_Y.transform(Y)

        variance_data_X = analysis.compute_normalized_variance(principal_components_X, X, depvar_names=['A', 'B', 'C', 'D', 'E'], bandwidth_values=np.logspace(-3, 2, 20), scale_unit_box=True)
        variance_data_Y = analysis.compute_normalized_variance(principal_components_Y, Y, depvar_names=['F', 'G', 'H', 'I', 'J'], bandwidth_values=np.logspace(-3, 2, 20), scale_unit_box=True)

        # try:
        #     plt = analysis.plot_normalized_variance_comparison((variance_data_X, variance_data_Y),
        #                                               ([0,1,2], [0,1,2]),
        #                                               ('Blues', 'Reds'))
        #     plt.close()
        # except:
        #     self.assertTrue(False)
        #
        # try:
        #     plt = analysis.plot_normalized_variance_comparison((variance_data_X, variance_data_Y),
        #                                               ([0,1,2], [0,1,2]),
        #                                               ('Blues', 'Reds'),
        #                                               figure_size=(10,5),
        #                                               title='Normalized variance comparison',
        #                                               save_filename=None)
        #     plt.close()
        # except:
        #     self.assertTrue(False)

# ------------------------------------------------------------------------------

    def test_analysis__plot_normalized_variance_comparison__not_allowed_calls(self):

        X = np.random.rand(100,5)
        Y = np.random.rand(100,5)

        pca_X = reduction.PCA(X, n_components=2)
        pca_Y = reduction.PCA(Y, n_components=2)
        principal_components_X = pca_X.transform(X)
        principal_components_Y = pca_Y.transform(Y)

        variance_data_X = analysis.compute_normalized_variance(principal_components_X, X, depvar_names=['A', 'B', 'C', 'D', 'E'], bandwidth_values=np.logspace(-3, 2, 20), scale_unit_box=True)
        variance_data_Y = analysis.compute_normalized_variance(principal_components_Y, Y, depvar_names=['F', 'G', 'H', 'I', 'J'], bandwidth_values=np.logspace(-3, 2, 20), scale_unit_box=True)

        with self.assertRaises(ValueError):
            plt = analysis.plot_normalized_variance_comparison((variance_data_X, variance_data_Y), ([0,1,2], [0,1,2]), ('Blues', 'Reds'), figure_size=[1])
            plt.close()

        with self.assertRaises(ValueError):
            plt = analysis.plot_normalized_variance_comparison((variance_data_X, variance_data_Y), ([0,1,2], [0,1,2]), ('Blues', 'Reds'), title=[1])
            plt.close()

        with self.assertRaises(ValueError):
            plt = analysis.plot_normalized_variance_comparison((variance_data_X, variance_data_Y), ([0,1,2], [0,1,2]), ('Blues', 'Reds'), save_filename=[1])
            plt.close()

# ------------------------------------------------------------------------------
