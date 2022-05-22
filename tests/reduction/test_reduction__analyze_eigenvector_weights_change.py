import unittest
import numpy as np
from PCAfold import preprocess
from PCAfold import reduction
from PCAfold import analysis

class Reduction(unittest.TestCase):

    def test_reduction__analyze_eigenvector_weights_change__allowed_calls(self):

        X = np.random.rand(200,20)
        idx = np.zeros((200,))
        idx[20:60,] = 1
        idx[150:190] = 2

        idx = idx.astype(int)

        try:

            equilibrate_pca = reduction.EquilibratedSamplePCA(X, idx, 'auto', 20, 1, X_source=None, n_iterations=20, stop_iter=0, random_seed=None, verbose=False)

            eigenvalues = equilibrate_pca.eigenvalues
            eigenvectors_matrix = equilibrate_pca.eigenvectors
            pc_scores_matrix = equilibrate_pca.pc_scores
            pc_sources_matrix = equilibrate_pca.pc_sources
            idx_train = equilibrate_pca.idx_train
            C_r = equilibrate_pca.C_r
            D_r = equilibrate_pca.D_r

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

            equilibrate_pca = reduction.EquilibratedSamplePCA(X, idx, 'auto', 20, 1, X_source=None, n_iterations=2, stop_iter=0, random_seed=None, verbose=False)

            eigenvalues = equilibrate_pca.eigenvalues
            eigenvectors_matrix = equilibrate_pca.eigenvectors
            pc_scores_matrix = equilibrate_pca.pc_scores
            pc_sources_matrix = equilibrate_pca.pc_sources
            idx_train = equilibrate_pca.idx_train
            C_r = equilibrate_pca.C_r
            D_r = equilibrate_pca.D_r

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

# ------------------------------------------------------------------------------
