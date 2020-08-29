import unittest
import numpy as np
from PCAfold import preprocess
from PCAfold import reduction
from PCAfold import PCA
from PCAfold import DataSampler
from scipy import linalg as lg

class TestReduction(unittest.TestCase):

    def test_PCA(self):

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

        if np.any(Q - pca.Q > tol) or np.any(Q - pca2.Q > tol):
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

    def test_transform_not_allowed_calls(self):

        test_data_set = np.random.rand(10,2)
        test_data_set_2 = np.random.rand(10,3)

        pca = PCA(test_data_set, scaling='auto')

        with self.assertRaises(ValueError):
            pca.transform(test_data_set_2)

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

    def test_analyze_eigenvector_weights_change_allowed_calls(self):

        X = np.random.rand(200,20)
        idx = np.zeros((200,))
        idx[20:60,] = 1
        idx[150:190] = 2

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

    def test_plot_2d_manifold_allowed_calls(self):

        X = np.random.rand(100,5)

        try:
            pca_X = PCA(X, scaling='auto', n_components=2)
            principal_components = pca_X.transform(X)
            plt = reduction.plot_2d_manifold(principal_components, color_variable=[], x_label=None, y_label=None, colorbar_label=None, title=None, save_filename=None)
            plt.close()
        except Exception:
            self.assertTrue(False)

        try:
            pca_X = PCA(X, scaling='auto', n_components=2)
            principal_components = pca_X.transform(X)
            plt = reduction.plot_2d_manifold(principal_components, color_variable=X[:,0], x_label=None, y_label=None, colorbar_label=None, title=None, save_filename=None)
            plt.close()
        except Exception:
            self.assertTrue(False)

        try:
            pca_X = PCA(X, scaling='auto', n_components=2)
            principal_components = pca_X.transform(X)
            plt = reduction.plot_2d_manifold(principal_components, color_variable='k', x_label=None, y_label=None, colorbar_label=None, title=None, save_filename=None)
            plt.close()
        except Exception:
            self.assertTrue(False)

        try:
            pca_X = PCA(X, scaling='auto', n_components=2)
            principal_components = pca_X.transform(X)
            plt = reduction.plot_2d_manifold(principal_components, color_variable='k', x_label='$x$', y_label='$y$', colorbar_label='$x_1$', title='Title', save_filename=None)
            plt.close()
        except Exception:
            self.assertTrue(False)

    def test_plot_eigenvectors_allowed_calls(self):

        X = np.random.rand(100,5)

        try:
            pca_X = PCA(X, scaling='auto', n_components=2)
            plts = reduction.plot_eigenvectors(pca_X.Q, eigenvectors_indices=[], variable_names=[], plot_absolute=False, bar_color=None, title=None, save_filename=None)
            for i in range(0, len(plts)):
                plts[i].close()
        except Exception:
            self.assertTrue(False)

        try:
            pca_X = PCA(X, scaling='auto', n_components=2)
            plts = reduction.plot_eigenvectors(pca_X.Q[:,0], eigenvectors_indices=[], variable_names=[], plot_absolute=False, bar_color=None, title=None, save_filename=None)
            for i in range(0, len(plts)):
                plts[i].close()
        except Exception:
            self.assertTrue(False)

        try:
            pca_X = PCA(X, scaling='auto', n_components=2)
            plts = reduction.plot_eigenvectors(pca_X.Q[:,2:4], eigenvectors_indices=[], variable_names=[], plot_absolute=False, bar_color=None, title=None, save_filename=None)
            for i in range(0, len(plts)):
                plts[i].close()
        except Exception:
            self.assertTrue(False)

        try:
            pca_X = PCA(X, scaling='auto', n_components=2)
            plts = reduction.plot_eigenvectors(pca_X.Q[:,0], eigenvectors_indices=[0], variable_names=['a', 'b', 'c', 'd', 'e'], plot_absolute=True, bar_color='r', title='Title', save_filename=None)
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
            plt = reduction.plot_eigenvectors_comparison((pca_1.Q[:,0], pca_2.Q[:,0], pca_3.Q[:,0]), legend_labels=[], variable_names=[], plot_absolute=False, color_map='coolwarm', title=None, save_filename=None)
            plt.close()
        except Exception:
            self.assertTrue(False)

        try:
            pca_1 = PCA(X, scaling='auto', n_components=2)
            pca_2 = PCA(X, scaling='range', n_components=2)
            pca_3 = PCA(X, scaling='vast', n_components=2)
            plt = reduction.plot_eigenvectors_comparison((pca_1.Q[:,0], pca_2.Q[:,0], pca_3.Q[:,0]), legend_labels=['$a$', '$b$', '$c$'], variable_names=['a', 'b', 'c', 'd', 'e'], plot_absolute=True, color_map='viridis', title='Title', save_filename=None)
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
