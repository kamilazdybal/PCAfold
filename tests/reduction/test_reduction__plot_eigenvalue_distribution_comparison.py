import unittest
import numpy as np
from PCAfold import preprocess
from PCAfold import reduction
from PCAfold import analysis

class Reduction(unittest.TestCase):

    def test_reduction__plot_eigenvalue_distribution_comparison__allowed_calls(self):

        X = np.random.rand(100,5)

        try:
            pca_1 = reduction.PCA(X, scaling='auto', n_components=2)
            pca_2 = reduction.PCA(X, scaling='range', n_components=2)
            pca_3 = reduction.PCA(X, scaling='vast', n_components=2)
            plt = reduction.plot_eigenvalue_distribution_comparison((pca_1.L, pca_2.L, pca_3.L), legend_labels=[], normalized=False, color_map='coolwarm', title=None, save_filename=None)
            plt.close()
        except Exception:
            self.assertTrue(False)

        try:
            pca_1 = reduction.PCA(X, scaling='auto', n_components=2)
            pca_2 = reduction.PCA(X, scaling='range', n_components=2)
            pca_3 = reduction.PCA(X, scaling='vast', n_components=2)
            plt = reduction.plot_eigenvalue_distribution_comparison((pca_1.L, pca_2.L, pca_3.L), legend_labels=['Auto', 'Range', 'Vast'], normalized=True, color_map='viridis', title='Title', save_filename=None)
            plt.close()
        except Exception:
            self.assertTrue(False)

# ------------------------------------------------------------------------------
