import unittest
import numpy as np
from PCAfold import preprocess
from PCAfold import reduction
from PCAfold import analysis

class Reduction(unittest.TestCase):

    def test_reduction__analyze_eigenvalue_distribution__allowed_calls(self):

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

# ------------------------------------------------------------------------------
