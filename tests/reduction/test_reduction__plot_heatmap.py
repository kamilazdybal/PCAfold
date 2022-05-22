import unittest
import numpy as np
from PCAfold import preprocess
from PCAfold import reduction
from PCAfold import analysis

class Reduction(unittest.TestCase):

    def test_reduction__plot_heatmap__allowed_calls(self):

        X = np.random.rand(100,5)
        pca_X = reduction.PCA(X)
        M = pca_X.S

        try:
            plt = reduction.plot_heatmap(M)
            plt.close()
        except Exception:
            self.assertTrue(False)

        try:
            plt = reduction.plot_heatmap(M[:,0:3])
            plt.close()
        except Exception:
            self.assertTrue(False)

        try:
            plt = reduction.plot_heatmap(M, annotate=True, text_color='w', format_displayed='%.2f', color_map='viridis', cbar=False, colorbar_label=None, figure_size=(5, 5), title=None)
            plt.close()
        except Exception:
            self.assertTrue(False)

        try:
            plt = reduction.plot_heatmap(M, annotate=True, text_color='w', format_displayed='%.1f', x_ticks=True, y_ticks=True, color_map='viridis', cbar=True, colorbar_label='C', figure_size=(10, 5), title='Title')
            plt.close()
        except Exception:
            self.assertTrue(False)

        X = np.random.rand(10,1)

        try:
            plt = reduction.plot_heatmap(X)
            plt.close()
        except Exception:
            self.assertTrue(False)

        X = np.random.rand(1,10)

        try:
            plt = reduction.plot_heatmap(X)
            plt.close()
        except Exception:
            self.assertTrue(False)

# ------------------------------------------------------------------------------

    def test_reduction__plot_heatmap__not_allowed_calls(self):

        X = np.random.rand(10,)

        with self.assertRaises(ValueError):
            plt = reduction.plot_heatmap(X)

        with self.assertRaises(ValueError):
            plt = reduction.plot_heatmap([1])

        X = np.random.rand(100,5)
        pca_X = reduction.PCA(X)
        M = pca_X.S

        with self.assertRaises(ValueError):
            plt = reduction.plot_heatmap(M, annotate=[1])

        with self.assertRaises(ValueError):
            plt = reduction.plot_heatmap(M, text_color=[1])

        with self.assertRaises(ValueError):
            plt = reduction.plot_heatmap(M, format_displayed=[1])

        with self.assertRaises(ValueError):
            plt = reduction.plot_heatmap(M, x_ticks=1)

        with self.assertRaises(ValueError):
            plt = reduction.plot_heatmap(M, y_ticks=1)

        with self.assertRaises(ValueError):
            plt = reduction.plot_heatmap(M, color_map=[1])

        with self.assertRaises(ValueError):
            plt = reduction.plot_heatmap(M, cbar=[1])

        with self.assertRaises(ValueError):
            plt = reduction.plot_heatmap(M, colorbar_label=[1])

        with self.assertRaises(ValueError):
            plt = reduction.plot_heatmap(M, figure_size=[1])

        with self.assertRaises(ValueError):
            plt = reduction.plot_heatmap(M, title=[1])

        with self.assertRaises(ValueError):
            plt = reduction.plot_heatmap(M, save_filename=[1])

# ------------------------------------------------------------------------------
