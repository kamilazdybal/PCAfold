import unittest
import numpy as np
from PCAfold import preprocess
from PCAfold import reduction
from PCAfold import analysis

class Reduction(unittest.TestCase):

    def test_reduction__plot_2d_manifold_sequence__allowed_calls(self):

        X = np.random.rand(100,5)
        subset_PCA = reduction.SubsetPCA(X)
        xy = subset_PCA.principal_components

        try:
            reduction.plot_2d_manifold_sequence(xy)
        except:
            self.assertTrue(False)

        try:
            reduction.plot_2d_manifold_sequence(xy, color=['k', 'b', 'r'])
        except:
            self.assertTrue(False)

        try:
            reduction.plot_2d_manifold_sequence(xy, color=['k', 'b', 'r'])
        except:
            self.assertTrue(False)

        try:
            reduction.plot_2d_manifold_sequence(xy, color='k')
        except:
            self.assertTrue(False)

        try:
            reduction.plot_2d_manifold_sequence(xy, color=X[:,0], x_label='A', y_label='B ', cbar=True, colorbar_label='V', color_map='viridis', figure_size=(10, 3), title=['A', 'B', 'C'])
        except:
            self.assertTrue(False)

        try:
            reduction.plot_2d_manifold_sequence(xy, color=[X[:,1], X[:,1], X[:,2]], x_label='A', y_label='B ', cbar=True, colorbar_label='V', color_map='viridis', figure_size=(10, 3), title=['A', 'B', 'C'])
        except:
            self.assertTrue(False)

# ------------------------------------------------------------------------------

    def test_reduction__plot_2d_manifold_sequence__not_allowed_calls(self):

        X = np.random.rand(100,5)
        subset_PCA = reduction.SubsetPCA(X)
        xy = subset_PCA.principal_components

        with self.assertRaises(ValueError):
            reduction.plot_2d_manifold_sequence(X)

        with self.assertRaises(ValueError):
            reduction.plot_2d_manifold_sequence([1,2,3])

        with self.assertRaises(ValueError):
            reduction.plot_2d_manifold_sequence(X[:,0])

        with self.assertRaises(ValueError):
            reduction.plot_2d_manifold_sequence(xy, color=[1])

        with self.assertRaises(ValueError):
            reduction.plot_2d_manifold_sequence(xy, color=X[:,0:2])

        with self.assertRaises(ValueError):
            reduction.plot_2d_manifold_sequence(xy, color=[X[:,1], X[:,1]])

        with self.assertRaises(ValueError):
            reduction.plot_2d_manifold_sequence(xy, x_label=[1])

        with self.assertRaises(ValueError):
            reduction.plot_2d_manifold_sequence(xy, y_label=[1])

        with self.assertRaises(ValueError):
            reduction.plot_2d_manifold_sequence(xy, cbar=[1])

        with self.assertRaises(ValueError):
            reduction.plot_2d_manifold_sequence(xy, colorbar_label=[1])

        with self.assertRaises(ValueError):
            reduction.plot_2d_manifold_sequence(xy,color_map=[1])

        with self.assertRaises(ValueError):
            reduction.plot_2d_manifold_sequence(xy, figure_size=[1])

        with self.assertRaises(ValueError):
            reduction.plot_2d_manifold_sequence(xy, title=[1])

        with self.assertRaises(ValueError):
            reduction.plot_2d_manifold_sequence(xy, title='Title')

        with self.assertRaises(ValueError):
            reduction.plot_2d_manifold_sequence(xy, title=['Title'])

# ------------------------------------------------------------------------------
