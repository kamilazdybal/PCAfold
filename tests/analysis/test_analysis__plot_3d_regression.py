import unittest
import numpy as np
from PCAfold import preprocess
from PCAfold import reduction
from PCAfold import analysis

class Analysis(unittest.TestCase):

    def test_analysis__plot_3d_regression__allowed_calls(self):

        X = np.random.rand(100,5)
        pca_X = reduction.PCA(X, scaling='auto', n_components=2)
        X_rec = pca_X.reconstruct(pca_X.transform(X))

        try:
            plt = analysis.plot_3d_regression(X[:,0], X[:,1], X[:,0], X_rec[:,0])
            plt.close()
        except Exception:
            self.assertTrue(False)

        try:
            plt = analysis.plot_3d_regression(X[:,0],
                                            X[:,1],
                                            X[:,0],
                                            X_rec[:,0],
                                            elev=45,
                                            azim=-45,
                                            x_label='$x$',
                                            y_label='$y$',
                                            z_label='$z$',
                                            figure_size=(7,7),
                                            title='3D regression')
            plt.close()
        except Exception:
            self.assertTrue(False)

        try:
            plt = analysis.plot_3d_regression(X[:,0:1],
                                            X[:,1:2],
                                            X[:,2:3],
                                            X_rec[:,0:1],
                                            elev=45,
                                            azim=-45,
                                            x_label='$x$',
                                            y_label='$y$',
                                            z_label='$z$',
                                            figure_size=(7,7),
                                            title='3D regression')
            plt.close()
        except Exception:
            self.assertTrue(False)

# ------------------------------------------------------------------------------

    def test_analysis__plot_3d_regression__not_allowed_calls(self):

        X = np.random.rand(100,5)

        with self.assertRaises(ValueError):
            plt = analysis.plot_3d_regression(X[:,0:2], X[:,1], X[:,2], X[:,0])
            plt.close()

        with self.assertRaises(ValueError):
            plt = analysis.plot_3d_regression(X[:,0], X[:,1:3], X[:,2], X[:,0])
            plt.close()

        with self.assertRaises(ValueError):
            plt = analysis.plot_3d_regression(X[:,0], X[:,1], X[:,2:4], X[:,0])
            plt.close()

        with self.assertRaises(ValueError):
            plt = analysis.plot_3d_regression(X[:,0], X[:,1], X[:,2], X[:,0:2])
            plt.close()

        with self.assertRaises(ValueError):
            plt = analysis.plot_3d_regression([1,2,3], X[:,1], X[:,2], X[:,2])
            plt.close()

        with self.assertRaises(ValueError):
            plt = analysis.plot_3d_regression(X[:,0], [1,2,3], X[:,2], X[:,2])
            plt.close()

        with self.assertRaises(ValueError):
            plt = analysis.plot_3d_regression(X[:,0], X[:,1], [1,2,3], X[:,2])
            plt.close()

        with self.assertRaises(ValueError):
            plt = analysis.plot_3d_regression(X[:,0], X[:,1], X[:,2], [1,2,3])
            plt.close()

        with self.assertRaises(ValueError):
            plt = analysis.plot_3d_regression(X[:,0], X[:,1], X[:,2], X[:,2], elev=[1])
            plt.close()

        with self.assertRaises(ValueError):
            plt = analysis.plot_3d_regression(X[:,0], X[:,1], X[:,2], X[:,2], azim=[1])
            plt.close()

        with self.assertRaises(ValueError):
            plt = analysis.plot_3d_regression(X[:,0], X[:,1], X[:,2], X[:,2], x_label=[1])
            plt.close()

        with self.assertRaises(ValueError):
            plt = analysis.plot_3d_regression(X[:,0], X[:,1], X[:,2], X[:,2], y_label=[1])
            plt.close()

        with self.assertRaises(ValueError):
            plt = analysis.plot_3d_regression(X[:,0], X[:,1], X[:,2], X[:,2], z_label=[1])
            plt.close()

        with self.assertRaises(ValueError):
            plt = analysis.plot_3d_regression(X[:,0], X[:,1], X[:,2], X[:,2], figure_size=[1])
            plt.close()

        with self.assertRaises(ValueError):
            plt = analysis.plot_3d_regression(X[:,0], X[:,1], X[:,2], X[:,2], title=[1])
            plt.close()

        with self.assertRaises(ValueError):
            plt = analysis.plot_3d_regression(X[:,0], X[:,1], X[:,2], X[:,2], save_filename=[1])
            plt.close()

# ------------------------------------------------------------------------------
