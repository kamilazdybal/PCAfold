import unittest
import numpy as np
from PCAfold import preprocess
from PCAfold import reduction
from PCAfold import analysis
from PCAfold import reconstruction

class Reconstruction(unittest.TestCase):

    def test_reconstruction__plot_2d_regression_streamplot__allowed_calls(self):

        X = np.random.rand(100,5)
        S_X = np.random.rand(100,5)

        pca_X = reduction.PCA(X, n_components=2)
        PCs = pca_X.transform(X)
        S_Z = pca_X.transform(S_X, nocenter=True)

        model = analysis.KReg(PCs, S_Z)

        def regression_model(query):

            predicted = model.predict(query, 'nearest_neighbors_isotropic', n_neighbors=1)

            return predicted

        grid_bounds = ([np.min(PCs[:,0]),np.max(PCs[:,0])],[np.min(PCs[:,1]),np.max(PCs[:,1])])

        try:
            plt = reconstruction.plot_2d_regression_streamplot(grid_bounds, regression_model)
            plt.close()
        except:
            self.assertTrue(False)

        try:
            plt = reconstruction.plot_2d_regression_streamplot(grid_bounds,
                                                regression_model,
                                                x=PCs[:,0],
                                                y=PCs[:,1],
                                                resolution=(15,15),
                                                extension=(20,20),
                                                color='r',
                                                x_label='$Z_1$',
                                                y_label='$Z_2$',
                                                manifold_color=X[:,0],
                                                colorbar_label='$X_1$',
                                                color_map='plasma',
                                                colorbar_range=(0,1),
                                                manifold_alpha=1,
                                                grid_on=False,
                                                figure_size=(10,6),
                                                title='Streamplot',
                                                save_filename=None)
            plt.close()
        except:
            self.assertTrue(False)

# ------------------------------------------------------------------------------

    def test_reconstruction__plot_2d_regression_streamplot__not_allowed_calls(self):

        X = np.random.rand(100,5)
        S_X = np.random.rand(100,5)

        pca_X = reduction.PCA(X, n_components=2)
        PCs = pca_X.transform(X)
        S_Z = pca_X.transform(S_X, nocenter=True)

        model = analysis.KReg(PCs, S_Z)

        def regression_model(query):

            predicted = model.predict(query, 'nearest_neighbors_isotropic', n_neighbors=1)

            return predicted

        grid_bounds = ([np.min(PCs[:,0]),np.max(PCs[:,0])],[np.min(PCs[:,1]),np.max(PCs[:,1])])

        with self.assertRaises(ValueError):
            plt = reconstruction.plot_2d_regression_streamplot([[0,1],[0,1]], regression_model)
            plt.close()

        with self.assertRaises(ValueError):
            plt = reconstruction.plot_2d_regression_streamplot(([0,1],[0,1],[0,1]), regression_model)
            plt.close()

        with self.assertRaises(ValueError):
            plt = reconstruction.plot_2d_regression_streamplot(grid_bounds, [1])
            plt.close()

        with self.assertRaises(ValueError):
            plt = reconstruction.plot_2d_regression_streamplot(grid_bounds, regression_model, x=[1])
            plt.close()

        with self.assertRaises(ValueError):
            plt = reconstruction.plot_2d_regression_streamplot(grid_bounds, regression_model, y=[1])
            plt.close()

        with self.assertRaises(ValueError):
            plt = reconstruction.plot_2d_regression_streamplot(grid_bounds, regression_model, resolution=[1])
            plt.close()

        with self.assertRaises(ValueError):
            plt = reconstruction.plot_2d_regression_streamplot(grid_bounds, regression_model, extension=[1])
            plt.close()

        with self.assertRaises(ValueError):
            plt = reconstruction.plot_2d_regression_streamplot(grid_bounds, regression_model, color=[1])
            plt.close()

        with self.assertRaises(ValueError):
            plt = reconstruction.plot_2d_regression_streamplot(grid_bounds, regression_model, x_label=[1])
            plt.close()

        with self.assertRaises(ValueError):
            plt = reconstruction.plot_2d_regression_streamplot(grid_bounds, regression_model, y_label=[1])
            plt.close()

        with self.assertRaises(ValueError):
            plt = reconstruction.plot_2d_regression_streamplot(grid_bounds, regression_model, manifold_color=[1])
            plt.close()

        with self.assertRaises(ValueError):
            plt = reconstruction.plot_2d_regression_streamplot(grid_bounds, regression_model, colorbar_label=[1])
            plt.close()

        with self.assertRaises(ValueError):
            plt = reconstruction.plot_2d_regression_streamplot(grid_bounds, regression_model, color_map=[1])
            plt.close()

        with self.assertRaises(ValueError):
            plt = reconstruction.plot_2d_regression_streamplot(grid_bounds, regression_model, colorbar_range=[1])
            plt.close()

        with self.assertRaises(ValueError):
            plt = reconstruction.plot_2d_regression_streamplot(grid_bounds, regression_model, manifold_alpha=[1])
            plt.close()

        with self.assertRaises(ValueError):
            plt = reconstruction.plot_2d_regression_streamplot(grid_bounds, regression_model, grid_on=[1])
            plt.close()

        with self.assertRaises(ValueError):
            plt = reconstruction.plot_2d_regression_streamplot(grid_bounds, regression_model, figure_size=[1])
            plt.close()

        with self.assertRaises(ValueError):
            plt = reconstruction.plot_2d_regression_streamplot(grid_bounds, regression_model, title=[1])
            plt.close()

        with self.assertRaises(ValueError):
            plt = reconstruction.plot_2d_regression_streamplot(grid_bounds, regression_model, save_filename=[1])
            plt.close()

# ------------------------------------------------------------------------------
