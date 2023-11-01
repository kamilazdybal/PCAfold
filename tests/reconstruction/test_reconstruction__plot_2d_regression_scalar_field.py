import unittest
import numpy as np
from PCAfold import preprocess
from PCAfold import reduction
from PCAfold import analysis
from PCAfold import reconstruction

class Reconstruction(unittest.TestCase):

    def test_reconstruction__plot_2d_regression_scalar_field__allowed_calls(self):

        # Generate dummy data set:
        X = np.random.rand(100,2)
        Z = np.random.rand(100,1)

        # Train the kernel regression model:
        model = analysis.KReg(X, Z)

        # Define the regression model:
        def regression_model(query):

            predicted = model.predict(query, 'nearest_neighbors_isotropic', n_neighbors=1)[0,0]

            return predicted

        # Define the bounds for the streamplot:
        grid_bounds = ([np.min(X[:,0]),np.max(X[:,0])],[np.min(X[:,1]),np.max(X[:,1])])

        try:
            plt = reconstruction.plot_2d_regression_scalar_field(grid_bounds,regression_model)
            plt.close()
        except:
            self.assertTrue(False)

        try:
            plt = reconstruction.plot_2d_regression_scalar_field(grid_bounds,
                                                regression_model,
                                                x=X[:,0],
                                                y=X[:,1],
                                                resolution=(100,100),
                                                extension=(10,10),
                                                x_label='$X_1$',
                                                y_label='$X_2$',
                                                s_field=4,
                                                s_manifold=60,
                                                manifold_color=Z,
                                                colorbar_label='$Z_1$',
                                                color_map='inferno',
                                                colorbar_range=(0,1),
                                                manifold_alpha=1,
                                                grid_on=False,
                                                figure_size=(10,6),
                                                title='2D regressed scalar field')
            plt.close()
        except:
            self.assertTrue(False)

# ------------------------------------------------------------------------------

    def test_reconstruction__plot_2d_regression_scalar_field__not_allowed_calls(self):

        # Generate dummy data set:
        X = np.random.rand(100,2)
        Z = np.random.rand(100,1)

        # Train the kernel regression model:
        model = analysis.KReg(X, Z)

        # Define the regression model:
        def regression_model(query):

            predicted = model.predict(query, 'nearest_neighbors_isotropic', n_neighbors=1)[0,0]

            return predicted

        # Define the bounds for the streamplot:
        grid_bounds = ([np.min(X[:,0]),np.max(X[:,0])],[np.min(X[:,1]),np.max(X[:,1])])

        with self.assertRaises(ValueError):
            plt = reconstruction.plot_2d_regression_scalar_field([[0,1],[0,1]], regression_model)
            plt.close()

        with self.assertRaises(ValueError):
            plt = reconstruction.plot_2d_regression_scalar_field(([0,1],[0,1],[0,1]), regression_model)
            plt.close()

        with self.assertRaises(ValueError):
            plt = reconstruction.plot_2d_regression_scalar_field(grid_bounds, [1])
            plt.close()

        with self.assertRaises(ValueError):
            plt = reconstruction.plot_2d_regression_scalar_field(grid_bounds, regression_model, x=[1])
            plt.close()

        with self.assertRaises(ValueError):
            plt = reconstruction.plot_2d_regression_scalar_field(grid_bounds, regression_model, y=[1])
            plt.close()

        with self.assertRaises(ValueError):
            plt = reconstruction.plot_2d_regression_scalar_field(grid_bounds, regression_model, resolution=[1])
            plt.close()

        with self.assertRaises(ValueError):
            plt = reconstruction.plot_2d_regression_scalar_field(grid_bounds, regression_model, extension=[1])
            plt.close()

        with self.assertRaises(ValueError):
            plt = reconstruction.plot_2d_regression_scalar_field(grid_bounds, regression_model, x_label=[1])
            plt.close()

        with self.assertRaises(ValueError):
            plt = reconstruction.plot_2d_regression_scalar_field(grid_bounds, regression_model, y_label=[1])
            plt.close()

        with self.assertRaises(ValueError):
            plt = reconstruction.plot_2d_regression_scalar_field(grid_bounds, regression_model, s_field=[1])
            plt.close()

        with self.assertRaises(ValueError):
            plt = reconstruction.plot_2d_regression_scalar_field(grid_bounds, regression_model, s_manifold=[1])
            plt.close()

        with self.assertRaises(ValueError):
            plt = reconstruction.plot_2d_regression_scalar_field(grid_bounds, regression_model, manifold_color=[1])
            plt.close()

        with self.assertRaises(ValueError):
            plt = reconstruction.plot_2d_regression_scalar_field(grid_bounds, regression_model, colorbar_label=[1])
            plt.close()

        with self.assertRaises(ValueError):
            plt = reconstruction.plot_2d_regression_scalar_field(grid_bounds, regression_model, color_map=[1])
            plt.close()

        with self.assertRaises(ValueError):
            plt = reconstruction.plot_2d_regression_scalar_field(grid_bounds, regression_model, colorbar_range=[1])
            plt.close()

        with self.assertRaises(ValueError):
            plt = reconstruction.plot_2d_regression_scalar_field(grid_bounds, regression_model, manifold_alpha=[1])
            plt.close()

        with self.assertRaises(ValueError):
            plt = reconstruction.plot_2d_regression_scalar_field(grid_bounds, regression_model, grid_on=[1])
            plt.close()

        with self.assertRaises(ValueError):
            plt = reconstruction.plot_2d_regression_scalar_field(grid_bounds, regression_model, figure_size=[1])
            plt.close()

        with self.assertRaises(ValueError):
            plt = reconstruction.plot_2d_regression_scalar_field(grid_bounds, regression_model, title=[1])
            plt.close()

        with self.assertRaises(ValueError):
            plt = reconstruction.plot_2d_regression_scalar_field(grid_bounds, regression_model, save_filename=[1])
            plt.close()

# ------------------------------------------------------------------------------
