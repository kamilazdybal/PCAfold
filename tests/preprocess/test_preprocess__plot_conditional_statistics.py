import unittest
import numpy as np
from PCAfold import preprocess
from PCAfold import reduction
from PCAfold import analysis

class Preprocess(unittest.TestCase):

    def test_preprocess__plot_conditional_statistics__allowed_calls(self):

        conditioning_variable = np.linspace(-1,1,1000)
        y = -conditioning_variable**2 + 1

        try:
            plt = preprocess.plot_conditional_statistics(y, conditioning_variable, k=10, x_label='$x$', y_label='$y$', figure_size=(10,3), title='Conditional mean')
            plt.close()
        except:
            self.assertTrue(False)

        try:
            plt = preprocess.plot_conditional_statistics(y, conditioning_variable, split_values=[-0.5,0.6], x_label='$x$', y_label='$y$', figure_size=(10,3), title='Conditional mean')
            plt.close()
        except:
            self.assertTrue(False)

        X = np.random.rand(100,)
        cond_variable = np.random.rand(100,)

        try:
            plt = preprocess.plot_conditional_statistics(X, cond_variable, k=2)
            plt.close()
        except:
            self.assertTrue(False)

        X = np.random.rand(100,1)
        cond_variable = np.random.rand(100,1)

        try:
            plt = preprocess.plot_conditional_statistics(X, cond_variable, k=2)
            plt.close()
        except:
            self.assertTrue(False)

        X = np.random.rand(100,)
        cond_variable = np.random.rand(100,)

        try:
            plt = preprocess.plot_conditional_statistics(X, cond_variable, k=2)
            plt.close()
        except:
            self.assertTrue(False)

# ------------------------------------------------------------------------------

    def test_preprocess__plot_conditional_statistics__not_allowed_calls(self):

        conditioning_variable = np.linspace(-1,1,1000)
        y = -conditioning_variable**2 + 1

        with self.assertRaises(ValueError):
            plt = preprocess.plot_conditional_statistics([1,2,3], conditioning_variable)

        with self.assertRaises(ValueError):
            plt = preprocess.plot_conditional_statistics(y, [1,2,3])

        with self.assertRaises(ValueError):
            plt = preprocess.plot_conditional_statistics(y, conditioning_variable, k=0)

        with self.assertRaises(ValueError):
            plt = preprocess.plot_conditional_statistics(y, conditioning_variable, k=-1)

        with self.assertRaises(ValueError):
            plt = preprocess.plot_conditional_statistics(y, conditioning_variable, k=2.5)

        with self.assertRaises(ValueError):
            plt = preprocess.plot_conditional_statistics(y, conditioning_variable, split_values=1)

        with self.assertRaises(ValueError):
            plt = preprocess.plot_conditional_statistics(y, conditioning_variable, statistics_to_plot=['none'])

        with self.assertRaises(ValueError):
            plt = preprocess.plot_conditional_statistics(y, conditioning_variable, statistics_to_plot=['mean', 'none'])

        with self.assertRaises(ValueError):
            plt = preprocess.plot_conditional_statistics(y, conditioning_variable, color=1)

        with self.assertRaises(ValueError):
            plt = preprocess.plot_conditional_statistics(y, conditioning_variable, x_label=1)

        with self.assertRaises(ValueError):
            plt = preprocess.plot_conditional_statistics(y, conditioning_variable, y_label=1)

        with self.assertRaises(ValueError):
            plt = preprocess.plot_conditional_statistics(y, conditioning_variable, colorbar_label=1)

        with self.assertRaises(ValueError):
            plt = preprocess.plot_conditional_statistics(y, conditioning_variable, color_map=1)

        with self.assertRaises(ValueError):
            plt = preprocess.plot_conditional_statistics(y, conditioning_variable, figure_size=1)

        with self.assertRaises(ValueError):
            plt = preprocess.plot_conditional_statistics(y, conditioning_variable, title=1)

        with self.assertRaises(ValueError):
            plt = preprocess.plot_conditional_statistics(y, conditioning_variable, save_filename=1)

# ------------------------------------------------------------------------------
