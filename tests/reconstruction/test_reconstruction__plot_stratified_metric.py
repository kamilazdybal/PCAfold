import unittest
import numpy as np
from PCAfold import preprocess
from PCAfold import reduction
from PCAfold import reconstruction

class Analysis(unittest.TestCase):

    def test_reconstruction__plot_stratified_metric__allowed_calls(self):

        X = np.random.rand(100,5)
        pca_X = reduction.PCA(X, scaling='auto', n_components=2)
        X_rec = pca_X.reconstruct(pca_X.transform(X))
        (idx, bins_borders) = preprocess.variable_bins(X[:,0], k=10, verbose=False)
        r2_in_bins = reconstruction.stratified_coefficient_of_determination(X[:,0], X_rec[:,0], idx=idx, use_global_mean=True, verbose=False)

        try:
            plt = reconstruction.plot_stratified_metric(r2_in_bins, bins_borders, variable_name='$X_1$', metric_name='$R^2$', yscale='log', figure_size=(10,5), title=None, save_filename=None)
            plt.close()
        except Exception:
            self.assertTrue(False)

        try:
            plt = reconstruction.plot_stratified_metric(r2_in_bins, bins_borders)
            plt.close()
        except Exception:
            self.assertTrue(False)

# ------------------------------------------------------------------------------

    def test_reconstruction__plot_stratified_metric__not_allowed_calls(self):

        pass

# ------------------------------------------------------------------------------
