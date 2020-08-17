"""
This is the base PCAfold directory.
"""

# Module: `preprocess`
from .preprocess import center_scale
from .preprocess import invert_center_scale
from .preprocess import remove_constant_vars
from .preprocess import PreProcessing
from .preprocess import analyze_centers_change
from .preprocess import DataSampler
from .preprocess import variable_bins
from .preprocess import predefined_variable_bins
from .preprocess import mixture_fraction_bins
from .preprocess import source_bins
from .preprocess import degrade_clusters
from .preprocess import flip_clusters
from .preprocess import get_centroids
from .preprocess import get_partition
from .preprocess import get_populations
from .preprocess import plot_2d_clustering
from .preprocess import plot_2d_train_test_samples

# Module: `reduction`
from .reduction import PCA
from .reduction import analyze_eigenvector_weights_movement
from .reduction import analyze_eigenvalue_distribution
from .reduction import equilibrate_cluster_populations
from .reduction import plot_2d_manifold

# Module: `analysis`
from .kernel_regression import KReg
from .analysis import compute_normalized_variance
from .analysis import r2value
from .analysis import logistic_fit
from .analysis import assess_manifolds
