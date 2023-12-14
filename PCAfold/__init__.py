"""This is the base PCAfold directory."""

__author__ = "Kamila Zdybal, Elizabeth Armstrong, Alessandro Parente and James C. Sutherland"
__copyright__ = "Copyright (c) 2020-2024, Kamila Zdybal and Elizabeth Armstrong"
__credits__ = ["Department of Chemical Engineering, University of Utah, Salt Lake City, Utah, USA", "Universite Libre de Bruxelles, Aero-Thermo-Mechanics Laboratory, Brussels, Belgium"]
__license__ = "MIT"
__version__ = "2.2.0"
__maintainer__ = ["Kamila Zdybal", "Elizabeth Armstrong"]
__email__ = ["kamilazdybal@gmail.com", "Elizabeth.Armstrong@chemeng.utah.edu", "James.Sutherland@chemeng.utah.edu"]
__status__ = "Production"

# Module: `preprocess`
from .preprocess import center_scale
from .preprocess import invert_center_scale
from .preprocess import log_transform
from .preprocess import power_transform
from .preprocess import zero_pivot_transform
from .preprocess import invert_zero_pivot_transform
from .preprocess import remove_constant_vars
from .preprocess import order_variables
from .preprocess import PreProcessing
from .preprocess import outlier_detection
from .preprocess import representative_sample_size
from .preprocess import KernelDensity
from .preprocess import DensityEstimation
from .preprocess import DataSampler
from .preprocess import ConditionalStatistics
from .preprocess import variable_bins
from .preprocess import predefined_variable_bins
from .preprocess import mixture_fraction_bins
from .preprocess import zero_neighborhood_bins
from .preprocess import degrade_clusters
from .preprocess import flip_clusters
from .preprocess import get_centroids
from .preprocess import get_partition
from .preprocess import get_populations
from .preprocess import get_average_centroid_distance
from .preprocess import plot_2d_clustering
from .preprocess import plot_3d_clustering
from .preprocess import plot_2d_train_test_samples
from .preprocess import plot_conditional_statistics

# Module: `reduction`
from .reduction import PCA
from .reduction import LPCA
from .reduction import SubsetPCA
from .reduction import VQPCA
from .reduction import SamplePCA
from .reduction import EquilibratedSamplePCA
from .reduction import analyze_centers_change
from .reduction import analyze_eigenvector_weights_change
from .reduction import analyze_eigenvalue_distribution
from .reduction import plot_2d_manifold
from .reduction import plot_3d_manifold
from .reduction import plot_2d_manifold_sequence
from .reduction import plot_parity
from .reduction import plot_mode
from .reduction import plot_eigenvectors
from .reduction import plot_eigenvectors_comparison
from .reduction import plot_eigenvalue_distribution
from .reduction import plot_eigenvalue_distribution_comparison
from .reduction import plot_cumulative_variance
from .reduction import plot_heatmap
from .reduction import plot_heatmap_sequence

# Module: `analysis`
from .kernel_regression import KReg
from .analysis import compute_normalized_variance
from .analysis import compute_normalized_range
from .analysis import normalized_variance_derivative
from .analysis import find_local_maxima
from .analysis import random_sampling_normalized_variance
from .analysis import feature_size_map
from .analysis import feature_size_map_smooth
from .analysis import cost_function_normalized_variance_derivative
from .analysis import plot_normalized_variance
from .analysis import plot_normalized_variance_comparison
from .analysis import plot_normalized_variance_derivative
from .analysis import plot_normalized_variance_derivative_comparison

# Module: `reconstruction`
from .reconstruction import ANN
from .reconstruction import PartitionOfUnityNetwork
from .reconstruction import init_uniform_partitions
from .reconstruction import RegressionAssessment
from .reconstruction import coefficient_of_determination
from .reconstruction import stratified_coefficient_of_determination
from .reconstruction import stratified_mean_absolute_error
from .reconstruction import stratified_mean_squared_error
from .reconstruction import stratified_mean_squared_logarithmic_error
from .reconstruction import stratified_root_mean_squared_error
from .reconstruction import stratified_normalized_root_mean_squared_error
from .reconstruction import mean_absolute_error
from .reconstruction import max_absolute_error
from .reconstruction import mean_squared_error
from .reconstruction import mean_squared_logarithmic_error
from .reconstruction import root_mean_squared_error
from .reconstruction import normalized_root_mean_squared_error
from .reconstruction import turning_points
from .reconstruction import good_estimate
from .reconstruction import good_direction_estimate
from .reconstruction import generate_tex_table
from .reconstruction import plot_2d_regression
from .reconstruction import plot_2d_regression_scalar_field
from .reconstruction import plot_2d_regression_streamplot
from .reconstruction import plot_3d_regression
from .reconstruction import plot_stratified_metric

# Module: `utilities`
from .utilities import QoIAwareProjection
from .utilities import manifold_informed_forward_variable_addition
from .utilities import manifold_informed_backward_variable_elimination
from .utilities import QoIAwareProjectionPOUnet
