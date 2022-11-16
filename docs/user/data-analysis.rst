.. module:: analysis

#################
Manifold analysis
#################

The ``analysis`` module contains functions for assessing the intrinsic
dimensionality and quality of manifolds.

.. note:: The format for the user-supplied input data matrix
  :math:`\mathbf{X} \in \mathbb{R}^{N \times Q}`, common to all modules, is that
  :math:`N` observations are stored in rows and :math:`Q` variables are stored
  in columns. Since typically :math:`N \gg Q`, the initial dimensionality of the
  data set is determined by the number of variables, :math:`Q`.

  .. math::

    \mathbf{X} =
    \begin{bmatrix}
    \vdots & \vdots & & \vdots \\
    X_1 & X_2 & \dots & X_{Q} \\
    \vdots & \vdots & & \vdots \\
    \end{bmatrix}

  The general agreement throughout this documentation is that :math:`i` will
  index observations and :math:`j` will index variables.

  The representation of the user-supplied data matrix in **PCAfold**
  is the input parameter ``X``, which should be of type ``numpy.ndarray``
  and of size ``(n_observations,n_variables)``.

--------------------------------------------------------------------------------

***********************
Manifold assessment
***********************

This section includes functions for quantitative assessments of
manifold dimensionality and for comparing manifold parameterizations
according to scales of variation and uniqueness of dependent variable values
as introduced in :cite:`Armstrong2021` and :cite:`zdybal2022cost`.

``compute_normalized_variance``
================================================

.. autofunction:: PCAfold.analysis.compute_normalized_variance

Class ``VarianceData``
======================

.. autoclass:: PCAfold.analysis.VarianceData

``normalized_variance_derivative``
================================================

.. autofunction:: PCAfold.analysis.normalized_variance_derivative

``find_local_maxima``
================================================

.. autofunction:: PCAfold.analysis.find_local_maxima

``random_sampling_normalized_variance``
================================================

.. autofunction:: PCAfold.analysis.random_sampling_normalized_variance

``feature_size_map``
================================================

.. autofunction:: PCAfold.analysis.feature_size_map

``feature_size_map_smooth``
================================================

.. autofunction:: PCAfold.analysis.feature_size_map_smooth

``cost_function_normalized_variance_derivative``
================================================

.. autofunction:: PCAfold.analysis.cost_function_normalized_variance_derivative

``manifold_informed_feature_selection``
================================================

.. autofunction:: PCAfold.analysis.manifold_informed_feature_selection

``manifold_informed_backward_elimination``
================================================

.. autofunction:: PCAfold.analysis.manifold_informed_backward_elimination

--------------------------------------------------------------------------------

******************
Kernel Regression
******************

This section includes details on the Nadaraya-Watson kernel regression
:cite:`Hardle1990` used in assessing manifolds. The ``KReg`` class may be used
for non-parametric regression in general.

Class ``KReg``
================================================

.. autoclass:: PCAfold.kernel_regression.KReg

``KReg.predict``
================================================

.. autofunction:: PCAfold.kernel_regression.KReg.predict

``KReg.compute_constant_bandwidth``
================================================

.. autofunction:: PCAfold.kernel_regression.KReg.compute_constant_bandwidth

``KReg.compute_bandwidth_isotropic``
================================================

.. autofunction:: PCAfold.kernel_regression.KReg.compute_bandwidth_isotropic

``KReg.compute_bandwidth_anisotropic``
================================================

.. autofunction:: PCAfold.kernel_regression.KReg.compute_bandwidth_anisotropic

``KReg.compute_nearest_neighbors_bandwidth_isotropic``
=========================================================

.. autofunction:: PCAfold.kernel_regression.KReg.compute_nearest_neighbors_bandwidth_isotropic

``KReg.compute_nearest_neighbors_bandwidth_anisotropic``
=========================================================

.. autofunction:: PCAfold.kernel_regression.KReg.compute_nearest_neighbors_bandwidth_anisotropic

--------------------------------------------------------------------------------

***********************
Regression assessment
***********************

Class ``RegressionAssessment``
================================================

.. autoclass:: PCAfold.analysis.RegressionAssessment

``RegressionAssessment.print_metrics``
================================================

.. autofunction:: PCAfold.analysis.RegressionAssessment.print_metrics

``RegressionAssessment.print_stratified_metrics``
=================================================

.. autofunction:: PCAfold.analysis.RegressionAssessment.print_stratified_metrics

``coefficient_of_determination``
================================

.. autofunction:: PCAfold.analysis.coefficient_of_determination

``stratified_coefficient_of_determination``
===========================================

.. autofunction:: PCAfold.analysis.stratified_coefficient_of_determination

``mean_absolute_error``
=======================

.. autofunction:: PCAfold.analysis.mean_absolute_error

``stratified_mean_absolute_error``
==================================

.. autofunction:: PCAfold.analysis.stratified_mean_absolute_error

``max_absolute_error``
=======================

.. autofunction:: PCAfold.analysis.max_absolute_error

``mean_squared_error``
======================

.. autofunction:: PCAfold.analysis.mean_squared_error

``stratified_mean_squared_error``
=================================

.. autofunction:: PCAfold.analysis.stratified_mean_squared_error

``root_mean_squared_error``
===========================

.. autofunction:: PCAfold.analysis.root_mean_squared_error

``stratified_root_mean_squared_error``
======================================

.. autofunction:: PCAfold.analysis.stratified_root_mean_squared_error

``normalized_root_mean_squared_error``
======================================

.. autofunction:: PCAfold.analysis.normalized_root_mean_squared_error

``stratified_normalized_root_mean_squared_error``
=================================================

.. autofunction:: PCAfold.analysis.stratified_normalized_root_mean_squared_error

``turning_points``
================================================

.. autofunction:: PCAfold.analysis.turning_points

``good_estimate``
================================================

.. autofunction:: PCAfold.analysis.good_estimate

``good_direction_estimate``
================================================

.. autofunction:: PCAfold.analysis.good_direction_estimate

``generate_tex_table``
================================================

.. autofunction:: PCAfold.analysis.generate_tex_table

--------------------------------------------------------------------------------

******************
Plotting functions
******************

``plot_2d_regression``
======================

.. autofunction:: PCAfold.analysis.plot_2d_regression

``plot_2d_regression_scalar_field``
============================================

.. autofunction:: PCAfold.analysis.plot_2d_regression_scalar_field

``plot_2d_regression_streamplot``
============================================

.. autofunction:: PCAfold.analysis.plot_2d_regression_streamplot

``plot_normalized_variance``
============================

.. autofunction:: PCAfold.analysis.plot_normalized_variance

``plot_normalized_variance_comparison``
=======================================

.. autofunction:: PCAfold.analysis.plot_normalized_variance_comparison

``plot_normalized_variance_derivative``
========================================

.. autofunction:: PCAfold.analysis.plot_normalized_variance_derivative

``plot_normalized_variance_derivative_comparison``
===================================================

.. autofunction:: PCAfold.analysis.plot_normalized_variance_derivative_comparison

``plot_stratified_metric``
===================================================

.. autofunction:: PCAfold.analysis.plot_stratified_metric

--------------------------------------------------------------------------------

************
Bibliography
************

.. bibliography:: data-analysis.bib
  :labelprefix: A
