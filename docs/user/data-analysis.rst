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
according to scales of variation and uniqueness of dependent variable values.

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

``r2value``
================================================

.. autofunction:: PCAfold.analysis.r2value

``stratified_r2``
================================================

.. autofunction:: PCAfold.analysis.stratified_r2

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

******************
Plotting functions
******************

``plot_2d_regression``
======================

.. autofunction:: PCAfold.analysis.plot_2d_regression

``plot_3d_regression``
======================

.. autofunction:: PCAfold.analysis.plot_3d_regression

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

``plot_stratified_r2``
===================================================

.. autofunction:: PCAfold.analysis.plot_stratified_r2

--------------------------------------------------------------------------------

************
Bibliography
************

.. bibliography:: data-analysis.bib
  :labelprefix: A
