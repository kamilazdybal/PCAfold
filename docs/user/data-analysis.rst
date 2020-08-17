.. module:: analysis

#############
Data analysis
#############

``analysis`` module contains functions for assessing the intrinsic
dimensionality and quality of manifolds.

--------------------------------------------------------------------------------

***********************
Manifold assessment
***********************

This section includes functions for quantitative assessments of
manifold dimensionality and for comparing manifold parameterizations
according to scales of variation and uniqueness of dependent variables values.

``compute_normalized_variance``
================================================

.. autofunction:: PCAfold.analysis.compute_normalized_variance

Class ``VarianceData``
======================

.. autoclass:: PCAfold.analysis.VarianceData

``r2value``
================================================

.. autofunction:: PCAfold.analysis.r2value

``logistic_fit``
================================================

.. autofunction:: PCAfold.analysis.logistic_fit

``assess_manifolds``
================================================

.. autofunction:: PCAfold.analysis.assess_manifolds

--------------------------------------------------------------------------------

******************
Kernel Regression
******************

This section includes details on the Nadaraya-Watson kernel regression
:cite:`Hardle1990` used in assessing manifolds. The ``KReg`` class may be used
for non-parametric regression in general.

``KReg``
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

************
Bibliography
************

.. bibliography:: data-analysis.bib
