.. module:: preprocess

##################
Data preprocessing
##################

The ``preprocess`` module can be used for performing data preprocessing
including centering and scaling, outlier detection and removal, kernel density
weighting of data sets, data clustering and data sampling.

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

*****************
Data manipulation
*****************

This section includes functions for performing basic data manipulation such
as centering and scaling and outlier detection and removal.

``center_scale``
==========================

.. autofunction:: PCAfold.preprocess.center_scale

``invert_center_scale``
==========================

.. autofunction:: PCAfold.preprocess.invert_center_scale

``remove_constant_vars``
==========================

.. autofunction:: PCAfold.preprocess.remove_constant_vars

``order_variables``
==========================

.. autofunction:: PCAfold.preprocess.order_variables

Class ``PreProcessing``
=======================

.. autoclass:: PCAfold.preprocess.PreProcessing

``outlier_detection``
======================

.. autofunction:: PCAfold.preprocess.outlier_detection

Class ``ConditionalStatistics``
===============================

.. autoclass:: PCAfold.preprocess.ConditionalStatistics

Class ``KernelDensity``
=======================

.. autoclass:: PCAfold.preprocess.KernelDensity

--------------------------------------------------------------------------------

***************
Data clustering
***************

This section includes functions for classifying data sets into
local clusters and performing some basic operations on clusters :cite:`Everitt2009`,
:cite:`Kaufman2009`.

Clustering functions
====================

Each function that clusters the data set returns a vector of integers ``idx``
of type ``numpy.ndarray`` of size ``(n_observations,)`` that specifies
classification of each observation from the original data set
:math:`\mathbf{X}` to a local cluster.

.. image:: ../images/clustering-idx.svg
  :width: 400
  :align: center

.. note:: The first cluster has index ``0`` within all ``idx`` vectors returned.

``variable_bins``
-----------------

.. autofunction:: PCAfold.preprocess.variable_bins

``predefined_variable_bins``
----------------------------

.. autofunction:: PCAfold.preprocess.predefined_variable_bins

``mixture_fraction_bins``
-------------------------

.. autofunction:: PCAfold.preprocess.mixture_fraction_bins

``zero_neighborhood_bins``
--------------------------

.. autofunction:: PCAfold.preprocess.zero_neighborhood_bins

Auxiliary functions
===================

``degrade_clusters``
--------------------

.. autofunction:: PCAfold.preprocess.degrade_clusters

``flip_clusters``
-----------------

.. autofunction:: PCAfold.preprocess.flip_clusters

``get_centroids``
-----------------

.. autofunction:: PCAfold.preprocess.get_centroids

``get_partition``
-----------------

.. autofunction:: PCAfold.preprocess.get_partition

``get_populations``
-------------------

.. autofunction:: PCAfold.preprocess.get_populations

``get_average_centroid_distance``
---------------------------------

.. autofunction:: PCAfold.preprocess.get_average_centroid_distance

--------------------------------------------------------------------------------

*************
Data sampling
*************

This section includes functions for splitting data sets into train and test data for use in machine learning algorithms.
Apart from random splitting that can be achieved with the commonly used
`sklearn.model_selection.train_test_split <https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html>`_,
extended methods are implemented here that allow for purposive sampling :cite:`Neyman1992`,
such as drawing samples at certain amount from local clusters :cite:`May2010`, :cite:`Gill2004`.
These functionalities can be specifically used to tackle *imbalanced data sets*
:cite:`He2009`, :cite:`Rastgoo2016`.

The general idea is to divide the entire data set ``X`` (or its portion) into train and test samples as presented below:

.. image:: ../images/tts-train-test-select.svg
  :width: 700
  :align: center

**Train data** is always sampled in the same way for a given sampling function.
Depending on the option selected, **test data** will be sampled differently, either as all
remaining samples that were not included in train data or as a subset of those.
You can select the option by setting the ``test_selection_option`` parameter for each sampling function.
Reach out to the documentation for a specific sampling function to see what options are available.

All splitting functions in this module return a tuple of two variables: ``(idx_train, idx_test)``.
Both ``idx_train`` and ``idx_test`` are vectors of integers of type ``numpy.ndarray`` and of size ``(_,)``.
These variables contain indices of observations that went into train data and test data respectively.

In your model learning algorithm you can then get the train and test observations, for instance in the following way:

.. code:: python

  X_train = X[idx_train,:]
  X_test = X[idx_test,:]

All functions are equipped with ``verbose`` parameter. If it is set to ``True`` some additional information on train and test selection is printed.

.. note:: It is assumed that the first cluster has index ``0`` within all input ``idx`` vectors.

Class ``DataSampler``
=====================

.. autoclass:: PCAfold.preprocess.DataSampler

``DataSampler.number``
======================

.. autofunction:: PCAfold.preprocess.DataSampler.number

``DataSampler.percentage``
==========================

.. autofunction:: PCAfold.preprocess.DataSampler.percentage

``DataSampler.manual``
======================

.. autofunction:: PCAfold.preprocess.DataSampler.manual

``DataSampler.random``
======================

.. autofunction:: PCAfold.preprocess.DataSampler.random

--------------------------------------------------------------------------------

******************
Plotting functions
******************

This section includes functions for data preprocessing related plotting such as
visualizing the formed clusters.

``plot_2d_clustering``
======================

.. autofunction:: PCAfold.preprocess.plot_2d_clustering

``plot_3d_clustering``
======================

.. autofunction:: PCAfold.preprocess.plot_3d_clustering

``plot_2d_train_test_samples``
==============================

.. autofunction:: PCAfold.preprocess.plot_2d_train_test_samples

--------------------------------------------------------------------------------

************
Bibliography
************

.. bibliography:: data-preprocessing.bib
  :labelprefix: P
