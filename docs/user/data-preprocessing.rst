.. module:: preprocess

##################
Data preprocessing
##################

``preprocess`` module can be used for performing data pre-processing
including centering and scaling, outlier detection and removal, data clustering
and data sampling.

--------------------------------------------------------------------------------

*****************
Data manipulation
*****************

.. note:: It is assumed that the rows of the data set :math:`\mathbf{X}` correspond to
  observations and columns correspond to variables.
  It has dimensions ``(n_observations, n_variables)``.

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

Class ``PreProcessing``
=======================

.. autoclass:: PCAfold.preprocess.PreProcessing

``analyze_centers_change``
==========================

.. autofunction:: PCAfold.preprocess.analyze_centers_change

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

.. image:: ../images/clustering-idx.png
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

``source_bins``
---------------

.. autofunction:: PCAfold.preprocess.source_bins

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

--------------------------------------------------------------------------------

*************
Data sampling
*************

Functions for splitting data sets into train and test data for use in machine learning algorithms.
Apart from random splitting that can be achieved with the commonly used `sklearn.model_selection.train_test_split <https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html>`_, new methods are implemented here that allow for purposive sampling, such as drawing samples at certain amount from local clusters :cite:`May2010`, :cite:`Gill2004`.

The general idea is to divide the entire data set ``X`` (or its portion) into train and test samples as presented below:

.. image:: ../images/tts-train-test-select.png
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

``plot_2d_clustering``
======================

.. autofunction:: PCAfold.preprocess.plot_2d_clustering

``plot_2d_train_test_samples``
==============================

.. autofunction:: PCAfold.preprocess.plot_2d_train_test_samples

--------------------------------------------------------------------------------

************
Bibliography
************

.. bibliography:: data-preprocessing.bib
