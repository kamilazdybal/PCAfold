.. module:: PCA.clustering

Clustering
==========

``clustering.py`` module contains functions for classifying data sets into local clusters and performing some basic operations on clusters [1], [2].

Clustering functions
--------------------

Each function that clusters the data set returns a vector ``idx`` of type ``numpy.ndarray`` of size ``(n_observations,)`` that specifies classification of each observation from the original data set ``X`` to a local cluster.

.. image:: ../images/clustering-idx.png
  :width: 400
  :align: center

.. note:: The first cluster has index ``0`` within all ``idx`` vectors returned. When verbose information is printed with ``verbose=True`` during function execution or on the plots the cluster numeration starts with ``1``.

.. autofunction:: PCA.clustering.variable_bins

.. code:: python

  # var_min                                                var_max
  #   |----------|----------|----------|----------|----------|
  #      bin 1      bin 2      bin 3       bin 4     bin 5

.. autofunction:: PCA.clustering.predefined_variable_bins

.. code:: python

  # var_min     value_1              value_2      value_3  var_max
  #   |----------|--------------------|------------|---------|
  #       bin 1           bin 2            bin 3      bin 4

.. autofunction:: PCA.clustering.mixture_fraction_bins

.. code:: python

  # Z_min           Z_stoich                                 Z_max
  #   |-------|-------|------------|------------|------------|
  #     bin 1   bin 2     bin 3        bin 4         bin 5

.. autofunction:: PCA.clustering.pc_source_bins

.. code:: python

  #                  -offset         +offset
  #                           \    /
  # pc_source_min             | 0 |                          pc_source_max
  #     |----------|----------|---|----------|----------|----------|
  #         bin 1      bin 2   bin 3  bin 4      bin 5      bin 6


.. code:: python

  #                  -offset     0     +offset
  #                           \  |   /
  # pc_source_min             |  |  |                          pc_source_max
  #     |----------|----------|--|--|----------|----------|----------|
  #        bin 1      bin 2    /   \   bin 5      bin 6      bin 7
  #                           /     \
  #                       bin 3       bin 4

.. autofunction:: PCA.clustering.vqpca

VQPCA algorithm used here was first proposed in :cite:`parente2009identification`. The general scheme for the iterative procedure is presented below:

.. image:: ../images/clustering-vqpca.png
  :width: 700
  :align: center

Auxiliary functions
-------------------

.. autofunction:: PCA.clustering.degrade_clusters

.. autofunction:: PCA.clustering.flip_clusters

.. autofunction:: PCA.clustering.get_centroids

.. autofunction:: PCA.clustering.get_partition

.. autofunction:: PCA.clustering.get_populations

Bibliography
------------

.. bibliography:: refs.bib
