.. module:: PCA

Clustering
==========

``clustering.py`` module contains functions for classifying data sets into local clusters and performing some basic operations on clusters [1], [2].

Clustering functions
--------------------

Each function that clusters the data set returns a vector ``idx`` of type ``numpy.ndarray`` of size ``(n_observations,)`` that specifies classification of each observation from the original data set ``X`` to a local cluster.

.. image:: ../images/clustering-idx.png
  :width: 200

.. note:: The first cluster has index ``0`` within all ``clustering.py`` functions. When verbose information is printed with ``verbose=True`` during function execution or on the plots the cluster numeration starts with ``1``.


.. autofunction:: PCA.clustering.variable_bins

.. autofunction:: PCA.clustering.predefined_variable_bins

.. autofunction:: PCA.clustering.mixture_fraction_bins

.. autofunction:: PCA.clustering.pc_source_bins

.. autofunction:: PCA.clustering.kmeans

.. autofunction:: PCA.clustering.vqpca

.. image:: ../images/clustering-vqpca.png
  :width: 700

Auxiliary functions
-------------------



.. autofunction:: PCA.clustering.degrade_clusters

.. autofunction:: PCA.clustering.flip_clusters

.. autofunction:: PCA.clustering.get_centroids

.. autofunction:: PCA.clustering.get_partition

.. autofunction:: PCA.clustering.get_populations
