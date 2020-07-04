.. module:: clustering

Clustering
==========

This is a documentation for ``clustering.py`` module. It contains functions for classifying data sets into local clusters and performing some basic operations on clusters [1], [2].

Clustering functions
--------------------

Each function that clusters the data set returns a vector ``idx`` of type ``numpy.ndarray`` of size ``(n_observations,)`` that specifies classification of each observation from the original data set ``X`` to a local cluster.

.. image:: ../images/clustering-idx.png
  :width: 200

.. note:: The first cluster has index ``0`` within all ``clustering.py`` functions. When some additional information is printed with ``verbose=True`` during function execution the cluster numeration starts with ``1``.


.. autofunction:: clustering.variable_bins






Auxiliary functions
-------------------


.. autofunction:: clustering.degrade_clusters

.. autofunction:: clustering.flip_clusters

.. autofunction:: clustering.get_centroids

.. autofunction:: clustering.get_partition

.. autofunction:: clustering.get_populations
