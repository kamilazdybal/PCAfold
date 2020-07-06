PCA-Python
==========

This is a documentation for the ``PCA-Python`` repository.
The core functionality is performing dimensionality reduction using Principal Component Analysis (PCA).

General notions
---------------

It is assumed that the raw data set ``X`` that is an input parameter for many functions in this project has dimensions ``(n_observations, n_variables)``:

.. image:: images/data-set.png
  :width: 300
  :align: center

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   user/PCA
   user/clustering
   user/cluster-biased-pca
   user/manifold-dimensionality
   user/train-test-select

.. toctree::
   :maxdepth: 2
   :caption: Tutorials

   tutorials/train-test-selection
