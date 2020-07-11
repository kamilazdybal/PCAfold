PCAfold
============

*PCAfold* is a Python library for generating and analyzing empirical low-dimensional manifolds obtained via Principal Component Analysis (PCA).
It incorporates data pre-processing, clustering and sampling techniques using PCA under the hood.

A general overview for how *PCAfold* modules can combine into one workflow is presented below:

.. image:: images/PCA-Python-overview.png
  :width: 700
  :align: center

Each module's functionalities can also be used as a standalone tool for performing a specific task.

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
