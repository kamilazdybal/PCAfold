PCAfold
============

*Low-dimensional PCA-derived manifolds and everything in between!*

--------------------------------------------------------------------------------

**PCAfold** is a Python software for generating, improving and analyzing
empirical low-dimensional manifolds obtained via Principal Component Analysis
(PCA).

**PCAfold** incorporates data pre-processing, clustering and sampling using PCA under the hood.

A general overview for how **PCAfold** modules can combine into one workflow is presented below.
Each module's functionalities can also be used as a standalone tool for performing a specific task.

.. image:: images/PCA-Python-overview.png
  :width: 500
  :align: center

--------------------------------------------------------------------------------

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   user/getting-started
   user/PCA
   user/clustering
   user/cluster-biased-pca
   user/manifold-dimensionality
   user/train-test-select

.. toctree::
   :maxdepth: 2
   :caption: Tutorials & Demos

   tutorials/demo-pca
   tutorials/demo-clustering
   tutorials/demo-cluster-biased-pca
   tutorials/demo-manifold-dimensionality
   tutorials/demo-train-test-selection
