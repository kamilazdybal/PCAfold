PCAfold
============

*Low-dimensional PCA-derived manifolds and everything in between!*

--------------------------------------------------------------------------------

**PCAfold** is a Python software for generating, improving and analyzing
empirical low-dimensional manifolds obtained via Principal Component Analysis
(PCA). It incorporates advanced data pre-processing tools (including data
clustering and sampling), uses PCA as a dimensionality reduction technique and
introduces analysis module to judge the topology of the obtained low-dimensional
manifolds.

The general overview for using **PCAfold** modules is presented in the diagram
below:

.. image:: images/PCAfold-diagram.png
  :width: 700
  :align: center

Each module's functionalities can also be used as a standalone tool for
performing a specific task and can easily combine with techniques outside of
this software, such as K-Means algorithm or Artificial Neural Networks.
Reach out to the `Getting started <https://pca-python.readthedocs.io/en/latest/user/getting-started.html#workflows>`_
section for possible workflows that can be achieved with **PCAfold**.

--------------------------------------------------------------------------------

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   user/getting-started
   user/data-pre-processing
   user/data-reduction
   user/data-analysis

.. toctree::
   :maxdepth: 2
   :caption: Tutorials & Demos

   tutorials/demo-clustering
   tutorials/demo-train-test-selection
   tutorials/demo-global-vs-local-pca
   tutorials/demo-cluster-biased-pca
   tutorials/demo-manifold-dimensionality
