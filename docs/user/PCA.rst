.. module:: pca_impl

Principal Component Analysis
============================

``pca_impl.py`` module contains functions for performing Principal Component Analysis :cite:`Jolliffe2002`.

It is assumed that the raw data set ``X`` on which PCA is performed has dimensions ``(n_observations, n_variables)``:

.. image:: ../images/data-set.png
  :width: 300
  :align: center

--------------------------------------------------------------------------------

Functions
---------

Center and scale
^^^^^^^^^^^^^^^^

.. autofunction:: PCAfold.pca_impl.center_scale

Uncenter and unscale
^^^^^^^^^^^^^^^^^^^^

.. autofunction:: PCAfold.pca_impl.inv_center_scale

Remove constant variables from a data set
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: PCAfold.pca_impl.remove_constant_vars

--------------------------------------------------------------------------------

Class ``PCA``
-------------

.. autoclass:: PCAfold.pca_impl.PCA

Functions within ``PCA`` class
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: PCAfold.pca_impl.PCA.x2eta

.. autofunction:: PCAfold.pca_impl.PCA.eta2x

.. autofunction:: PCAfold.pca_impl.PCA.calculate_r2

.. autofunction:: PCAfold.pca_impl.PCA.data_consistency_check

.. autofunction:: PCAfold.pca_impl.PCA.convergence

.. autofunction:: PCAfold.pca_impl.PCA.eig_bar_plot_maker

.. autofunction:: PCAfold.pca_impl.PCA.plot_convergence

.. autofunction:: PCAfold.pca_impl.PCA.principal_variables

.. autofunction:: PCAfold.pca_impl.PCA.r2converge

.. autofunction:: PCAfold.pca_impl.PCA.write_file_for_cpp

.. autofunction:: PCAfold.pca_impl.PCA.set_retained_eigenvalues

.. autofunction:: PCAfold.pca_impl.PCA.u_scores

.. autofunction:: PCAfold.pca_impl.PCA.w_scores

--------------------------------------------------------------------------------

Bibliography
------------

.. bibliography:: PCA.bib
