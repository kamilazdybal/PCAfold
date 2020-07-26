.. note:: This tutorial was generated from a Jupyter notebook that can be
          accessed `here <https://gitlab.multiscale.utah.edu/common/PCA-python/-/blob/regression/docs/tutorials/demo-cluster-biased-pca.ipynb>`_.

Cluster-biased PCA
==================

In this tutorial we present the main functionalities of the ``cluster_biased_pca`` module. To import the module:

.. code:: python

  import PCAfold.cluster_biased_pca as cbpca

As an example, we use a data set describing combustion of syngas in air.

--------------------------------------------------------------------------------

Analyze centers movement
^^^^^^^^^^^^^^^^^^^^^^^^

Plotting example
""""""""""""""""

This function will produce a plot that shows the normalized centers and a percentage by which the new centers have moved with respect to the original ones. Example of a plot:

.. image:: ../images/relative_centers_movement.png
    :width: 500
    :align: center

If you do not wish to plot all variables present in a data set, use the ``plot_variables`` list as an input parameter to select indices of variables to plot:

.. image:: ../images/relative_centers_movement_selected_variables.png
    :width: 260
    :align: center

Equilibrate cluster populations iteratively
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^




Analyze eigenvector weights movement
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Plotting example
""""""""""""""""

Three weight normalization variants are available:

- No normalization, the absolute values of the eigenvector weights are plotted. To use this variant set ``normalize=False``. Example can be seen below:

.. image:: ../images/documentation-plot-non-normalized.png
    :width: 500
    :align: center

- Normalizing so that the highest weight is equal to 1 and the smallest weight is between 0 and 1. This is useful for judging the severity of the weight movement. To use this variant set ``normalize=True`` and ``zero_norm=False``. Example can be seen below:

.. image:: ../images/documentation-plot-normalized.png
    :width: 500
    :align: center

- Normalizing so that weights are between 0 and 1. This is useful for judging the movement trends since it will blow up even the smallest changes to the entire range 0-1. To use this variant set ``normalize=True`` and ``zero_norm=True``. Example can be seen below:

.. image:: ../images/documentation-plot-normalized-to-zero.png
    :width: 500
    :align: center

If you do not wish to plot all variables present in a data set, use the ``plot_variables`` list as an input parameter to select indices of variables to plot:

.. image:: ../images/documentation-plot-pre-selected-variables.png
    :width: 280
    :align: center

Analyze eigenvalue distribution
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Plotting example
""""""""""""""""

This function will produce a plot that shows the eigenvalues distribution for the original data set and for different versions of the equilibrated data set. Example of a plot:

.. image:: ../images/documentation-eigenvalues.png
    :width: 500
    :align: center
