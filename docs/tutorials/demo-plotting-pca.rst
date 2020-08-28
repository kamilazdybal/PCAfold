.. note:: This tutorial was generated from a Jupyter notebook that can be
          accessed `here <https://gitlab.multiscale.utah.edu/common/PCAfold/-/blob/regression/docs/tutorials/demo-pca.ipynb>`_.

####################
Plotting PCA results
####################

In this tutorial we present plotting functionalities of the ``reduction`` module,
specifically some functions that aid in visualizing PCA results.

As an example, we will use a data set representing combustion of syngas
(CO/H2 mixture) in air generated from steady laminar flamelet model.
This data set has 11 variables and 50,000 observations. To load the data set
from the tutorials directory:

.. code:: python

  from PCAfold import PCA
  from PCAfold import reduction
  import numpy as np
  import pandas as pd

  X = pd.read_csv('data-state-space.csv', sep = ',', header=None).to_numpy()
  X_names = ['$T$', '$H_2$', '$O_2$', '$O$', '$OH$', '$H_2O$', '$H$', '$HO_2$', '$CO$', '$CO_2$', '$HCO$']

We generate three PCA objects corresponding to three scaling criteria:

.. code:: python

  pca_X_Auto = PCA(X, scaling='auto', n_components=2)
  pca_X_Range = PCA(X, scaling='range', n_components=2)
  pca_X_Vast = PCA(X, scaling='vast', n_components=2)

and we will plot PCA results from the generated objects.

--------------------------------------------------------------------------------

*******************
Single eigenvectors
*******************

Weights of a single eigenvector can be plotted using ``reduction.plot_eigenvectors`` function.
Note that multiple eigenvectors can be passed as an input and this function will
generate as many plots as there are eigenvectors supplied.

Below is an example of plotting just the first eigenvector:

.. code::

  plt = reduction.plot_eigenvectors(pca_X_Auto.Q[:,0], variable_names=X_names, plot_absolute=False, title=None, save_filename=None)

To plot all eigenvectors resulting from a single ``PCA`` class object:

.. code::

  plts = reduction.plot_eigenvectors(pca_X_Auto.Q, variable_names=X_names, plot_absolute=False, title=None, save_filename=None)

Plotting example
^^^^^^^^^^^^^^^^

Two weight normalizations are available:

- No normalization. To use this variant set ``plot_absolute=False``. Example can be seen below:

.. image:: ../images/plotting-pca-eigenvector-1.png
    :width: 500
    :align: center

- Absolute values. To use this variant set ``plot_absolute=True``. Example can be seen below:

.. image:: ../images/plotting-pca-absolute-eigenvector-1.png
    :width: 500
    :align: center

***********************
Eigenvectors comparison
***********************

Eigenvectors resulting from, for instance, different ``PCA`` class objects can
be compared on a single plot using ``reduction.plot_eigenvectors_comparison`` function.

.. code::

  plt = reduction.plot_eigenvectors_comparison((pca_X_Auto.Q[:,0], pca_X_Range.Q[:,0], pca_X_Vast.Q[:,0]), legend_labels=['Auto', 'Range', 'Vast'], variable_names=X_names, plot_absolute=False, color_map='coolwarm', title=None, save_filename=None)

Plotting example
^^^^^^^^^^^^^^^^

Two weight normalizations are available:

- No normalization. To use this variant set ``plot_absolute=False``. Example can be seen below:

.. image:: ../images/plotting-pca-eigenvectors-comparison.png
    :width: 500
    :align: center

- Absolute values. To use this variant set ``plot_absolute=True``. Example can be seen below:

.. image:: ../images/plotting-pca-eigenvectors-comparison-absolute.png
    :width: 500
    :align: center

***********************
Eigenvalue distribution
***********************

Eigenvalue distribution can be plotted using ``reduction.plot_eigenvalue_distribution``.

.. code::

  plt = reduction.plot_eigenvalue_distribution(pca_X_Auto.L, normalized=False, title=None, save_filename=None)

Plotting example
^^^^^^^^^^^^^^^^

Two eigenvalue normalizations are available:

- No normalization. To use this variant set ``normalized=False``. Example can be seen below:

.. image:: ../images/plotting-pca-eigenvalue-distribution.png
    :width: 500
    :align: center

- Normalized to 1. To use this variant set ``normalized=True``. Example can be seen below:

.. image:: ../images/plotting-pca-eigenvalue-distribution-normalized.png
    :width: 500
    :align: center

**********************************
Eigenvalue distribution comparison
**********************************

Eigenvalues resulting from, for instance, different ``PCA`` class objects can
be compared on a single plot using ``reduction.plot_eigenvalues_comparison`` function.

.. code::

  plt = reduction.plot_eigenvalue_distribution_comparison((pca_X_Auto.L, pca_X_Range.L, pca_X_Vast.L), legend_labels=['Auto', 'Range', 'Vast'], normalized=True, color_map='coolwarm', title=None, save_filename=None)

Plotting example
^^^^^^^^^^^^^^^^

Two eigenvalue normalizations are available:

- No normalization. To use this variant set ``normalized=False``. Example can be seen below:

.. image:: ../images/plotting-pca-eigenvalue-distribution-comparison.png
    :width: 500
    :align: center

- Normalized to 1. To use this variant set ``normalized=True``. Example can be seen below:

.. image:: ../images/plotting-pca-eigenvalue-distribution-comparison-normalized.png
    :width: 500
    :align: center

************************
Two-dimensional manifold
************************

Two-dimensional manifold resulting from performing PCA transformation can be
plotted using ``reduction.plot_2d_manifold`` function. We first calculate
the Principal Components by transforming the original data set to the new basis:

.. code::

  principal_components = pca_X_Vast.transform(X)

and we plot the resulting manifold:

.. code::

  plt = reduction.plot_2d_manifold(principal_components, color_variable='k', x_label='$\mathbf{Z_1}$', y_label='$\mathbf{Z_2}$', colorbar_label=None, title=None, save_filename=None)

Plotting example
^^^^^^^^^^^^^^^^

Example of a plot:

.. image:: ../images/plotting-pca-2d-manifold-black.png
    :width: 400
    :align: center

By setting ``color_variable=X[:,0]`` parameter, the manifold can be additionally
colored by the first variable in the data set:

.. image:: ../images/plotting-pca-2d-manifold.png
    :width: 500
    :align: center
