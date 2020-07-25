.. note:: This tutorial was generated from a Jupyter notebook that can be
          downloaded `here <https://gitlab.multiscale.utah.edu/common/PCA-python/-/blob/regression/docs/tutorials/demo-clustering.ipynb>`_.

Clustering
==========

In this tutorial we present the main functionalities of the ``clustering_data`` module. To import the module:

.. code:: python

  import PCAfold.clustering_data as cl

First, we generate a synthetic two-dimensional data set:

.. code:: python

  var = np.linspace(-1,1,100)
  y = -var**2 + 1

Clustering into bins of a one-dimensional vector will be performed based on ``var``.

Which can be seen below:

.. image:: ../images/tutorial-clustering-original-data-set.png
  :width: 350
  :align: center

Cluster into variable bins
^^^^^^^^^^^^^^^^^^^^^^^^^^

This clustering will divide the data set into equal bins of a one-dimensional variable vector.

.. code:: python

  (idx_variable_bins) = cl.variable_bins(var, 4, verbose=True)

With ``verbose=True`` we will see some detailed information on clustering:

.. code-block:: text

  Border values for each bin are:
  [-1.0, -0.5, 0.0, 0.5, 1.0]

  Bounds for cluster 1:
  	-1.0, -0.5152
  Bounds for cluster 2:
  	-0.4949, -0.0101
  Bounds for cluster 3:
  	0.0101, 0.4949
  Bounds for cluster 4:
  	0.5152, 1.0

The visual result of this clustering can be seen below:

.. image:: ../images/tutorial-clustering-variable-bins-k4.png
  :width: 350
  :align: center

Cluster into pre-defined variable bins
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This clustering will divide the data set into bins of a one-dimensional variable vector whose borders are specified by the user. Let's specify the split values as ``split_values = [-0.6, 0.4, 0.8]``

.. code:: python

  split_values = [-0.6, 0.4, 0.8]
  (idx_predefined_variable_bins) = cl.predefined_variable_bins(var, split_values, verbose=True)

With ``verbose=True`` we will see some detailed information on clustering:

.. code-block:: text

  Border values for bins:
  [-1.0, -0.6, 0.4, 0.8, 1.0]

  Bounds for cluster 1:
  	-1.0, -0.6162
  Bounds for cluster 2:
  	-0.596, 0.3939
  Bounds for cluster 3:
  	0.4141, 0.798
  Bounds for cluster 4:
  	0.8182, 1.0

The visual result of this clustering can be seen below:

.. image:: ../images/tutorial-clustering-predefined-variable-bins-k4.png
  :width: 350
  :align: center
