.. note:: This tutorial was generated from a Jupyter notebook that can be
          accessed `here <https://gitlab.multiscale.utah.edu/common/PCAfold/-/blob/regression/docs/tutorials/demo-data-manipulation.ipynb>`_.

#################
Data manipulation
#################

In this tutorial we present data manipulation functionalities of the ``preprocess`` module.

To import the module:

.. code:: python

  from PCAfold import preprocess

--------------------------------------------------------------------------------

******************************
Multivariate outlier detection
******************************

We first generate a synthetic data set with artificially appended outliers.
This data set, with outliers visible as a cloud in the top right corner, can be seen below:

.. image:: ../images/data-manipulation-initial-data.png
  :width: 350
  :align: center

We will first detect outliers with ``'MULTIVARIATE TRIMMING'`` method and we
will demonstrate the effect of setting two levels of ``trimming_fraction``.

We first set ``trimming_fraction=0.6``:

.. code:: python

  (idx_outliers_removed, idx_outliers) = preprocess.outlier_detection(X, scaling='auto', detection_method='MULTIVARIATE TRIMMING', trimming_fraction=0.6, n_iterations=0, verbose=True)

With ``verbose=True`` we will see some detailed information on outliers detected:

.. code-block:: text

  Number of observations classified as outliers: 20

We can visualize the observations that were classified as outliers using the
``preprocess.plot_2d_clustering``, assuming that the cluster :math:`k_0` (blue) will be
observations with removed outliers and cluster :math:`k_1` (red) will be the detected outliers.

We first create a dummy ``idx`` vector of cluster classifications based on
``idx_outliers`` obtained. This can for instance be done in the following way:

.. code:: python

  n_observations = N + N_outliers
  idx_new = np.zeros((n_observations,))
  for i in range(0, n_observations):
    if i in idx_outliers:
        idx_new[i] = 1

The result of this detection can be seen below:

.. image:: ../images/data-manipulation-outliers-multivariate-trimming-60.png
  :width: 350
  :align: center

We then set the ``trimming_fraction=0.3`` which will capture outliers earlier (at smaller
Mahalanobis distances from the variables' centroids).

.. code:: python

  (idx_outliers_removed, idx_outliers) = preprocess.outlier_detection(X, scaling='auto', detection_method='MULTIVARIATE TRIMMING', trimming_fraction=0.3, n_iterations=0, verbose=True)

.. code-block:: text

  Number of observations classified as outliers: 180

The result of this detection can be seen below:

.. image:: ../images/data-manipulation-outliers-multivariate-trimming-30.png
  :width: 350
  :align: center

It can be seen that the algorithm started to pick up outlier observations at the perimeter of
the original data set.
