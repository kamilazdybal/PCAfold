.. module:: reconstruction

###############################
Module: reconstruction
###############################

The ``reconstruction`` module contains functions for reconstructing quantities of interest (QoIs) from the
low-dimensional data representations and functionalities to asses the quality of that reconstruction.

.. note:: The format for the user-supplied input data matrix
  :math:`\mathbf{X} \in \mathbb{R}^{N \times Q}`, common to all modules, is that
  :math:`N` observations are stored in rows and :math:`Q` variables are stored
  in columns. Since typically :math:`N \gg Q`, the initial dimensionality of the
  data set is determined by the number of variables, :math:`Q`.

  .. math::

    \mathbf{X} =
    \begin{bmatrix}
    \vdots & \vdots & & \vdots \\
    X_1 & X_2 & \dots & X_{Q} \\
    \vdots & \vdots & & \vdots \\
    \end{bmatrix}

  The general agreement throughout this documentation is that :math:`i` will
  index observations and :math:`j` will index variables.

  The representation of the user-supplied data matrix in **PCAfold**
  is the input parameter ``X``, which should be of type ``numpy.ndarray``
  and of size ``(n_observations,n_variables)``.

--------------------------------------------------------------------------------

******************************************************
Tools for reconstructing quantities of interest (QoIs)
******************************************************

Class ``ANN``
=============

.. autoclass:: PCAfold.reconstruction.ANN
.. autofunction:: PCAfold.reconstruction.ANN.summary
.. autofunction:: PCAfold.reconstruction.ANN.train
.. autofunction:: PCAfold.reconstruction.ANN.predict
.. autofunction:: PCAfold.reconstruction.ANN.print_weights_and_biases_init
.. autofunction:: PCAfold.reconstruction.ANN.print_weights_and_biases_trained
.. autofunction:: PCAfold.reconstruction.ANN.plot_losses

--------------------------------------------------------------------------------

Class ``PartitionOfUnityNetwork``
==================================

.. autoclass:: PCAfold.reconstruction.PartitionOfUnityNetwork
.. autofunction:: PCAfold.reconstruction.PartitionOfUnityNetwork.load_data_from_file
.. autofunction:: PCAfold.reconstruction.PartitionOfUnityNetwork.load_from_file
.. autofunction:: PCAfold.reconstruction.PartitionOfUnityNetwork.load_data_from_txt
.. autofunction:: PCAfold.reconstruction.PartitionOfUnityNetwork.write_data_to_file
.. autofunction:: PCAfold.reconstruction.PartitionOfUnityNetwork.write_data_to_txt
.. autofunction:: PCAfold.reconstruction.PartitionOfUnityNetwork.build_training_graph
.. autofunction:: PCAfold.reconstruction.PartitionOfUnityNetwork.update_lr
.. autofunction:: PCAfold.reconstruction.PartitionOfUnityNetwork.update_l2reg
.. autofunction:: PCAfold.reconstruction.PartitionOfUnityNetwork.lstsq
.. autofunction:: PCAfold.reconstruction.PartitionOfUnityNetwork.train
.. autofunction:: PCAfold.reconstruction.PartitionOfUnityNetwork.__call__
.. autofunction:: PCAfold.reconstruction.PartitionOfUnityNetwork.derivatives
.. autofunction:: PCAfold.reconstruction.PartitionOfUnityNetwork.partition_prenorm

.. autofunction:: PCAfold.reconstruction.init_uniform_partitions

--------------------------------------------------------------------------------

***********************
Regression assessment
***********************

Class ``RegressionAssessment``
================================================

.. autoclass:: PCAfold.reconstruction.RegressionAssessment
.. autofunction:: PCAfold.reconstruction.RegressionAssessment.print_metrics
.. autofunction:: PCAfold.reconstruction.RegressionAssessment.print_stratified_metrics

Regression quality metrics
================================================

.. autofunction:: PCAfold.reconstruction.coefficient_of_determination
.. autofunction:: PCAfold.reconstruction.stratified_coefficient_of_determination

.. autofunction:: PCAfold.reconstruction.mean_absolute_error
.. autofunction:: PCAfold.reconstruction.stratified_mean_absolute_error

.. autofunction:: PCAfold.reconstruction.max_absolute_error
.. autofunction:: PCAfold.reconstruction.stratified_max_absolute_error

.. autofunction:: PCAfold.reconstruction.mean_squared_error
.. autofunction:: PCAfold.reconstruction.stratified_mean_squared_error

.. autofunction:: PCAfold.reconstruction.mean_squared_logarithmic_error
.. autofunction:: PCAfold.reconstruction.stratified_mean_squared_logarithmic_error

.. autofunction:: PCAfold.reconstruction.root_mean_squared_error
.. autofunction:: PCAfold.reconstruction.stratified_root_mean_squared_error

.. autofunction:: PCAfold.reconstruction.normalized_root_mean_squared_error
.. autofunction:: PCAfold.reconstruction.stratified_normalized_root_mean_squared_error

.. autofunction:: PCAfold.reconstruction.turning_points

.. autofunction:: PCAfold.reconstruction.good_estimate

.. autofunction:: PCAfold.reconstruction.good_direction_estimate

.. autofunction:: PCAfold.reconstruction.generate_tex_table

--------------------------------------------------------------------------------

******************
Plotting functions
******************

.. autofunction:: PCAfold.reconstruction.plot_2d_regression
.. autofunction:: PCAfold.reconstruction.plot_2d_regression_scalar_field
.. autofunction:: PCAfold.reconstruction.plot_2d_regression_streamplot
.. autofunction:: PCAfold.reconstruction.plot_3d_regression
.. autofunction:: PCAfold.reconstruction.plot_stratified_metric
