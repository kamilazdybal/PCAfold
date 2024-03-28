.. module:: reconstruction

#################
Reconstruction
#################

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
