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

``ANN.summary``
===============

.. autofunction:: PCAfold.reconstruction.ANN.summary

``ANN.train``
=============

.. autofunction:: PCAfold.reconstruction.ANN.train

``ANN.predict``
===============

.. autofunction:: PCAfold.reconstruction.ANN.predict

``ANN.print_weights_and_biases_init``
=====================================

.. autofunction:: PCAfold.reconstruction.ANN.print_weights_and_biases_init

``ANN.print_weights_and_biases_trained``
========================================

.. autofunction:: PCAfold.reconstruction.ANN.print_weights_and_biases_trained

``ANN.plot_losses``
===================

.. autofunction:: PCAfold.reconstruction.ANN.plot_losses

--------------------------------------------------------------------------------

Class ``PartitionOfUnityNetwork``
==================================

.. autoclass:: PCAfold.reconstruction.PartitionOfUnityNetwork

``PartitionOfUnityNetwork.load_data_from_file``
================================================

.. autofunction:: PCAfold.reconstruction.PartitionOfUnityNetwork.load_data_from_file

``PartitionOfUnityNetwork.load_from_file``
===========================================

.. autofunction:: PCAfold.reconstruction.PartitionOfUnityNetwork.load_from_file

``PartitionOfUnityNetwork.load_data_from_txt``
===============================================

.. autofunction:: PCAfold.reconstruction.PartitionOfUnityNetwork.load_data_from_txt

``PartitionOfUnityNetwork.write_data_to_file``
===============================================

.. autofunction:: PCAfold.reconstruction.PartitionOfUnityNetwork.write_data_to_file

``PartitionOfUnityNetwork.write_data_to_txt``
===============================================

.. autofunction:: PCAfold.reconstruction.PartitionOfUnityNetwork.write_data_to_txt

``PartitionOfUnityNetwork.build_training_graph``
==================================================

.. autofunction:: PCAfold.reconstruction.PartitionOfUnityNetwork.build_training_graph

``PartitionOfUnityNetwork.update_lr``
=======================================

.. autofunction:: PCAfold.reconstruction.PartitionOfUnityNetwork.update_lr

``PartitionOfUnityNetwork.update_l2reg``
==========================================

.. autofunction:: PCAfold.reconstruction.PartitionOfUnityNetwork.update_l2reg

``PartitionOfUnityNetwork.lstsq``
==================================

.. autofunction:: PCAfold.reconstruction.PartitionOfUnityNetwork.lstsq

``PartitionOfUnityNetwork.train``
====================================

.. autofunction:: PCAfold.reconstruction.PartitionOfUnityNetwork.train

``PartitionOfUnityNetwork.__call__``
======================================

.. autofunction:: PCAfold.reconstruction.PartitionOfUnityNetwork.__call__

``PartitionOfUnityNetwork.derivatives``
========================================

.. autofunction:: PCAfold.reconstruction.PartitionOfUnityNetwork.derivatives

``PartitionOfUnityNetwork.partition_prenorm``
===============================================

.. autofunction:: PCAfold.reconstruction.PartitionOfUnityNetwork.partition_prenorm

``init_uniform_partitions``
==============================

.. autofunction:: PCAfold.reconstruction.init_uniform_partitions

--------------------------------------------------------------------------------

***********************
Regression assessment
***********************

Class ``RegressionAssessment``
================================================

.. autoclass:: PCAfold.reconstruction.RegressionAssessment

``RegressionAssessment.print_metrics``
================================================

.. autofunction:: PCAfold.reconstruction.RegressionAssessment.print_metrics

``RegressionAssessment.print_stratified_metrics``
=================================================

.. autofunction:: PCAfold.reconstruction.RegressionAssessment.print_stratified_metrics

``coefficient_of_determination``
================================

.. autofunction:: PCAfold.reconstruction.coefficient_of_determination

``stratified_coefficient_of_determination``
===========================================

.. autofunction:: PCAfold.reconstruction.stratified_coefficient_of_determination

``mean_absolute_error``
=======================

.. autofunction:: PCAfold.reconstruction.mean_absolute_error

``stratified_mean_absolute_error``
==================================

.. autofunction:: PCAfold.reconstruction.stratified_mean_absolute_error

``max_absolute_error``
=======================

.. autofunction:: PCAfold.reconstruction.max_absolute_error

``mean_squared_error``
======================

.. autofunction:: PCAfold.reconstruction.mean_squared_error

``stratified_mean_squared_error``
=================================

.. autofunction:: PCAfold.reconstruction.stratified_mean_squared_error

``mean_squared_logarithmic_error``
==================================

.. autofunction:: PCAfold.reconstruction.mean_squared_logarithmic_error

``stratified_mean_squared_logarithmic_error``
=============================================

.. autofunction:: PCAfold.reconstruction.stratified_mean_squared_logarithmic_error

``root_mean_squared_error``
===========================

.. autofunction:: PCAfold.reconstruction.root_mean_squared_error

``stratified_root_mean_squared_error``
======================================

.. autofunction:: PCAfold.reconstruction.stratified_root_mean_squared_error

``normalized_root_mean_squared_error``
======================================

.. autofunction:: PCAfold.reconstruction.normalized_root_mean_squared_error

``stratified_normalized_root_mean_squared_error``
=================================================

.. autofunction:: PCAfold.reconstruction.stratified_normalized_root_mean_squared_error

``turning_points``
================================================

.. autofunction:: PCAfold.reconstruction.turning_points

``good_estimate``
================================================

.. autofunction:: PCAfold.reconstruction.good_estimate

``good_direction_estimate``
================================================

.. autofunction:: PCAfold.reconstruction.good_direction_estimate

``generate_tex_table``
================================================

.. autofunction:: PCAfold.reconstruction.generate_tex_table

--------------------------------------------------------------------------------

******************
Plotting functions
******************

``plot_2d_regression``
======================

.. autofunction:: PCAfold.reconstruction.plot_2d_regression

``plot_2d_regression_scalar_field``
===================================

.. autofunction:: PCAfold.reconstruction.plot_2d_regression_scalar_field

``plot_2d_regression_streamplot``
=================================

.. autofunction:: PCAfold.reconstruction.plot_2d_regression_streamplot

``plot_3d_regression``
======================

.. autofunction:: PCAfold.reconstruction.plot_3d_regression

``plot_stratified_metric``
==========================

.. autofunction:: PCAfold.reconstruction.plot_stratified_metric

--------------------------------------------------------------------------------

************
Bibliography
************

.. bibliography::
  :cited:
  :labelprefix: E
