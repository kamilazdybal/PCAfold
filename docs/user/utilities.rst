.. module:: utilities

#################
Utilities
#################

The ``utilities`` module contains functions optimizing manifold topology in an automated way.

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

*******************************************
Tools for optimizing manifold topology
*******************************************

Class ``QoIAwareProjection``
============================

.. autoclass:: PCAfold.utilities.QoIAwareProjection
.. autofunction:: PCAfold.utilities.QoIAwareProjection.summary
.. autofunction:: PCAfold.utilities.QoIAwareProjection.train
.. autofunction:: PCAfold.utilities.QoIAwareProjection.print_weights_and_biases_init
.. autofunction:: PCAfold.utilities.QoIAwareProjection.print_weights_and_biases_trained
.. autofunction:: PCAfold.utilities.QoIAwareProjection.get_best_basis
.. autofunction:: PCAfold.utilities.QoIAwareProjection.plot_losses

Class ``QoIAwareProjectionPOUnet``
===================================

.. autoclass:: PCAfold.utilities.QoIAwareProjectionPOUnet
.. autofunction:: PCAfold.utilities.QoIAwareProjectionPOUnet.projection
.. autofunction:: PCAfold.utilities.QoIAwareProjectionPOUnet.tf_projection
.. autofunction:: PCAfold.utilities.QoIAwareProjectionPOUnet.update_lr
.. autofunction:: PCAfold.utilities.QoIAwareProjectionPOUnet.update_l2reg
.. autofunction:: PCAfold.utilities.QoIAwareProjectionPOUnet.build_training_graph
.. autofunction:: PCAfold.utilities.QoIAwareProjectionPOUnet.train
.. autofunction:: PCAfold.utilities.QoIAwareProjectionPOUnet.__call__
.. autofunction:: PCAfold.utilities.QoIAwareProjectionPOUnet.write_data_to_file
.. autofunction:: PCAfold.utilities.QoIAwareProjectionPOUnet.load_data_from_file
.. autofunction:: PCAfold.utilities.QoIAwareProjectionPOUnet.load_from_file

Manifold-informed feature selection
=====================================

.. autofunction:: PCAfold.utilities.manifold_informed_forward_variable_addition
.. autofunction:: PCAfold.utilities.manifold_informed_backward_variable_elimination
