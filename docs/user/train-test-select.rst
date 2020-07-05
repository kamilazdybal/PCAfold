.. module:: PCA.train_test_select

Train and test data selection
=============================

``train_test_select.py`` module contains functions for splitting data sets into train and test data for use in machine learning algorithms.
Apart from random splitting that can be achieved with the commonly used ``sklearn.model_selection.train_test_split``, new methods are implemented here that allow for purposive selection, such as drawing samples at a certain amount from local clusters.
The general idea is to divide the entire data set ``X`` into train and test samples as presented below:

.. image:: ../images/tts-train-test-select.png
  :width: 350
  :align: center

All splitting functions in this module return a tuple of two variables: ``(idx_train, idx_test)``. These variables contain indices of observations that went into train data and test data respectively. In your model learning algorithm you can then get the train and test observations, for instance in the following way:

.. code:: python

  Input_train = Input[idx_train,:]
  Output_train = Output[idx_train,:]

  Input_test = Input[idx_test,:]
  Output_test = Output[idx_test,:]

All functions are equipped with ``verbose=False`` parameter. If it is set to ``True`` some additional information on train and test selection is printed.

.. note:: It is assumed that the first cluster has index ``0`` within all input ``idx`` vectors. When verbose information is printed with ``verbose=True`` during function execution or on the plots the cluster numeration starts with ``1``.

Functions
---------

Select fixed number
^^^^^^^^^^^^^^^^^^^

.. autofunction:: PCA.train_test_select.train_test_split_fixed_number_from_idx

Train data
""""""""""

Train data is always selected as an equal number of samples from local clusters but no more than 50% of the cluster's samples will be selected (see: 50% bar). This is to avoid oversampling small clusters which might in turn result in too little test data. The number of samples ``n_of_samples`` is calculated based on a percentage ``perc`` provided:

.. math::

    \verb|n_of_samples| = \verb|int| \Big( \frac{\verb|perc| \cdot \verb|n_obs|}{\verb|k| \cdot 100} \Big)

where ``n_obs`` is the total number of samples in a data set and ``k`` is the number of clusters.

Test data
"""""""""

Depending on the option selected, test data will be created differently, either as all
remaining samples that were not included in train data or as a subset of those.
You can select the option by setting the ``test_selection_option`` parameter which is 1 by default.
The scheme below presents graphically how train and test data can be selected using ``test_selection_option`` parameter:

.. image:: ../images/tts-test-selection-option.png
  :width: 700
  :align: center

Select fixed percentage
^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: PCA.train_test_select.train_test_split_percentage_from_idx

Select manually
^^^^^^^^^^^^^^^

.. autofunction:: PCA.train_test_select.train_test_split_manual_from_idx

Select at random
^^^^^^^^^^^^^^^^

.. autofunction:: PCA.train_test_select.train_test_split_random
