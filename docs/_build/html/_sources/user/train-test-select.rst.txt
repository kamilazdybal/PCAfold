.. module:: PCA.train_test_select

Train and test data selection
=============================

``train_test_select.py`` module contains functions for splitting data sets into train and test data for use in machine learning algorithms. The general idea is to divide the entire data set into train and test samples as presented below:



However, apart from random splitting that can be achieved with the commonly used ``sklearn.model_selection.train_test_split``, new methods were implemented here that allow for different sampling, such as drawing samples of train data from local clusters.

All splitting functions in this module return a tuple of two variables: ``(idx_train, idx_test)``. These contain indices of observations that went into train data or test data respectively. In your model learning algorithm you can then get the train and test observations, for instance in the following way:

.. code:: python

  Input_train = Input[idx_train,:]
  Output_train = Output[idx_train,:]

  Input_test = Input[idx_test,:]
  Output_test = Output[idx_test,:]


All functions are equipped with ``verbose=False`` parameter. If it is set to `True` some additional information on train and test selection will be printed.

.. note:: It is assumed that cluster numeration inside ``idx`` starts from index ``0``. However, anytime the information about clustering is printed verbally, the cluster numeration starts with index ``1``.
