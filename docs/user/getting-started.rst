Getting started
===============

Installation
------------

Clone the ``PCA-python`` repository and move into the ``PCA-python`` directory created:

.. code::

  git clone http://gitlab.multiscale.utah.edu/common/PCA-python.git
  cd PCA-python

Run the ``setup.py`` script as below to complete the installation:

.. code::

  python3.5 setup.py install

Dependencies
^^^^^^^^^^^^

``PCAfold`` requires ``python3`` (developed with ``python3.5``) and the following packages:

- ``copy``
- ``matplotlib``
- ``numpy``
- ``random``
- ``scipy``
- ``sklearn``

Testing
^^^^^^^

To run regression tests of all modules execute:

.. code:: python

  from PCAfold import test

  test.test()

You can also test each module separately:

.. code:: python

  from PCAfold import test

  test.test_preprocess()
  test.test_reduction()
  test.test_analysis()

Plotting
--------

Some functions within **PCAfold** result in plot outputs. Global styles for the
plots are set using the ``styles.py`` file. This file can be updated with new
settings that will be seen globally by **PCAfold** modules. Re-build the project
after changing ``styles.py`` file:

.. code::

  python3.5 setup.py install

Workflows
---------

Below we discuss several popular workflows that can be achieved using
functionalities of **PCAfold**.

Data manipulation
^^^^^^^^^^^^^^^^^

Data manipulation such as centering, scaling or outlier detection and removal
can be achieved using ``preprocess`` module.

Data clustering
^^^^^^^^^^^^^^^

Data clustering can be achieved using ``preprocess`` module. This can be
useful for data analysis or feature detection.

Data sampling
^^^^^^^^^^^^^

Data sampling can be achieved using ``preprocess`` module only. Possible
use-case for sampling data sets could be to split data sets into train and test
samples for other Machine Learning algorithms, as well as sample unbalanced
data sets.

Global PCA
^^^^^^^^^^

Global PCA can be performed using ``reduction`` module. Typically, you might
want to first pre-process the data set which can be achieved using
``preprocess`` module.

Local PCA
^^^^^^^^^

Local PCA can be performed using ``reduction`` module. Typically, you might
want to first pre-process the data set which can be achieved using
``preprocess`` module. Then, data set needs to be clustered with a technique of
choice which can be achieved with any clustering technique from the
``preprocess`` module or using any algorithm outside of **PCAfold**.

PCA on sampled data sets
^^^^^^^^^^^^^^^^^^^^^^^^

PCA on sampled data sets can be performed using ``reduction`` module.
Typically, you might want to first pre-process the data set which can be
achieved using ``preprocess`` module. Then, data set can be clustered and
sampled using ``preprocess`` module.

Assessing manifold quality
^^^^^^^^^^^^^^^^^^^^^^^^^^

Once you have a low-dimensional manifold, the quality of the manifold can be
assessed using ``analysis`` module.
