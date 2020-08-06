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

  test.test_clustering()
  test.test_sampling()

Plotting
--------

Some functions within **PCAfold** result in plot outputs. Global styles for the
plots are set using the ``styles.py`` file. This file can be updated with new
settings that will be seen globally by **PCAfold** modules.
