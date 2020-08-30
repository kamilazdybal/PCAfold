Getting started
===============

Installation
------------

Dependencies
^^^^^^^^^^^^

**PCAfold** requires ``python3.7`` and the following packages:

- ``copy``
- ``Cython``
- ``matplotlib``
- ``numpy``
- ``random``
- ``scipy``

Build from source
^^^^^^^^^^^^^^^^^

Clone the ``PCAfold`` repository and move into the ``PCAfold`` directory created:

.. code-block:: text

  git clone http://gitlab.multiscale.utah.edu/common/PCAfold.git
  cd PCAfold

Run the ``setup.py`` script as below to complete the installation:

.. code-block:: text

  python3.7 setup.py install

You are ready to ``import PCAfold``!

Local documentation build
^^^^^^^^^^^^^^^^^^^^^^^^^

To build the documentation locally, you need ``sphinx`` installed on your machine,
along with the following extensions:

.. code-block:: text

  sphinx.ext.todo
  sphinx.ext.githubpages
  sphinx.ext.autodoc
  sphinx.ext.napoleon
  sphinx.ext.mathjax
  sphinx.ext.autosummary
  sphinxcontrib.bibtex

Then, navigate to ``docs/`` directory and build the documentation:

.. code-block:: text

  sphinx-build -b html . builddir

  make html

Documentation main page ``_build/html/index.html`` can be opened in a web browser.

In MacOS you can open it directly from the terminal:

.. code-block:: text

  open _build/html/index.html

Testing
^^^^^^^

To run regression tests from the base repo directory run:

.. code-block:: text

  python3.7 -m unittest discover

To switch verbose on, use the ``-v`` flag.

All tests should be passing. If any of the tests is failing and you can’t sort
out why, please open an issue on `GitLab <https://gitlab.multiscale.utah.edu/common/PCAfold>`_.

Plotting
--------

Some functions within **PCAfold** result in plot outputs. Global styles for the
plots are set using the ``styles.py`` file. This file can be updated with new
settings that will be seen globally by **PCAfold** modules. Re-build the project
after changing ``styles.py`` file:

.. code-block:: text

  python3.7 setup.py install

All plotting functions return handles to generated plots.

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

Data sampling can be achieved using ``preprocess`` module. Possible
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
