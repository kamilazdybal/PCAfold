.. note:: This tutorial was generated from a Jupyter notebook that can be
          accessed `here <https://mybinder.org/v2/git/https%3A%2F%2Fgitlab.multiscale.utah.edu%2Fcommon%2FPCAfold/master?filepath=docs%2Ftutorials%2Fdemo-handling-source-terms.ipynb>`_.

#################################
Handling source terms
#################################

This tutorial can be of interest to researchers working with reactive flows data sets. We present how source terms of the original state variables can be handled using **PCAfold** software. Specifically, **PCAfold** functionalities accommodate treatment of sources of principal components (PCs) which can be valuable for implementing PC-transport approaches such as proposed in  :cite:`Sutherland2009`.

*********
Theory
*********

The methodology for the standard PC-transport approach was first proposed
in :cite:`Sutherland2009`. As an illustrative example, PC-transport
equations adequate to a 0D chemical reactor are presented below.
The reader is referred to :cite:`Biglari2015`, :cite:`Echekki2015` for treatment of the full PC-transport equations including diffusion.

We assume that the data set containing original state-space variables is:

.. math::

  \mathbf{X} = [T, Y_1, Y_2, \dots, Y_{N_s-1}]

where :math:`T` is temperature and :math:`Y_i` is a mass fraction of species
:math:`i`. :math:`N_s` is the total number of chemical species. The corresponding
source terms of the original state-space variables are:

.. math::

  \mathbf{S_X} = \Big[-\frac{1}{\rho c_p} \sum_{i=1}^{N_s} ( \omega_i h_i ), \frac{\omega_1}{\rho}, \frac{\omega_2}{\rho}, \dots, \frac{\omega_{N_s-1}}{\rho} \Big]

where :math:`\rho` is the density of the mixture and :math:`c_p` is the specific heat capacity of the mixture,
:math:`\omega_i` is the net mass production rate of species :math:`i` and :math:`h_i` is the enthalpy of species :math:`i`.

For a 0D-system, we can write the evolution equation as:

.. math::

  \frac{d \mathbf{X}}{dt} = \mathbf{S_X}

This equation can be instead written in the space of principal components by applying
a linear operator :math:`\mathbf{A}` identified by PCA. We can also account for
centering and scaling the original data set :math:`\mathbf{X}` using centers
:math:`\mathbf{C}` and scales :math:`\mathbf{D}`:

.. math::

  \frac{d \Big( \frac{\mathbf{X} - \mathbf{C}}{\mathbf{D}} \Big) \mathbf{A}}{dt} = \frac{\mathbf{S_X}}{\mathbf{D}}\mathbf{A}

It is worth noting that when the original data set is centered and scaled,
the corresponding source terms should only be scaled and not centered, since:

.. math::

  \frac{d \frac{\mathbf{C}}{\mathbf{D}} \mathbf{A}}{dt} = 0

for constant :math:`\mathbf{C}`, :math:`\mathbf{D}` and :math:`\mathbf{A}`.

We finally obtain the 0D PC-transport equation where the evolved variables
are principal components instead of the original state-space variables:

.. math::

  \frac{d \mathbf{Z}}{dt} = \mathbf{S_{Z}}

where :math:`\mathbf{Z} = \Big( \frac{\mathbf{X} - \mathbf{C}}{\mathbf{D}} \Big) \mathbf{A}`
and :math:`\mathbf{S_{Z}} = \frac{\mathbf{S_X}}{\mathbf{D}}\mathbf{A}`.

**********************
Code implementation
**********************

We import the necessary modules:

.. code::

    from PCAfold import PCA
    import numpy as np

A data set representing combustion of syngas in air generated from steady laminar
flamelet model using *Spitfire* software :cite:`Hansen2020` and a chemical
mechanism by Hawkes et al. :cite:`Hawkes2007` is used as a demo data set.

We begin by importing the data set composed of the original state space variables
:math:`\mathbf{X}` and the corresponding source terms :math:`\mathbf{S_X}`:

.. code::

  X = np.genfromtxt('data-state-space.csv', delimiter=',')
  S_X = np.genfromtxt('data-state-space-sources.csv', delimiter=',')

We perform PCA on the original data:

.. code::

  pca_X = PCA(X, scaling='auto', n_components=2)

We transform the original data set  to the newly identified basis and
compute the principal components (PCs) :math:`\mathbf{Z}`:

.. code::

  Z = pca_X.transform(X, nocenter=False)

Transform the source terms to the newly identified basis and compute the sources
of principal components :math:`\mathbf{S_Z}`:

.. code::

  S_Z = pca_X.transform(S_X, nocenter=True)

Note that we set the flag ``nocenter=True`` which is a specific setting that
should be applied when transforming source terms.
With that setting, only scales :math:`\mathbf{D}` will be applied when transforming :math:`\mathbf{S_X}`
to the new basis defined by :math:`\mathbf{A}` and thus the transformation will be consistent with the discussion presented
in the previous section.

--------------------------------------------------------------------------------

**********************
Bibliography
**********************

.. bibliography:: demo-handling-source-terms.bib
  :labelprefix: T
