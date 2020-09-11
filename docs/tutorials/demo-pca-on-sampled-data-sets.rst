.. note:: This tutorial was generated from a Jupyter notebook that can be
          accessed `here <https://gitlab.multiscale.utah.edu/common/PCAfold/-/blob/master/docs/tutorials/demo-pca-on-sampled-data-sets.ipynb>`_.

PCA on sampled data sets
========================

In this tutorial we present how PCA can be performed on sampled data sets using
various helpful functions from the ``preprocess`` and the ``reduction`` module.

We are first going to present major functionalities for performing and analyzing PCA
on sampled data set using a special case of sampling - by taking equal number
of samples from each cluster. Later, we are going to show a more general way to
perform PCA on data sets that are sampled in any way of choice.

Reach out to the relevant section that you'd like to learn about:

- `Special case of PCA on sampled data sets <https://pcafold.readthedocs.io/en/latest/tutorials/demo-pca-on-sampled-data-sets.html#equilibrate-cluster-populations-iteratively>`_ (for data sets formed by taking equal number of samples from local clusters)
- `Generalization of PCA on sampled data sets <https://pcafold.readthedocs.io/en/latest/tutorials/demo-pca-on-sampled-data-sets.html#generalization-of-pca-on-sampled-data-set>`_

To import the necessary modules and functionalities:

.. code:: python

  from PCAfold import preprocess
  from PCAfold import reduction
  from PCAfold import DataSampler
  from PCAfold import PCA
  import numpy as np

A general overview for performing PCA on sampled data set is presented below:

.. image:: ../images/PCA-on-sampled-data-set.png
  :width: 700

The main goal is to inform PCA of some of the characteristics of the sampled
data set :math:`\mathbf{X_r}`. There are several ways in which that information
can be incorporated within PCA transformation and they can be controlled using a selected
`biasing option <https://pcafold.readthedocs.io/en/latest/user/data-reduction.html#id14>`_
and setting the ``biasing_option`` input parameter whenever needed. In this
example, we choose ``biasing_option=2``. It is understood that PCA performed on a
sampled data set is *biased* in some way, since that data set contains different
proportions of features in terms of sample density than their original
contribution within the full :math:`\mathbf{X}`.
Those features can be identified using any clustering technique of choice.

As an example, we will use a data set representing combustion of syngas
(CO/H2 mixture) in air generated from steady laminar flamelet model using chemical
mechanism by Hawkes et al. :cite:`Hawkes2007`.
This data set has 11 variables and 50,000 observations. The data set was generated
using *Spitfire* software :cite:`Hansen2020`. To load the data set
from the tutorials directory:

.. code:: python

  # Original variables:
  state_space = np.genfromtxt('data-state-space.csv', delimiter=',')

  # Corresponding source terms of the original variables:
  state_space_sources = np.genfromtxt('data-state-space-sources.csv', delimiter=',')

  # Mixture fraction vector:
  mf = np.genfromtxt('data-mixture-fraction.csv', delimiter=',')

We are also going to set some useful parameters, some of which will be used in
plotting functions:

.. code:: python

  # Select scaling method as per `preprocess.center_scale` function:
  scal_crit = 'auto'

  # Select biasing option:
  biasing_option = 2

  # Select number of clusters to partition the data set:
  n_clusters = 4

  # Select number of Principal Components that will be returned:
  n_components = 2

  # Set random seed for clustering and sampling techniques:
  random_seed = 100

  # Create labels for legends:
  legend_label = ['$\mathbf{X}$', '$\mathbf{X_r}$']

  # Create a list for variable annotations:
  state_space_names = ['$T$', '$H_2$', '$O_2$', '$O$', '$OH$', '$H_2O$', '$H$', '$HO_2$', '$CO$', '$CO_2$', '$HCO$']

  # By default plots will not be saved:
  save_filename = None

We start with clustering the data set that will result in an `idx` vector.
Clustering can be performed with any technique of choice. Here we will use one
of the available functions from the `preprocess` module ``preprocess.zero_neighborhood_bins``
and use the first Principal Component source term as the conditioning variable:

.. code:: python

  # Instantiate PCA class object:
  pca_X = PCA(state_space, scaling='auto', n_components=2)

  # Transform source terms of the original variables to PC-space:
  S_Z = pca_X.transform(state_space_sources, nocenter=True)

  # Cluster the data set:
  idx = preprocess.zero_neighborhood_bins(S_Z[:,0], k=4, zero_offset_percentage=2, split_at_zero=True, verbose=True)

--------------------------------------------------------------------------------

Equilibrate cluster populations iteratively
-------------------------------------------

This function is a special case of performing PCA on sampled data set.
Specifically, it uses equal number of samples from each cluster and allows to
analyze what happens when the data set is sampled gradually. It begins with
performing PCA on the original data set and then in
``n_iterations`` it will gradually decrease the number of populations in each
cluster larger than the smallest cluster, heading towards population of the
smallest cluster, in each cluster.
At each iteration we obtain a new sampled data set on which PCA is performed.
At the last iteration, the number of populations in each cluster are equal and
finally PCA is performed at the *equilibrated* data set.

A schematic representation of this procedure is presented in the figure below:

.. image:: ../images/cluster-biased-PCA-equilibration.png
    :width: 700
    :align: center

.. code:: python

  (eigenvalues, eigenvectors, pc_scores, _, idx_train, _, _) = reduction.equilibrate_cluster_populations(state_space, idx, scaling=scal_crit, X_source=[], n_components=n_components, biasing_option=biasing_option, n_iterations=10, stop_iter=0, random_seed=random_seed, verbose=True)

With ``verbose=True`` we will see some detailed information on number of samples
in each cluster at each iteration:

.. code-block:: text

  Biasing is performed with option 2.

  At iteration 1 taking samples:
  {0: 4144, 1: 14719, 2: 24689, 3: 2416}

  At iteration 2 taking samples:
  {0: 3953, 1: 13352, 2: 22215, 3: 2416}

  At iteration 3 taking samples:
  {0: 3762, 1: 11985, 2: 19741, 3: 2416}

  At iteration 4 taking samples:
  {0: 3571, 1: 10618, 2: 17267, 3: 2416}

  At iteration 5 taking samples:
  {0: 3380, 1: 9251, 2: 14793, 3: 2416}

  At iteration 6 taking samples:
  {0: 3189, 1: 7884, 2: 12319, 3: 2416}

  At iteration 7 taking samples:
  {0: 2998, 1: 6517, 2: 9845, 3: 2416}

  At iteration 8 taking samples:
  {0: 2807, 1: 5150, 2: 7371, 3: 2416}

  At iteration 9 taking samples:
  {0: 2616, 1: 3783, 2: 4897, 3: 2416}

  At iteration 10 taking samples:
  {0: 2416, 1: 2416, 2: 2416, 3: 2416}

Analyze centers change
----------------------

This function compares centers computed on the original data set
:math:`\mathbf{X}` versus on the sampled data set :math:`\mathbf{X_r}`.
The ``idx_train`` that is an input parameter could for instance be obtained
from ``equilibrate_cluster_populations``
and will thus represent the equilibrated data set sampled from the original data
set. It could also be obtained as sampled indices using any of the sampling
function from the ``DataSampler`` class.

.. code:: python

  (centers_X, centers_X_r, perc, plt) = reduction.analyze_centers_change(state_space, idx_train, variable_names=state_space_names, legend_label=legend_label, save_filename=save_filename)

Plotting example
^^^^^^^^^^^^^^^^

This function will produce a plot that shows the normalized centers and a
percentage by which the new centers have moved with respect to the original
ones. Example of a plot:

.. image:: ../images/centers-change.png
    :width: 500
    :align: center

If you do not wish to plot all variables present in a data set, use the
``plot_variables`` list as an input parameter to select indices of variables to
plot:

.. image:: ../images/centers-change-selected-variables.png
    :width: 260
    :align: center

Analyze eigenvector weights change
----------------------------------

The ``eigenvectors`` 3D array obtained from ``equilibrate_cluster_populations``
can now be used as an input parameter for plotting the eigenvector weights change
as we were gradually equilibrating cluster populations.

We are going to plot the first eigenvector (PC-1) weights change with three
variants of normalization.
To access the first eigenvector one can simply do:

.. code:: python

  eigenvectors[:,0,:]

similarly, to access the second eigenvector:

.. code:: python

  eigenvectors[:,1,:]

and so on.

.. code:: python

  plt = reduction.analyze_eigenvector_weights_change(eigenvectors[:,0,:], state_space_names, plot_variables=[], normalize=False, zero_norm=False, save_filename=save_filename)

Plotting example
^^^^^^^^^^^^^^^^

Three weight normalization variants are available:

- No normalization, the absolute values of the eigenvector weights are plotted. \
  To use this variant set ``normalize=False``. Example can be seen below:

.. image:: ../images/eigenvector-weights-movement-non-normalized.png
    :width: 500
    :align: center

- Normalizing so that the highest weight is equal to 1 and the smallest weight \
  is between 0 and 1. This is useful for judging the severity of the weight change. \
  To use this variant set ``normalize=True`` and ``zero_norm=False``. \
  Example can be seen below:

.. image:: ../images/eigenvector-weights-movement-normalized.png
    :width: 500
    :align: center

- Normalizing so that weights are between 0 and 1. This is useful for judging \
  the movement trends since it will blow up even the smallest changes to the entire \
  range 0-1. To use this variant set ``normalize=True`` and ``zero_norm=True``. \
  Example can be seen below:

.. image:: ../images/eigenvector-weights-movement-normalized-to-zero.png
    :width: 500
    :align: center

If you do not wish to plot all variables present in a data set, use the
``plot_variables`` list as an input parameter to select indices of variables to
plot:

.. image:: ../images/eigenvector-weights-movement-selected-variables.png
    :width: 280
    :align: center

If you are only interested in plotting a comparison in eigenvector weights
change between the original data set :math:`\mathbf{X}` and the sampled data set
:math:`\mathbf{X_r}`, you can set the ``eigenvectors`` input parameter to only
contain these two sets of weights.
The function will then understand that only these two should be compared:

.. image:: ../images/eigenvector-weights-movement-X-Xr.png
    :width: 500
    :align: center

Such plot can be done as well for pre-selected variables using the
``plot_variables`` list:

.. image:: ../images/eigenvector-weights-movement-X-Xr-selected-variables.png
    :width: 280
    :align: center

Analyze eigenvalue distribution
-------------------------------

Plotting example
^^^^^^^^^^^^^^^^

This function will produce a plot that shows the normalized eigenvalues
distribution for the original data set :math:`\mathbf{X}` and for the sampled
data set :math:`\mathbf{X_r}`.

.. code:: python

  plt = reduction.analyze_eigenvalue_distribution(state_space, idx_train, scal_crit, biasing_option, legend_label=legend_label, save_filename=save_filename)

Example of a plot:

.. image:: ../images/eigenvalue-distribution.png
    :width: 500
    :align: center

Visualize the biased manifold
-----------------------------

Plotting example
^^^^^^^^^^^^^^^^

Using the function ``reduction.plot_2d_manifold`` you can visualize any
2-dimensional manifold and additionally color it with a variable of choice.
Here we are going to plot the biased manifold resulting from performing PCA on
the sampled data set.

.. code:: python

  plt = reduction.plot_2d_manifold(pc_scores[:,0,-1], pc_scores[:,1,-1], color_variable=state_space[:,0], x_label='$Z_{r, 1}$', y_label='$Z_{r, 2}$', colorbar_label='$T$ [K]', save_filename=save_filename)

Example of a plot:

.. image:: ../images/biased-manifold.png
    :width: 500
    :align: center

--------------------------------------------------------------------------------

Generalization of PCA on sampled data set
-----------------------------------------

A more general approach to performing PCA on sampled data sets (instead of using
``equilibrate_cluster_populations`` function) is to use
``pca_on_sampled_data_set`` function. This function allows to perform PCA on
data that has been sampled in any way (in contrast to *equilibrated* sampling
which always samples equal number of samples from each cluster).

.. note::

  It is worth noting that function ``equilibrate_cluster_populations`` uses
  ``pca_on_sampled_data_set`` inside.

We will first inspect how many samples each cluster has (in the clusters we
identified earlier with the K-Means algorithm):

.. code:: python

  print(preprocess.get_populations(idx))

which shows us populations of each cluster to be:

.. code-block:: text

  [4335, 16086, 27163, 2416]

We begin by generating a manual sampling using the already identified clusters.
Suppose that we would like to severely under-represent the
two largest clusters and over-represent the features of the two smallest
clusters. Let's select 4000 samples from :math:`k_0`, 1000 samples from :math:`k_1`,
1000 samples from :math:`k_2` and 2400 samples from :math:`k_3`:

.. code:: python

  sample = DataSampler(idx, idx_test=[], random_seed=random_seed, verbose=True)

  (idx_manual, _) = sample.manual({0:4000, 1:1000, 2:1000, 3:2400}, sampling_type='number', test_selection_option=1)

In this example we are not interested in generating test samples, so we can
suppress returning those. The verbose information will tell us how sample
densities compare in terms of percentage of samples in each cluster:

.. code-block:: text

  Cluster 0: taking 4000 train samples out of 4335 observations (92.3%).
  Cluster 1: taking 1000 train samples out of 16086 observations (6.2%).
  Cluster 2: taking 1000 train samples out of 27163 observations (3.7%).
  Cluster 3: taking 2400 train samples out of 2416 observations (99.3%).

  Cluster 0: taking 335 test samples out of 335 remaining observations (100.0%).
  Cluster 1: taking 15086 test samples out of 15086 remaining observations (100.0%).
  Cluster 2: taking 26163 test samples out of 26163 remaining observations (100.0%).
  Cluster 3: taking 16 test samples out of 16 remaining observations (100.0%).

  Selected 8400 train samples (16.8%) and 41600 test samples (83.2%).

We now perform PCA on a data set that has been sampled according to
``idx_manual`` using the ``pca_on_sampled_data_set`` function:

.. code:: python

  (eigenvalues, eigenvectors, pc_scores, _, _, _, _, _) = reduction.pca_on_sampled_data_set(state_space, idx_manual, scal_crit, n_components, biasing_option)

Finally, we can generate all the same plots that were shown before.
Here, we are only going to present the new biased manifold resulting from
current manual sampling:

.. code::

  plt = reduction.plot_2d_manifold(pc_scores[:,0], pc_scores[:,1], color_variable=state_space[:,0], x_label='$Z_{r, 1}$', y_label='$Z_{r, 2}$', colorbar_label='$T$ [K]', save_filename=save_filename)

.. image:: ../images/generalize-sampling-biased-manifold.png
    :width: 500
    :align: center

--------------------------------------------------------------------------------

************
Bibliography
************

.. bibliography:: demo-pca-on-sampled-data-sets.bib
