import numpy as np
import random
import copy
import matplotlib.pyplot as plt
from PCAfold.styles import *

################################################################################
#
# Data Manipulation
#
################################################################################

_scalings_list = ['none', '', 'auto', 'std', 'pareto', 'vast', 'range', '-1to1', 'level', 'max', 'poisson', 'vast_2', 'vast_3', 'vast_4']

def center_scale(X, scaling, nocenter=False):
    """
    This function centers and scales the data set.

    Below we understand that :math:`\mathbf{X}_i` is the :math:`i^{th}` column
    of :math:`\mathbf{X}`.

    - Centering is performed by subtracting a center :math:`C_i` from
      :math:`\mathbf{X}_i`, where centers for all columns are stored
      in a vector :math:`\mathbf{C}`:

        .. math::

            \mathbf{X_c} = \mathbf{X} - \mathbf{C}

      Centers for each column are computed as:

        .. math::

            C_i = mean(\mathbf{X}_i)

      The only exception is the *-1 to 1* scaling which introduces a different
      quantity to center each column.

    - Scaling is performed by dividing :math:`\mathbf{X}_i` by a scaling
      factor :math:`d_i`, where scaling factors
      for all columns are stored in a vector :math:`\mathbf{D}`:

      .. math::

          \mathbf{X_s} = \mathbf{X} \\cdot \mathbf{D}^{-1}

    If both centering and scaling is applied:

    .. math::

        \mathbf{X_{cs}} = (\mathbf{X} - \mathbf{C}) \\cdot \mathbf{D}^{-1}

    Several scaling options are implemented here:

    +-----------------+--------------------------+--------------------------------------------------------------------+
    | Scaling method  | ``scaling``              | Scaling factor :math:`d_i`                                         |
    +=================+==========================+====================================================================+
    | None            | ``'none'``               | 1                                                                  |
    +-----------------+--------------------------+--------------------------------------------------------------------+
    | Auto            | ``'auto'`` or ``'std'``  | :math:`\sigma`                                                     |
    +-----------------+--------------------------+--------------------------------------------------------------------+
    | Pareto          | ``'pareto'``             | :math:`\sigma^2`                                                   |
    +-----------------+--------------------------+--------------------------------------------------------------------+
    | Vast            | ``'vast'``               | :math:`\sigma^2 / mean(\mathbf{X}_i)`                              |
    +-----------------+--------------------------+--------------------------------------------------------------------+
    | Range           | ``'range'``              | :math:`max(\mathbf{X}_i) - min(\mathbf{X}_i)`                      |
    +-----------------+--------------------------+--------------------------------------------------------------------+
    | | -1 to 1       | | ``'-1to1'``            | | :math:`d_i = 0.5 \cdot (max(\mathbf{X}_i) - min(\mathbf{X}_i))`  |
    | |               | |                        | | :math:`C_i = 0.5 \cdot (max(\mathbf{X}_i) + min(\mathbf{X}_i))`  |
    +-----------------+--------------------------+--------------------------------------------------------------------+
    | Level           | ``'level'``              | :math:`mean(\mathbf{X}_i)`                                         |
    +-----------------+--------------------------+--------------------------------------------------------------------+
    | Max             | ``'max'``                | :math:`max(\mathbf{X}_i)`                                          |
    +-----------------+--------------------------+--------------------------------------------------------------------+
    | Poisson         |``'poisson'``             | :math:`\sqrt{mean(\mathbf{X}_i)}`                                  |
    +-----------------+--------------------------+--------------------------------------------------------------------+
    | Vast-2          | ``'vast_2'``             | :math:`\sigma^2 k^2 / mean(\mathbf{X}_i)`                          |
    +-----------------+--------------------------+--------------------------------------------------------------------+
    | Vast-3          | ``'vast_3'``             | :math:`\sigma^2 k^2 / max(\mathbf{X}_i)`                           |
    +-----------------+--------------------------+--------------------------------------------------------------------+
    | Vast-4          | ``'vast_4'``             | :math:`\sigma^2 k^2 / (max(\mathbf{X}_i) - min(\mathbf{X}_i))`     |
    +-----------------+--------------------------+--------------------------------------------------------------------+

    where :math:`\sigma` is the standard deviation of :math:`\mathbf{X}_i`
    and :math:`k` is the kurtosis of :math:`\mathbf{X}_i`.

    **Example:**

    .. code:: python

        from PCAfold import center_scale
        import numpy as np

        # Generate dummy data set:
        X = np.random.rand(100,20)

        # Center and scale:
        (X_cs, X_center, X_scale) = center_scale(X, 'range', nocenter=False)

    :param X:
        original data set :math:`\mathbf{X}`.
    :param scaling:
        string specifying the scaling methodology.
    :param nocenter: (optional)
        boolean specifying whether data should be centered by mean.

    :raises ValueError:
        if ``scaling`` method is not a string or is not within the available scalings.

    :raises ValueError:
        if ``nocenter`` is not a boolean.

    :return:
        - **X_cs** - centered and scaled data set :math:`\mathbf{X_{cs}}`.
        - **X_center** - vector of centers :math:`\mathbf{C}` applied on the original data set :math:`\mathbf{X}`.
        - **X_scale** - vector of scales :math:`\mathbf{D}` applied on the original data set :math:`\mathbf{X}`.
    """

    if not isinstance(scaling, str):
        raise ValueError("Parameter `scaling` has to be a string.")
    else:
        if scaling.lower() not in _scalings_list:
            raise ValueError("Unrecognized scaling method.")

    if not isinstance(nocenter, bool):
        raise ValueError("Parameter `nocenter` has to be a boolean.")

    try:
        (n_observations, n_variables) = np.shape(X)
    except:
        X = X[:,np.newaxis]
        (n_observations, n_variables) = np.shape(X)

    X_cs = np.zeros_like(X, dtype=float)
    X_center = X.mean(axis=0)

    dev = 0 * X_center
    kurt = 0 * X_center

    for i in range(0, n_variables):

        # Calculate the standard deviation (required for some scalings):
        dev[i] = np.std(X[:, i], ddof=0)

        # Calculate the kurtosis (required for some scalings):
        kurt[i] = np.sum((X[:, i] - X_center[i]) ** 4) / n_observations / (np.sum((X[:, i] - X_center[i]) ** 2) / n_observations) ** 2

    scaling = scaling.upper()
    eps = np.finfo(float).eps
    if scaling == 'NONE' or scaling == '':
       X_scale = np.ones(n_variables)
    elif scaling == 'AUTO' or scaling == 'STD':
       X_scale = dev
    elif scaling == 'VAST':
       X_scale = dev * dev / (X_center + eps)
    elif scaling == 'VAST_2':
       X_scale = dev * dev * kurt * kurt / (X_center + eps)
    elif scaling == 'VAST_3':
       X_scale = dev * dev * kurt * kurt / np.max(X, axis=0)
    elif scaling == 'VAST_4':
       X_scale = dev * dev * kurt * kurt / (np.max(X, axis=0) - np.min(X, axis=0))
    elif scaling == 'RANGE':
       X_scale = np.max(X, axis=0) - np.min(X, axis=0)
    elif scaling == '-1TO1':
       X_center = 0.5*(np.max(X, axis=0) + np.min(X, axis=0))
       X_scale = 0.5*(np.max(X, axis=0) - np.min(X, axis=0))
    elif scaling == 'LEVEL':
       X_scale = X_center
    elif scaling == 'MAX':
       X_scale = np.max(X, axis=0)
    elif scaling == 'PARETO':
       X_scale = np.zeros(n_variables)
       for i in range(0, n_variables):
           X_scale[i] = np.sqrt(np.std(X[:, i], ddof=0))
    elif scaling == 'POISSON':
       X_scale = np.sqrt(X_center)
    else:
        raise ValueError('Unsupported scaling option')

    for i in range(0, n_variables):
        if nocenter:
            X_cs[:, i] = (X[:, i]) / X_scale[i]
        else:
            X_cs[:, i] = (X[:, i] - X_center[i]) / X_scale[i]

    if nocenter:
        X_center = np.zeros(n_variables)

    return(X_cs, X_center, X_scale)

def invert_center_scale(X_cs, X_center, X_scale):
    """
    This function inverts whatever centering and scaling was done by
    ``center_scale`` function:

    .. math::

        \mathbf{X} = \mathbf{X_{cs}} \\cdot \mathbf{D} + \mathbf{C}

    **Example:**

    .. code:: python

        from PCAfold import center_scale, invert_center_scale
        import numpy as np

        # Generate dummy data set:
        X = np.random.rand(100,20)

        # Center and scale:
        (X_cs, X_center, X_scale) = center_scale(X, 'range', nocenter=False)

        # Uncenter and unscale:
        X = invert_center_scale(X_cs, X_center, X_scale)

    :param X_cs:
        centered and scaled data set :math:`\mathbf{X_{cs}}`.
    :param X_center:
        vector of centers :math:`\mathbf{C}` applied on the original data set :math:`\mathbf{X}`.
    :param X_scale:
        vector of scales :math:`\mathbf{D}` applied on the original data set :math:`\mathbf{X}`.

    :return:
        - **X** - original data set :math:`\mathbf{X}`.
    """

    try:
        (_, n_variables) = np.shape(X_cs)
    except:
        n_variables = 1

    if n_variables == 1:
        X = X_cs * X_scale + X_center
    else:
        X = np.zeros_like(X_cs, dtype=float)
        for i in range(0, n_variables):
            X[:, i] = X_cs[:, i] * X_scale[i] + X_center[i]

    return(X)

class PreProcessing:
    """
    This class performs a composition of data manipulation done by ``remove_constant_vars``
    and ``center_scale`` functions on the original data set
    :math:`\mathbf{X}`. It can be used to store the result of that manipulation.
    Specifically, it:

    - checks for constant columns in a data set and removes them,
    - centers and scales the data.

    **Example:**

    .. code:: python

        from PCAfold import PreProcessing
        import numpy as np

        # Generate dummy data set with a constant variable:
        X = np.random.rand(100,20)
        X[:,5] = np.ones((100,))

        # Instantiate PreProcessing class object:
        preprocessed = PreProcessing(X, 'range', nocenter=False)

    :param X:
        original data set :math:`\mathbf{X}`.
    :param scaling:
        string specifying the scaling methodology as per
        ``preprocess.center_scale`` function.
    :param nocenter: (optional)
        boolean specifying whether data should be centered by mean.

    **Attributes:**

        - **X_removed** - data set with removed constant columns.
        - **idx_removed** - the indices of columns removed from :math:`\mathbf{X}`.
        - **idx_retained** - the indices of columns retained in :math:`\mathbf{X}`.
        - **X_cs** - centered and scaled data set :math:`\mathbf{X_{cs}}`.
        - **X_center** - vector of centers :math:`\mathbf{C}` applied on the original data set :math:`\mathbf{X}`.
        - **X_scale** - vector of scales :math:`\mathbf{D}` applied on the original data set :math:`\mathbf{X}`.
    """

    def __init__(self, X, scaling='none', nocenter=False):

        (self.X_removed, self.idx_removed, self.idx_retained) = remove_constant_vars(X)
        (self.X_cs, self.X_center, self.X_scale) = center_scale(self.X_removed, scaling, nocenter=nocenter)

def remove_constant_vars(X, maxtol=1e-12, rangetol=1e-4):
    """
    This function removes any constant columns in the data set :math:`\mathbf{X}`.
    The :math:`i^{th}` column :math:`\mathbf{X}_i` is considered constant if either of the following is true:

    - the maximum of an absolute value of a column :math:`\mathbf{X}_i` is less than ``maxtol``:

    .. math::

        max(|\mathbf{X}_i|) < \\verb|maxtol|

    - the ratio of the range of values in a column :math:`\mathbf{X}_i` to :math:`max(|\mathbf{X}_i|)` is less than ``rangetol``:

    .. math::

        \\frac{max(\mathbf{X}_i) - min(\mathbf{X}_i)}{max(|\mathbf{X}_i|)} < \\verb|rangetol|

    Specifically, it can be used as pre-processing for PCA so the eigenvalue
    calculation doesn't break.

    **Example:**

    .. code:: python

        from PCAfold import remove_constant_vars
        import numpy as np

        # Generate dummy data set with a constant variable:
        X = np.random.rand(100,20)
        X[:,5] = np.ones((100,))

        # Remove the constant column:
        (X_removed, idx_removed, idx_retained) = remove_constant_vars(X)

    :param X:
        original data set :math:`\mathbf{X}`.
    :param maxtol:
        tolerance for :math:`max(|\mathbf{X}_i|)`.
    :param rangetol:
        tolerance for :math:`max(\mathbf{X}_i) - min(\mathbf{X}_i)` over :math:`max(|\mathbf{X}_i|)`.

    :return:
        - **X_removed** - original data set :math:`\mathbf{X}` with any constant columns removed.
        - **idx_removed** - the indices of columns removed from :math:`\mathbf{X}`.
        - **idx_retained** - the indices of columns retained in :math:`\mathbf{X}`.
    """

    (n_observations, n_variables) = np.shape(X)

    idx_removed = []
    idx_retained = []

    for i in reversed(range(0, n_variables)):

        min = np.min(X[:, i], axis=0)
        max = np.max(X[:, i], axis=0)
        maxabs = np.max(np.abs(X[:, i]), axis=0)

        if (maxabs < maxtol) or ((max - min) / maxabs < rangetol):
            X = np.delete(X, i, 1)
            idx_removed.append(i)
        else:
            idx_retained.append(i)

    X_removed = X
    idx_removed = idx_removed[::-1]
    idx_retained = idx_retained[::-1]

    return(X_removed, idx_removed, idx_retained)

def analyze_centers_change(X, idx_X_r, variable_names=[], plot_variables=[], legend_label=[], title=None, save_filename=None):
    """
    This function analyzes the change in normalized centers computed on the
    sampled subset of the original data set :math:`\mathbf{X_r}` with respect
    to the full original data set :math:`\mathbf{X}`.

    The original data set :math:`\mathbf{X}` is first normalized so that each
    variable ranges from 0 to 1:

    .. math::

        ||\mathbf{X}|| = \\frac{\mathbf{X} - min(\mathbf{X})}{max(\mathbf{X} - min(\mathbf{X}))}

    This normalization is done so that centers can be compared across variables
    on one plot.
    Samples are then extracted from :math:`||\mathbf{X}||`, according to
    ``idx_X_r``, to form :math:`||\mathbf{X_r}||`.

    Normalized centers are computed as:

    .. math::

        ||\mathbf{C}|| = mean(||\mathbf{X}||)

    .. math::

        ||\mathbf{C_r}|| = mean(||\mathbf{X_r}||)

    Percentage measuring the relative change in normalized centers is
    computed as:

    .. math::

        p = \\frac{||\mathbf{C_r}|| - ||\mathbf{C}||}{||\mathbf{C}||} \cdot 100\%

    **Example:**

    .. code:: python

        from PCAfold import analyze_centers_change, DataSampler
        import numpy as np

        # Generate dummy data set:
        X = np.random.rand(100,10)

        # Generate dummy sampling indices:
        idx = np.zeros((100,))
        idx[50:80] = 1
        selection = DataSampler(idx)
        (idx_X_r, _) = selection.number(20, test_selection_option=1)

        # Analyze the change in normalized centers:
        (normalized_C, normalized_C_r, center_movement_percentage, plt) = analyze_centers_change(X, idx_X_r)

    :param X:
        original data set :math:`\mathbf{X}`.
    :param idx_X_r:
        vector of indices that should be extracted from :math:`\mathbf{X}` to
        form :math:`\mathbf{X_r}`.
    :param variable_names: (optional)
        list of strings specifying variable names.
    :param plot_variables: (optional)
        list of integers specifying indices of variables to be plotted.
        By default, all variables are plotted.
    :param legend_label: (optional)
        list of strings specifying labels for the legend. First entry will refer
        to :math:`||\mathbf{C}||` and second entry to :math:`||\mathbf{C_r}||`.
        If the list is empty, legend will not be plotted.
    :param title: (optional)
        string specifying plot title. If set to ``None``
        title will not be plotted.
    :param save_filename: (optional)
        string specifying plot save location/filename. If set to ``None``
        plot will not be saved.

    :return:
        - **normalized_C** - normalized centers :math:`||\mathbf{C}||`.
        - **normalized_C_r** - normalized centers :math:`||\mathbf{C_r}||`.
        - **center_movement_percentage** - percentage :math:`p`\
        measuring the relative change in normalized centers.
        - **plt** - plot handle.
    """

    color_X = '#191b27'
    color_X_r = '#ff2f18'
    color_link = '#bbbbbb'

    (n_observations, n_variables) = np.shape(X)

    # Create default labels for variables:
    if len(variable_names) == 0:
        variable_names = ['$X_{' + str(i) + '}$' for i in range(0, n_variables)]

    if len(plot_variables) != 0:
        X = X[:,plot_variables]
        variable_names = [variable_names[i] for i in plot_variables]
        (_, n_variables) = np.shape(X)

    X_normalized = (X - np.min(X, axis=0))
    X_normalized = X_normalized / np.max(X_normalized, axis=0)

    # Extract X_r using the provided idx_X_r:
    X_r_normalized = X_normalized[idx_X_r,:]

    # Find centers:
    normalized_C = np.mean(X_normalized, axis=0)
    normalized_C_r = np.mean(X_r_normalized, axis=0)

    # Compute the relative percentage by how much the center has moved:
    center_movement_percentage = (normalized_C_r - normalized_C) / normalized_C * 100

    x_range = np.arange(1, n_variables+1)

    fig, ax = plt.subplots(figsize=(n_variables, 6))

    plt.scatter(x_range, normalized_C, c=color_X, marker='o', s=marker_size, edgecolor='none', alpha=1, zorder=2)
    plt.scatter(x_range, normalized_C_r, c=color_X_r, marker='>', s=marker_size, edgecolor='none', alpha=1, zorder=2)
    plt.xticks(x_range, variable_names, fontsize=font_axes, **csfont)
    plt.yticks(fontsize=font_axes, **csfont)
    plt.ylabel('Normalized center [-]', fontsize=font_labels, **csfont)
    plt.ylim(-0.05,1.05)
    plt.xlim(0, n_variables+1.5)
    plt.grid(alpha=0.3, zorder=0)

    for i in range(0, n_variables):

        dy = normalized_C_r[i] - normalized_C[i]
        plt.arrow(x_range[i], normalized_C[i], 0, dy, color=color_link, ls='-', lw=1, zorder=1)

    if title != None:
        plt.title(title, fontsize=font_title, **csfont)

    for i, value in enumerate(center_movement_percentage):
        plt.text(i+1.05, normalized_C_r[i]+0.01, str(int(value)) + ' %', fontsize=font_text, c=color_X_r, **csfont)

    ax.spines["top"].set_visible(True)
    ax.spines["bottom"].set_visible(True)
    ax.spines["right"].set_visible(True)
    ax.spines["left"].set_visible(True)

    if len(legend_label) != 0:
        lgnd = plt.legend(legend_label, fontsize=font_legend, markerscale=marker_scale_legend, loc="upper right")

    if save_filename != None:
        plt.savefig(save_filename, dpi = 500, bbox_inches='tight')

    return(normalized_C, normalized_C_r, center_movement_percentage, plt)

################################################################################
#
# Data Sampling
#
################################################################################

class DataSampler:
    """
    This class enables selecting train and test data samples.

    **Example:**

    .. code::

      from PCAfold import DataSampler
      import numpy as np

      # Generate dummy idx vector:
      idx = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1])

      # Instantiate DataSampler class object:
      selection = DataSampler(idx, idx_test=[5,9], random_seed=100, verbose=True)

    :param idx:
        vector of cluster classifications.
    :param idx_test: (optional)
        vector or list of user-provided indices for test data. If specified, train
        data will be selected ignoring the indices in ``idx_test`` and the test
        data will be returned the same as the user-provided ``idx_test``.
        If not specified, test samples will be selected according to the
        ``test_selection_option`` parameter (see documentation for each sampling function).
        Setting fixed ``idx_test`` parameter may be useful if training a machine
        learning model on specific test samples is desired.
    :param random_seed: (optional)
        integer specifying random seed for random sample selection.
    :param verbose: (optional)
        boolean for printing sampling details.

    :raises ValueError:
        if ``idx`` vector has length zero, is not a list or ``numpy.ndarray``.
    :raises ValueError:
        if ``idx_test`` vector has more unique observations than ``idx``, is not a list or ``numpy.ndarray``.
    :raises ValueError:
        if ``random_seed`` is not an integer or ``None``.
    :raises ValueError:
        if ``verbose`` is not a boolean.
    """

    def __init__(self, idx, idx_test=[], random_seed=None, verbose=False):

        if len(idx) == 0:
            raise ValueError("Parameter `idx` has length zero.")
        else:
            if len(np.unique(idx_test)) > len(idx):
                raise ValueError("Parameter `idx` has less observations than current `idx_test`.")
            else:
                if isinstance(idx, list) or isinstance(idx, np.ndarray):
                    self.__idx = idx
                else:
                    raise ValueError("Parameter `idx` has to be a list or numpy.ndarray.")

        if len(np.unique(idx_test)) > len(idx):
            raise ValueError("Parameter `idx_test` has more unique observations than `idx`.")
        else:
            if isinstance(idx_test, list) or isinstance(idx_test, np.ndarray):
                self.__idx_test = idx_test
            else:
                raise ValueError("Parameter `idx_test` has to be a list or numpy.ndarray.")

        if random_seed != None:
            if not isinstance(random_seed, int):
                raise ValueError("Parameter `random_seed` has to be an integer or None.")
            if isinstance(random_seed, bool):
                raise ValueError("Parameter `random_seed` has to be an integer or None.")
            else:
                self.__random_seed = random_seed
        else:
            self.__random_seed = random_seed

        if not isinstance(verbose, bool):
            raise ValueError("Parameter `verbose` has to be a boolean.")
        else:
            self.__verbose = verbose

        if len(np.unique(idx_test)) != 0:
            self.__using_user_defined_idx_test = True
            if self.verbose==True:
                print('User defined test samples will be used. Parameter `test_selection_option` will be ignored.\n')
        else:
            self.__using_user_defined_idx_test = False

    @property
    def idx(self):
        return self.__idx

    @property
    def idx_test(self):
        return self.__idx_test

    @property
    def random_seed(self):
        return self.__random_seed

    @property
    def verbose(self):
        return self.__verbose

    @idx.setter
    def idx(self, new_idx):
        if len(new_idx) == 0:
            raise ValueError("Parameter `idx` has length zero.")
        else:
            if len(np.unique(self.idx_test)) > len(new_idx):
                raise ValueError("Parameter `idx` has less observations than current `idx_test`.")
            else:
                if isinstance(new_idx, list) or isinstance(new_idx, np.ndarray):
                    self.__idx = new_idx
                else:
                    raise ValueError("Parameter `idx` has to be a list or numpy.ndarray.")

    @idx_test.setter
    def idx_test(self, new_idx_test):
        if len(new_idx_test) > len(self.idx):
            raise ValueError("Parameter `idx_test` has more unique observations than `idx`.")
        else:
            if isinstance(new_idx_test, list) or isinstance(new_idx_test, np.ndarray):
                self.__idx_test = new_idx_test
            else:
                raise ValueError("Parameter `idx_test` has to be a list or numpy.ndarray.")

            if len(np.unique(new_idx_test)) != 0:
                self.__using_user_defined_idx_test = True
                if self.verbose==True:
                    print('User defined test samples will be used. Parameter `test_selection_option` will be ignored.\n')
            else:
                self.__using_user_defined_idx_test = False

    @random_seed.setter
    def random_seed(self, new_random_seed):
        if new_random_seed != None:
            if not isinstance(new_random_seed, int):
                raise ValueError("Parameter `random_seed` has to be an integer or None.")
            if isinstance(new_random_seed, bool):
                raise ValueError("Parameter `random_seed` has to be an integer or None.")
            else:
                self.__random_seed = new_random_seed
        else:
            self.__random_seed = new_random_seed

    @verbose.setter
    def verbose(self, new_verbose):
        if not isinstance(new_verbose, bool):
            raise ValueError("Parameter `verbose` has to be a boolean.")
        else:
            self.__verbose = new_verbose

    def __print_verbose_information_sampling(self, idx, idx_train, idx_test):
        """
        This private function prints detailed information on train and test sampling when
        ``verbose=True``.

        :param idx:
            vector of cluster classifications.
        :param idx_train:
            indices of the train data.
        :param idx_test:
            indices of the test data.
        """

        cluster_populations = get_populations(idx)
        k = np.size(np.unique(idx))
        n_observations = len(idx)

        for cl_id in range(0,k):
            train_indices = [t_id for t_id in idx_train if idx[t_id,]==cl_id]
            if cluster_populations[cl_id] != 0:
                print("Cluster " + str(cl_id) + ": taking " + str(len(train_indices)) + " train samples out of " + str(cluster_populations[cl_id]) + " observations (%.1f" % (len(train_indices)/(cluster_populations[cl_id])*100) + "%).")
            else:
                print("Cluster " + str(cl_id) + ": taking " + str(len(train_indices)) + " train samples out of " + str(cluster_populations[cl_id]) + " observations (%.1f" % (0) + "%).")
        print("")

        for cl_id in range(0,k):
            train_indices = [t_id for t_id in idx_train if idx[t_id,]==cl_id]
            test_indices = [t_id for t_id in idx_test if idx[t_id,]==cl_id]
            if (cluster_populations[cl_id] - len(train_indices)) != 0:
                print("Cluster " + str(cl_id) + ": taking " + str(len(test_indices)) + " test samples out of " + str(cluster_populations[cl_id] - len(train_indices)) + " remaining observations (%.1f" % (len(test_indices)/(cluster_populations[cl_id] - len(train_indices))*100) + "%).")
            else:
                print("Cluster " + str(cl_id) + ": taking " + str(len(test_indices)) + " test samples out of " + str(cluster_populations[cl_id] - len(train_indices)) + " remaining observations (%.1f" % (0) + "%).")

        print('\nSelected ' + str(np.size(idx_train)) + ' train samples (%.1f' % (np.size(idx_train)*100/n_observations) + '%) and ' + str(np.size(idx_test)) + ' test samples (%.1f' % (np.size(idx_test)*100/n_observations) + '%).\n')

    def number(self, perc, test_selection_option=1):
        """
        This function uses classifications into :math:`k` clusters and samples
        fixed number of observations from every cluster as training data.
        In general, this results in a balanced representation of features
        identified by a clustering algorithm.

        **Example:**

        .. code::

          from PCAfold import DataSampler
          import numpy as np

          # Generate dummy idx vector:
          idx = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1])

          # Instantiate DataSampler class object:
          selection = DataSampler(idx, verbose=True)

          # Generate sampling:
          (idx_train, idx_test) = selection.number(20, test_selection_option=1)

        **Train data:**

        The number of train samples is estimated based on the percentage
        ``perc`` provided.
        First, the total number of samples for training is estimated as a
        percentage ``perc`` from the total number of observations ``n_observations`` in a data set.
        Next, this number is divided equally into :math:`k` clusters. The
        result ``n_of_samples`` is the number of samples that will be selected
        from each cluster:

        .. math::

            \\verb|n_of_samples| = \\verb|int| \Big( \\frac{\\verb|perc| \cdot \\verb|n_observations|}{k \cdot 100} \Big)

        **Test data:**

        Two options for sampling test data are implemented. If you select
        ``test_selection_option=1`` all remaining samples that were not taken as
        train data become the test data. If you select ``test_selection_option=2``,
        the smallest cluster is found and the remaining number of observations
        :math:`m` are taken as test data in that cluster. Next, the same number
        of samples :math:`m` is taken from all remaining larger clusters.

        The scheme below presents graphically how train and test data can be selected using ``test_selection_option`` parameter:

        .. image:: ../images/sampling-test-selection-option-number.png
          :width: 700
          :align: center

        Here :math:`n` and :math:`m` are fixed numbers for each cluster.
        In general, :math:`n \\neq m`.

        :param perc:
            percentage of data to be selected as training data from the entire data set.
            For instance, set ``perc=20`` if you want to select 20%.
        :param test_selection_option: (optional)
            option for how the test data is selected.
            Select ``test_selection_option=1`` if you want all remaining samples
            to become test data.
            Select ``test_selection_option=2`` if you want to select a subset
            of the remaining samples as test data.

        :raises ValueError:
            if ``perc`` is not a number between 0-100.

        :raises ValueError:
            if ``test_selection_option`` is not equal to 1 or 2.

        :raises ValueError:
            if the percentage specified is too high in combination with the
            user-provided ``idx_test`` vector and there aren't enough samples to
            select as train data.

        :return:
            - **idx_train** - indices of the train data.
            - **idx_test** - indices of the test data.
        """

        # Check if `perc` parameter was passed correctly:
        if (perc < 0) or (perc > 100):
            raise ValueError("Percentage has to be between 0-100.")

        # Check if `test_selection_option` parameter was passed correctly:
        _test_selection_option = [1,2]
        if test_selection_option not in _test_selection_option:
            raise ValueError("Test selection option can only be 1 or 2.")

        # Degrade clusters if needed:
        if (len(np.unique(self.idx)) != (np.max(self.idx)+1)) or (np.min(self.idx) != 0):
            (self.idx, _) = degrade_clusters(self.idx, verbose=False)

        # Initialize vector of indices 0..n_observations:
        n_observations = len(self.idx)
        idx_full = np.arange(0, n_observations)
        idx_test = np.unique(np.array(self.idx_test))
        idx_full_no_test = np.setdiff1d(idx_full, idx_test)

        # Find the number of clusters:
        k = np.size(np.unique(self.idx))

        # Calculate fixed number of samples that will be taken from every cluster as the training data:
        n_of_samples = int(perc*n_observations/k/100)

        # Initialize auxiliary variables:
        idx_train = []
        cluster_test = []

        # Get clusters and split them into train and test indices:
        for cl_id in range(0,k):

            if self.random_seed != None:
                random.seed(self.random_seed)

            # Variable `cluster` contains indices of observations that are allowed to be selected as train samples in a particular cluster:
            cluster = []
            for i, id in enumerate(self.idx[idx_full_no_test]):
                if id == cl_id:
                    cluster.append(idx_full_no_test[i])

            # Selection of training data:
            if int(len(cluster)) < n_of_samples:
                raise ValueError("The requested percentage requires taking more samples from cluster " + str(cl_id) + " than there are available observations in that cluster. Consider lowering the percentage or use a different sampling function.")
            else:
                cluster_train = np.array(random.sample(cluster, n_of_samples))
                idx_train = np.concatenate((idx_train, cluster_train))

            if self.__using_user_defined_idx_test==False:

                # Selection of test data - all data that remains is test data:
                if test_selection_option == 1:

                    cluster_test = np.setdiff1d(cluster, cluster_train)
                    idx_test = np.concatenate((idx_test, cluster_test))

        # Selection of test data - equal samples from each cluster:
                if test_selection_option == 2:

                    cluster_test.append(np.setdiff1d(cluster, cluster_train))

        if self.__using_user_defined_idx_test==False:

            if test_selection_option == 2:

                # Search for the smallest number of remaining observations in any cluster:
                minimum_test_samples = n_observations
                for cl_id in range(0,k):
                    if len(cluster_test[cl_id]) < minimum_test_samples:
                        minimum_test_samples = len(cluster_test[cl_id])

                # Sample that amount from every cluster:
                for cl_id in range(0,k):
                    idx_test = np.concatenate((idx_test, random.sample(list(cluster_test[cl_id]), minimum_test_samples)))

        idx_train = np.sort(idx_train.astype(int))
        idx_test = np.sort(idx_test.astype(int))

        # Print detailed information on sampling:
        if self.verbose == True:
            self.__print_verbose_information_sampling(self.idx, idx_train, idx_test)

        return (idx_train, idx_test)

    def percentage(self, perc, test_selection_option=1):
        """
        This function uses classifications into :math:`k` clusters and
        samples a certain percentage ``perc`` from every cluster as the training data.

        **Example:**

        .. code:: python

          from PCAfold import DataSampler
          import numpy as np

          # Generate dummy idx vector:
          idx = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1])

          # Instantiate DataSampler class object:
          selection = DataSampler(idx, verbose=True)

          # Generate sampling:
          (idx_train, idx_test) = selection.percentage(20, test_selection_option=1)

        *Note:*
        If the cluster sizes are comparable, this function will give a similar
        train sample distribution as random sampling (``DataSampler.random``).
        This sampling can be useful in cases where one cluster is significantly
        smaller than others and there is a chance that this cluster will not get
        covered in the train data if random sampling was used.

        **Train data:**

        The number of train samples is estimated based on the percentage ``perc`` provided.
        First, the size of the :math:`i^{th}` cluster is estimated ``cluster_size_i``
        and then a percentage ``perc`` of that number is selected.

        **Test data:**

        Two options for sampling test data are implemented. If you select
        ``test_selection_option=1`` all remaining samples that were not taken as
        train data become the test data. If you select
        ``test_selection_option=2`` the same procedure will be used to select
        test data as was used to select train data (only allowed if the number of samples
        taken as train data from any cluster did not exceed 50% of observations
        in that cluster).

        The scheme below presents graphically how train and test data can be
        selected using ``test_selection_option`` parameter:

        .. image:: ../images/sampling-test-selection-option-percentage.png
          :width: 700
          :align: center

        Here :math:`p` is the percentage ``perc`` provided.

        :param perc:
            percentage of data to be selected as training data from each cluster.
            For instance, set ``perc=20`` if you want to select 20%.
        :param test_selection_option: (optional)
            option for how the test data is selected.
            Select ``test_selection_option=1`` if you want all remaining samples
            to become test data.
            Select ``test_selection_option=2`` if you want to select a subset
            of the remaining samples as test data.

        :raises ValueError:
            if ``perc`` is not a number between 0-100.

        :raises ValueError:
            if ``test_selection_option`` is not equal to 1 or 2.

        :raises ValueError:
            if the perecentage specified is too high in combination with the
            user-provided ``idx_test`` vector and there aren't enough samples to
            select as train data.

        :return:
            - **idx_train** - indices of the train data.
            - **idx_test** - indices of the test data.
        """

        # Check if `perc` parameter was passed correctly:
        if (perc < 0) or (perc > 100):
            raise ValueError("Percentage has to be between 0-100.")

        # Check if `test_selection_option` parameter was passed correctly:
        _test_selection_option = [1,2]
        if test_selection_option not in _test_selection_option:
            raise ValueError("Test selection option can only be 1 or 2.")

        # Degrade clusters if needed:
        if (len(np.unique(self.idx)) != (np.max(self.idx)+1)) or (np.min(self.idx) != 0):
            (self.idx, _) = degrade_clusters(self.idx, verbose=False)

        # Initialize vector of indices 0..n_observations:
        n_observations = len(self.idx)
        idx_full = np.arange(0, n_observations)
        idx_test = np.unique(np.array(self.idx_test))
        idx_full_no_test = np.setdiff1d(idx_full, idx_test)

        # Find the number of clusters:
        k = np.size(np.unique(self.idx))

        # Initialize auxiliary variables:
        idx_train = []

        # Get cluster populations:
        cluster_populations = get_populations(self.idx)

        # Get clusters and split them into training and test indices:
        for cl_id in range(0,k):

            if self.random_seed != None:
                random.seed(self.random_seed)

            # Variable `cluster` contains indices of observations that are allowed to be selected as train samples in a particular cluster:
            cluster = []
            for i, id in enumerate(self.idx[idx_full_no_test]):
                if id == cl_id:
                    cluster.append(idx_full_no_test[i])

            # Selection of training data:
            if int(len(cluster)) < int(cluster_populations[cl_id]*perc/100):
                raise ValueError("The requested percentage requires taking more samples from cluster " + str(cl_id) + " than there are available observations in that cluster. Consider lowering the percentage or use a different sampling function.")
            else:
                cluster_train = np.array(random.sample(cluster, int(cluster_populations[cl_id]*perc/100)))
                idx_train = np.concatenate((idx_train, cluster_train))

            if self.__using_user_defined_idx_test==False:

                # Selection of test data - all data that remains is test data:
                if test_selection_option == 1:

                    cluster_test = np.setdiff1d(cluster, cluster_train)
                    idx_test = np.concatenate((idx_test, cluster_test))

                if test_selection_option == 2:

                    # Check if there is enough test samples to select:
                    if perc > 50:
                        raise ValueError("Percentage is larger than 50% and test samples cannot be selected with `test_selection_option=2`.")
                    else:
                        cluster_test = np.setdiff1d(cluster, cluster_train)
                        cluster_test_sampled = np.array(random.sample(list(cluster_test), int(cluster_populations[cl_id]*perc/100)))
                        idx_test = np.concatenate((idx_test, cluster_test_sampled))

        idx_train = np.sort(idx_train.astype(int))
        idx_test = np.sort(idx_test.astype(int))

        # Print detailed information on sampling:
        if self.verbose == True:
            self.__print_verbose_information_sampling(self.idx, idx_train, idx_test)

        return (idx_train, idx_test)

    def manual(self, sampling_dictionary, sampling_type='percentage', test_selection_option=1):
        """
        This function uses classifications into :math:`k` clusters
        and a dictionary ``sampling_dictionary`` in which you manually specify what
        ``'percentage'`` (or what ``'number'``) of samples will be
        selected as the train data from each cluster. The dictionary keys are
        cluster classifications as per ``idx`` and the dictionary values are either
        percentage or number of train samples to be selected. The default dictionary
        values are percentage but you can select ``sampling_type='number'`` in order
        to interpret the values as a number of samples.

        **Example:**

        .. code:: python

          from PCAfold import DataSampler
          import numpy as np

          # Generate dummy idx vector:
          idx = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2])

          # Instantiate DataSampler class object:
          selection = DataSampler(idx, verbose=True)

          # Generate sampling:
          (idx_train, idx_test) = selection.manual({0:1, 1:1, 2:1}, sampling_type='number', test_selection_option=1)

        **Train data:**

        The number of train samples selected from each cluster is estimated based
        on the ``sampling_dictionary``. For ``key : value``, percentage ``value``
        (or number ``value``) of samples will be selected from cluster ``key``.

        **Test data:**

        Two options for sampling test data are implemented.
        If you select ``test_selection_option=1`` all remaining samples that
        were not taken as train data become the test data.
        If you select
        ``test_selection_option=2`` the same procedure will be used to select
        test data as was used to select train data (only allowed if the number
        of samples taken as train data from any cluster did not exceed 50%
        of observations in that cluster).

        The scheme below presents graphically how train and test data can be
        selected using ``test_selection_option`` parameter:

        .. image:: ../images/sampling-test-selection-option-manual.png
          :width: 700
          :align: center

        Here it is understood that :math:`n_1` train samples were requested from
        the first cluster, :math:`n_2` from the second cluster and :math:`n_3`
        from the third cluster, where :math:`n_i` can be interpreted as number
        or as percentage. This can be achieved by setting:

        .. code:: python

            sampling_dictionary = {0:n_1, 1:n_2, 2:n_3}

        :param sampling_dictionary:
            dictionary specifying manual sampling. Keys are cluster classifications and
            values are either ``percentage`` or ``number`` of samples to be taken from
            that cluster. Keys should match the cluster classifications as per ``idx``.
        :param sampling_type: (optional)
            string specifying whether percentage or number is given in the
            ``sampling_dictionary``. Available options: ``percentage`` or ``number``.
            The default is ``percentage``.
        :param test_selection_option: (optional)
            option for how the test data is selected.
            Select ``test_selection_option=1`` if you want all remaining samples
            to become test data.
            Select ``test_selection_option=2`` if you want to select a subset
            of the remaining samples as test data.

        :raises ValueError:
            if ``sampling_type`` is not ``'percentage'`` or ``'number'``.

        :raises ValueError:
            if the number of entries in ``sampling_dictionary`` does not match the
            number of clusters specified in ``idx``.

        :raises ValueError:
            if any ``value`` in ``sampling_dictionary`` is not a number between 0-100 when ``sampling_type='percentage'``.

        :raises ValueError:
            if any ``value`` in ``sampling_dictionary`` is not a non-negative integer when ``sampling_type='number'``.

        :raises ValueError:
            if ``test_selection_option`` is not equal to 1 or 2.

        :raises ValueError:
            if the perecentage specified is too high in combination with the
            user-provided ``idx_test`` vector and there aren't enough samples to
            select as train data.

        :raises ValueError:
            if the number specified is too high in combination with the
            user-provided ``idx_test`` vector and there aren't enough samples to
            select as train data.

        :return:
            - **idx_train** - indices of the train data.
            - **idx_test** - indices of the test data.
        """

        # Check that sampling_type is passed correctly:
        _sampling_type = ['percentage', 'number']
        if sampling_type not in _sampling_type:
            raise ValueError("Variable `sampling_type` has to be one of the following: 'percentage' or 'number'.")

        # Degrade clusters if needed:
        if (len(np.unique(self.idx)) != (np.max(self.idx)+1)) or (np.min(self.idx) != 0):
            (self.idx, _) = degrade_clusters(self.idx, verbose=False)

        # Check that dictionary has consistend number of entries with respect to `idx`:
        if len(np.unique(self.idx)) != len(sampling_dictionary.keys()):
            raise ValueError("The number of entries inside `sampling_dictionary` does not match the number of clusters specified in `idx`.")

        # Check that keys and values are properly defined:
        for key, value in sampling_dictionary.items():

            # Check that all keys are present in the `idx`:
            if key not in np.unique(self.idx):
                raise ValueError("Key " + str(key) + " does not match an entry in `idx`.")

            # Check that keys are non-negative integers:
            if not (isinstance(key, int) and key >= 0):
                raise ValueError("Error in cluster " + str(key) + ". Key must be a non-negative integer.")

            # Check that percentage is between 0 and 100:
            if sampling_type == 'percentage':
                if not (value >= 0 and value <= 100):
                    raise ValueError("Error in cluster " + str(key) + ". The percentage has to be between 0-100.")

            # Check that number is a non-negative integer:
            if sampling_type == 'number':
                if not (isinstance(value, int) and value >= 0):
                    raise ValueError("Error in cluster " + str(key) + ". The number must be a non-negative integer.")

        # Check that `test_selection_option` parameter was passed correctly:
        _test_selection_option = [1,2]
        if test_selection_option not in _test_selection_option:
            raise ValueError("Test selection option can only be 1 or 2.")

        # Initialize vector of indices 0..n_observations:
        n_observations = len(self.idx)
        idx_full = np.arange(0,n_observations)
        idx_test = np.unique(np.array(self.idx_test))
        idx_full_no_test = np.setdiff1d(idx_full, idx_test)

        # Find the number of clusters:
        k = np.size(np.unique(self.idx))

        # Initialize auxiliary variables:
        idx_train = []

        # Get cluster populations:
        cluster_populations = get_populations(self.idx)

        # Sampling the user-specified percentage of observations:
        if sampling_type == 'percentage':

            # Get clusters and split them into training and test indices:
            for key, value in sampling_dictionary.items():

                if self.random_seed != None:
                    random.seed(self.random_seed)

                cluster = []
                for i, id in enumerate(self.idx[idx_full_no_test]):
                    if id == key:
                        cluster.append(idx_full_no_test[i])

                # Selection of training data:
                if int(cluster_populations[key]*value/100) <= len(cluster):
                    cluster_train = np.array(random.sample(cluster, int(cluster_populations[key]*value/100)))
                else:
                    raise ValueError("The training percentage specified is too high, there aren't enough samples.")

                idx_train = np.concatenate((idx_train, cluster_train))

                if self.__using_user_defined_idx_test==False:

                    # Selection of test data - all data that remains is test data:
                    if test_selection_option == 1:

                        cluster_test = np.setdiff1d(cluster, cluster_train)
                        idx_test = np.concatenate((idx_test, cluster_test))

                    if test_selection_option == 2:

                        if value > 50:
                            raise ValueError("Percentage in cluster " + str(key) + " is larger than 50% and test samples cannot be selected with `test_selection_option=2`.")

                        cluster_test = np.setdiff1d(cluster, cluster_train)
                        cluster_test_sampled = np.array(random.sample(list(cluster_test), int(len(cluster)*value/100)))
                        idx_test = np.concatenate((idx_test, cluster_test_sampled))

        # Sampling the user-specified number of observations:
        if sampling_type == 'number':

            # Get clusters and split them into training and test indices:
            for key, value in sampling_dictionary.items():

                if self.random_seed != None:
                    random.seed(self.random_seed)

                cluster = []
                for i, id in enumerate(self.idx[idx_full_no_test]):
                    if id == key:
                        cluster.append(idx_full_no_test[i])

                if value > len(cluster):
                    raise ValueError("Error in cluster " + str(key) + ". The number of samples in `sampling_dictionary` cannot exceed the number of observations in a cluster.")

                # Selection of training data:
                cluster_train = np.array(random.sample(cluster, value))
                idx_train = np.concatenate((idx_train, cluster_train))

                if self.__using_user_defined_idx_test==False:

                    # Selection of test data - all data that remains is test data:
                    if test_selection_option == 1:

                        cluster_test = np.setdiff1d(cluster, cluster_train)
                        idx_test = np.concatenate((idx_test, cluster_test))

                    if test_selection_option == 2:

                        if value > int(cluster_populations[key]*0.5):
                            raise ValueError("Number of samples in cluster " + str(key) + " is larger than 50% and test samples cannot be selected with `test_selection_option=2`.")

                        cluster_test = np.setdiff1d(cluster, cluster_train)
                        cluster_test_sampled = np.array(random.sample(list(cluster_test), value))
                        idx_test = np.concatenate((idx_test, cluster_test_sampled))

        idx_train = np.sort(idx_train.astype(int))
        idx_test = np.sort(idx_test.astype(int))

        # Print detailed information on sampling:
        if self.verbose == True:
            self.__print_verbose_information_sampling(self.idx, idx_train, idx_test)

        return (idx_train, idx_test)

    def random(self, perc, test_selection_option=1):
        """
        This function samples train data at random from the entire data set.

        **Example:**

        .. code:: python

          from PCAfold import DataSampler
          import numpy as np

          # Generate dummy idx vector:
          idx = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1])

          # Instantiate DataSampler class object:
          selection = DataSampler(idx, verbose=True)

          # Generate sampling:
          (idx_train, idx_test) = selection.random(20, test_selection_option=1)

        Due to the nature of this sampling technique, it is not necessary to
        have ``idx`` classifications since random samples can also be selected
        from unclassified data sets. You can achieve that by generating a dummy
        ``idx`` vector that has the same number of observations
        ``n_observations`` as your data set. For instance:

        .. code:: python

          from PCAfold import DataSampler
          import numpy as np

          # Generate dummy idx vector:
          n_observations = 100
          idx = np.zeros(n_observations)

          # Instantiate DataSampler class object:
          selection = DataSampler(idx)

          # Generate sampling:
          (idx_train, idx_test) = selection.random(20, test_selection_option=1)

        **Train data:**

        The total number of train samples is computed as a percentage ``perc``
        from the total number of observations in a data set. These samples are
        then drawn at random from the entire data set, independent of cluster
        classifications.

        **Test data:**

        Two options for sampling test data are implemented. If you select
        ``test_selection_option=1`` all remaining samples that were not taken
        as train data become the test data. If you select
        ``test_selection_option=2`` the same procedure is used to select
        test data as was used to select train data
        (only allowed if ``perc`` is less than 50%).

        The scheme below presents graphically how train and test data can be
        selected using ``test_selection_option`` parameter:

        .. image:: ../images/sampling-test-selection-option-random.png
          :width: 700
          :align: center

        Here :math:`p` is the percentage ``perc`` provided.

        :param perc:
            percentage of data to be selected as training data from each cluster.
            Set ``perc=20`` if you want 20%.
        :param test_selection_option: (optional)
            option for how the test data is selected.
            Select ``test_selection_option=1`` if you want all remaining samples
            to become test data.
            Select ``test_selection_option=2`` if you want to select a subset
            of the remaining samples as test data.

        :raises ValueError:
            if ``perc`` is not a number between 0-100.

        :raises ValueError:
            if ``test_selection_option`` is not equal to 1 or 2.

        :raises ValueError:
            if the perecentage specified is too high in combination with the
            user-provided ``idx_test`` vector and there aren't enough samples to
            select as train data.

        :return:
            - **idx_train** - indices of the train data.
            - **idx_test** - indices of the test data.
        """

        # Check if `perc` parameter was passed correctly:
        if (perc < 0) or (perc > 100):
            raise ValueError("Percentage has to be between 0-100.")

        # Check that `test_selection_option` parameter was passed correctly:
        _test_selection_option = [1,2]
        if test_selection_option not in _test_selection_option:
            raise ValueError("Test selection option can only be 1 or 2.")

        # Degrade clusters if needed:
        if (len(np.unique(self.idx)) != (np.max(self.idx)+1)) or (np.min(self.idx) != 0):
            (self.idx, _) = degrade_clusters(self.idx, verbose=False)

        # Initialize vector of indices 0..n_observations:
        n_observations = len(self.idx)
        idx_full = np.arange(0, n_observations)
        idx_test = np.unique(np.array(self.idx_test))
        idx_full_no_test = np.setdiff1d(idx_full, idx_test)

        # Find the number of clusters:
        k = np.size(np.unique(self.idx))

        if self.random_seed != None:
            random.seed(self.random_seed)

        if self.__using_user_defined_idx_test==True:

            if int(len(idx_full)*perc/100) <= len(idx_full_no_test):
                idx_train = np.array(random.sample(idx_full_no_test.tolist(), int(len(idx_full)*perc/100)))
            else:
                raise ValueError("The training percentage specified is too high, there aren't enough samples.")

        else:

            if test_selection_option==1:

                idx_train = np.array(random.sample(idx_full.tolist(), int(len(idx_full)*perc/100)))
                idx_test = np.setdiff1d(idx_full, idx_train)

            elif test_selection_option==2:

                idx_train = np.array(random.sample(idx_full.tolist(), int(len(idx_full)*perc/100)))

                # Check if there is enough test samples to select:
                if perc > 50:
                    raise ValueError("Percentage is larger than 50% and test samples cannot be selected with `test_selection_option=2`.")
                else:
                    test_pool = np.setdiff1d(idx_full, idx_train)
                    idx_test = np.array(random.sample(list(test_pool), int(len(idx_full)*perc/100)))

        idx_train = np.sort(idx_train.astype(int))
        idx_test = np.sort(idx_test.astype(int))

        # Print detailed information on sampling:
        if self.verbose == True:
            self.__print_verbose_information_sampling(self.idx, idx_train, idx_test)

        return (idx_train, idx_test)

################################################################################
#
# Data Clustering
#
################################################################################

def __print_verbose_information_clustering(var, idx, bins_borders):

    k = len(np.unique(idx))

    print('Border values for bins:')
    print(bins_borders)
    print('')

    for cl_id in range(0,k):
        print("Bounds for cluster " + str(cl_id) + ":")
        print("\t" + str(round(np.min(var[np.argwhere(idx==cl_id)]), 4)) + ", " + str(round(np.max(var[np.argwhere(idx==cl_id)]), 4)))

def variable_bins(var, k, verbose=False):
    """
    This function does clustering by dividing a variable vector ``var`` into
    bins of equal lengths.

    An example of how a vector can be partitioned with this function is presented below:

    .. image:: ../images/clustering-variable-bins.png
      :width: 600
      :align: center

    **Example:**

    .. code::

        from PCAfold import variable_bins
        import numpy as np

        # Generate dummy variable:
        x = np.linspace(-1,1,100)

        # Create partitioning according to bins of x:
        idx = variable_bins(x, 4, verbose=True)

    :param var:
        vector of variable values.
    :param k:
        number of clusters to partition the data.
    :param verbose: (optional)
        boolean for printing clustering details.

    :raises ValueError:
        if number of clusters ``k`` is not a positive integer.

    :return:
        - **idx** - vector of cluster classifications.
    """

    # Check that the number of clusters is an integer and is non-zero:
    if not (isinstance(k, int) and k > 0):
        raise ValueError("The number of clusters must be a positive integer.")

    var_min = np.min(var)
    var_max = np.max(var)
    bin_length = (var_max - var_min)/k
    idx = []
    bins_borders = []

    for cl in range(0,k+1):
        if cl == k+1:
            bins_borders.append(var_max)
        else:
            bins_borders.append(cl*bin_length + var_min)

    # Split into bins of variable:
    for val in var:
        if val == var_max:
            idx.append(k-1)
        else:
            for cl in range(0,k):
                if (val >= bins_borders[cl]) and (val < bins_borders[cl+1]):
                    idx.append(cl)

    # Degrade clusters if needed:
    if (len(np.unique(idx)) != (np.max(idx)+1)) or (np.min(idx) != 0):
        (idx, _) = degrade_clusters(idx, verbose)

    idx = np.asarray(idx)

    if verbose==True:
        __print_verbose_information_clustering(var, idx, bins_borders)

    return(idx)

def predefined_variable_bins(var, split_values, verbose=False):
    """
    This function does clustering by dividing a variable vector ``var`` into
    bins such that the split is done at values specified in the ``split_values``
    list. In general: ``split_values = [value_1, value_2, ..., value_n]``.

    *Note:*
    When a split is performed at a given ``value_i``, the observation in ``var``
    that takes exactly that value is assigned to the newly created bin.

    An example of how a vector can be partitioned with this function is presented below:

    .. image:: ../images/clustering-predefined-variable-bins.png
      :width: 600
      :align: center

    **Example:**

    .. code::

        from PCAfold import predefined_variable_bins
        import numpy as np

        # Generate dummy variable:
        x = np.linspace(-1,1,100)

        # Create partitioning according to pre-defined bins of x:
        idx = predefined_variable_bins(x, [-0.6, 0.4, 0.8], verbose=True)

    :param var:
        vector of variable values.
    :param split_values:
        list containing values at which the split to bins should be performed.
    :param verbose: (optional)
        boolean for printing clustering details.

    :raises ValueError:
        if any value within ``split_values`` is not within the range of
        vector ``var`` values.

    :return:
        - **idx** - vector of cluster classifications.
    """

    var_min = np.min(var)
    var_max = np.max(var)

    # Check that all values specified in `split_values` fall within the range of
    # the variable `var`:
    for value in split_values:
        if value < var_min or value > var_max:
            raise ValueError("Value " + str(value) + " is not within the range of the variable values.")

    idx = []
    bins_borders = copy.deepcopy(split_values)
    bins_borders.insert(0, var_min)
    bins_borders.append(var_max)

    k = len(bins_borders) - 1

    # Split into bins of variable:
    for val in var:
        if val == var_max:
            idx.append(k-1)
        else:
            for cl in range(0,k):
                if (val >= bins_borders[cl]) and (val < bins_borders[cl+1]):
                    idx.append(cl)

    # Degrade clusters if needed:
    if (len(np.unique(idx)) != (np.max(idx)+1)) or (np.min(idx) != 0):
        (idx, _) = degrade_clusters(idx, verbose)

    idx = np.asarray(idx)

    if verbose==True:
        __print_verbose_information_clustering(var, idx, bins_borders)

    return(idx)

def mixture_fraction_bins(Z, k, Z_stoich, verbose=False):
    """
    This function does clustering by dividing a mixture fraction vector
    ``Z`` into bins of equal lengths. The vector is first split to lean and rich
    side (according to the stoichiometric mixture fraction ``Z_stoich``) and
    then the sides get divided further into clusters. When ``k`` is even,
    this function will always create equal number of clusters on the lean and
    rich side. When ``k`` is odd, there will be one more cluster on the rich side
    compared to the lean side.

    An example of how a vector can be partitioned with this function is presented below:

    .. image:: ../images/clustering-mixture-fraction-bins.png
      :width: 600
      :align: center

    **Example:**

    .. code::

        from PCAfold import mixture_fraction_bins
        import numpy as np

        # Generate dummy mixture fraction variable:
        Z = np.linspace(0,1,100)

        # Create partitioning according to bins of mixture fraction:
        idx = mixture_fraction_bins(Z, 4, 0.4, verbose=True)

    :param Z:
        vector of mixture fraction values.
    :param k:
        number of clusters to partition the data.
    :param Z_stoich:
        stoichiometric mixture fraction.
    :param verbose: (optional)
        boolean for printing clustering details.

    :raises ValueError:
        if number of clusters ``k`` is not a positive integer.

    :return:
        - **idx** - vector of cluster classifications.
    """

    # Check that the number of clusters is an integer and is non-zero:
    if not (isinstance(k, int) and k > 0):
        raise ValueError("The number of clusters must be a positive integer.")

    # Number of interval borders:
    n_bins_borders = k + 1

    # Minimum and maximum mixture fraction:
    min_Z = np.min(Z)
    max_Z = np.max(Z)

    # Partition the Z-space:
    if k == 1:

        borders = np.linspace(min_Z, max_Z, n_bins_borders)

    else:

        # Z-space lower than stoichiometric mixture fraction:
        borders_lean = np.linspace(min_Z, Z_stoich, int(np.ceil(n_bins_borders/2)))

        # Z-space higher than stoichiometric mixture fraction:
        borders_rich = np.linspace(Z_stoich, max_Z, int(np.ceil((n_bins_borders+1)/2)))

        # Combine the two partitions:
        borders = np.concatenate((borders_lean[0:-1], borders_rich))

    # Bin data matrices initialization:
    idx_clust = []
    idx = np.zeros((len(Z),))

    # Create the cluster division vector:
    for bin in range(0,k):

        idx_clust.append(np.where((Z >= borders[bin]) & (Z <= borders[bin+1])))
        idx[idx_clust[bin]] = bin+1

    idx = [int(i) for i in idx]

    # Degrade clusters if needed:
    if (len(np.unique(idx)) != (np.max(idx)+1)) or (np.min(idx) != 0):
        (idx, _) = degrade_clusters(idx, verbose=False)

    idx = np.asarray(idx)

    if verbose==True:
        __print_verbose_information_clustering(Z, idx, borders)

    return(idx)

def zero_neighborhood_bins(var, k, zero_offset_percentage=0.1, split_at_zero=False, verbose=False):
    """
    This function aims to separate close-to-zero observations in a vector into one
    cluster (``split_at_zero=False``) or two clusters (``split_at_zero=True``).
    It can be useful for partitioning any variable
    that has many observations clustered around zero value and relatively few
    observations far away from zero on either side.

    The offset from zero at which splits are performed is computed
    based on the input parameter ``zero_offset_percentage``:

    .. math::

        \\verb|offset| = \\frac{(max(\\verb|var|) - min(\\verb|var|)) \cdot \\verb|zero_offset_percentage|}{100}

    Further clusters are found by splitting positive and negative values in a vector
    alternatingly into bins of equal lengths.

    Two examples of how a vector can be partitioned with this function are presented below.

    With ``split_at_zero=False``:

    .. image:: ../images/clustering-zero-neighborhood-bins.png
      :width: 700
      :align: center

    If ``split_at_zero=False`` the smallest allowed number of clusters is 3.
    This is to assure that there are at least three clusters:
    with negative values, with close to zero values, with positive values.

    With ``split_at_zero=True``:

    .. image:: ../images/clustering-zero-neighborhood-bins-zero-split.png
      :width: 700
      :align: center

    If ``split_at_zero=True`` the smallest allowed number of clusters is 4.
    This is to assure that there are at least four clusters: with negative
    values, with negative values close to zero, with positive values close to
    zero and with positive values.

    **Example:**

    .. code::

        from PCAfold import zero_neighborhood_bins
        import numpy as np

        # Generate dummy variable:
        x = np.linspace(-100,100,1000)

        # Create partitioning according to bins of x:
        idx = zero_neighborhood_bins(x, 4, zero_offset_percentage=10, split_at_zero=True, verbose=True)

    :param var:
        vector of variable values.
    :param k:
        number of clusters to partition the data.
        Cannot be smaller than 3 if ``split_at_zero=False`` or smaller
        than 4 if ``split_at_zero=True``.
    :param zero_offset_percentage: (optional)
        percentage of :math:`max(\\verb|var|) - min(\\verb|var|)` range
        to take as the offset from zero value. For instance, set
        ``zero_offset_percentage=10`` if you want 10% as offset.
    :param split_at_zero: (optional)
        boolean specifying whether partitioning should be done at ``var=0``.
    :param verbose: (optional)
        boolean for printing clustering details.

    :raises ValueError:
        if number of clusters ``k`` is not an integer or smaller than 3 when
        ``split_at_zero=False`` or smaller than 4 when ``split_at_zero=True``.

    :raises ValueError:
        if the vector ``var`` has only non-negative or only
        non-positive values. For such vectors it is recommended to use
        ``predefined_variable_bins`` function instead.

    :raises ValueError:
        if the requested offset from zero crosses the minimum or maximum value
        of the variable vector ``var``. If that is the case, it is
        recommended to lower the ``zero_offset_percentage`` value.

    :return:
        - **idx** - vector of cluster classifications.
    """

    # Check that the number of clusters is an integer and is larger than 2:
    if (not split_at_zero) and (not (isinstance(k, int) and k > 2)):
        raise ValueError("The number of clusters must be an integer not smaller than 3 when not splitting at zero.")

    # Check that the number of clusters is an integer and is larger than 2:
    if split_at_zero and (not (isinstance(k, int) and k > 3)):
        raise ValueError("The number of clusters must be an integer not smaller than 4 when splitting at zero.")

    var_min = np.min(var)
    var_max = np.max(var)
    var_range = abs(var_max - var_min)
    offset = zero_offset_percentage * var_range / 100

    # Basic checks on the variable vector:
    if not (var_min <= 0):
        raise ValueError("Source vector does not have negative values. Use `predefined_variable_bins` as a clustering technique instead.")

    if not (var_max >= 0):
        raise ValueError("Source vector does not have positive values. Use `predefined_variable_bins` as a clustering technique instead.")

    if (var_min > -offset) or (var_max < offset):
        raise ValueError("Offset from zero crosses the minimum or maximum value of the variable vector. Consider lowering `zero_offset_percentage`.")

    # Number of interval borders:
    if split_at_zero:
        n_bins_borders = k-1
    else:
        n_bins_borders = k

    # Generate cluster borders on the negative side:
    borders_negative = np.linspace(var_min, -offset, int(np.ceil(n_bins_borders/2)))

    # Generate cluster borders on the positive side:
    borders_positive = np.linspace(offset, var_max, int(np.ceil((n_bins_borders+1)/2)))

    # Combine the two partitions:
    if split_at_zero:
        borders = np.concatenate((borders_negative, np.array([0]), borders_positive))
    else:
        borders = np.concatenate((borders_negative, borders_positive))

    # Bin data matrices initialization:
    idx_clust = []
    idx = np.zeros((len(var),))

    # Create the cluster division vector:
    for bin in range(0,k):

        idx_clust.append(np.where((var >= borders[bin]) & (var <= borders[bin+1])))
        idx[idx_clust[bin]] = bin+1

    idx = np.asarray([int(i) for i in idx])

    # Degrade clusters if needed:
    if (len(np.unique(idx)) != (np.max(idx)+1)) or (np.min(idx) != 0):
        (idx, _) = degrade_clusters(idx, verbose=False)

    if verbose==True:
        __print_verbose_information_clustering(var, idx, borders)

    return(idx)

def degrade_clusters(idx, verbose=False):
    """
    This function renumerates clusters if either of these two cases is true:

    - ``idx`` is composed of non-consecutive integers, or

    - the smallest cluster number in ``idx`` is not equal to ``0``.

    **Example:**

    .. code:: python

        from PCAfold import degrade_clusters

        # Generate dummy idx vector:
        idx = [0, 0, 2, 0, 5, 10]

        # Degrade clusters:
        (idx_degraded, k_update) = degrade_clusters(idx)

    The code above will produce:

    .. code-block:: text

        >>> idx_degraded
        array([0, 0, 1, 0, 2, 3])

    Alternatively:

    .. code:: python

        from PCAfold import degrade_clusters

        # Generate dummy idx vector:
        idx = [1, 1, 2, 2, 3, 3]

        # Degrade clusters:
        (idx_degraded, k_update) = degrade_clusters(idx)

    will produce:

    .. code-block:: text

        >>> idx_degraded
        array([0, 0, 1, 1, 2, 2])

    :param idx:
        vector of cluster classifications.
    :param verbose: (optional)
        boolean for printing clustering details.

    :raises ValueError:
        if ``idx`` vector contains entries other than integers, is not a list or ``numpy.ndarray``.

    :return:
        - **idx_degraded** degraded vector of cluster classifications. The first cluster has index 0.
        - **k_update** - the updated number of clusters.
    """

    if isinstance(idx, list):
        if not all(isinstance(i, int) for i in idx) or any(isinstance(i, bool) for i in idx):
            raise ValueError("Vector of cluster classifications can only contain integers.")
    elif isinstance(idx, np.ndarray):
        if not all(isinstance(i, np.integer) for i in idx):
            raise ValueError("Vector of cluster classifications can only contain integers.")
    else:
        raise ValueError("Vector of cluster classifications should be a list or numpy.ndarray.")

    index = 0
    dictionary = {}
    sorted = np.unique(idx)
    k_init = np.max(idx) + 1
    dictionary[sorted[0]] = 0
    old_val = sorted[0]
    idx_degraded = [0 for i in range(0, len(idx))]

    for val in sorted:
        if val > old_val:
            index += 1
        dictionary[val] = index

    for i, val in enumerate(idx):
        idx_degraded[i] = dictionary[val]

    k_update = np.max(idx_degraded) + 1

    if verbose == True:
        print('Clusters have been degraded.')
        print('The number of clusters have been reduced from ' + str(k_init) + ' to ' + str(k_update) + '.')

    return (np.asarray(idx_degraded), k_update)

def flip_clusters(idx, dictionary):
    """
    This function flips cluster labelling according to instructions provided
    in the dictionary. For a ``dictionary = {key : value}``, a cluster with a
    number ``key`` will get a number ``value``.

    **Example:**

    .. code:: python

        from PCAfold import flip_clusters

        # Generate dummy idx vector:
        idx = [0,0,0,1,1,1,1,2,2]

        # Swap cluster number 1 with cluster number 2:
        flipped_idx = flip_clusters(idx, {1:2, 2:1})

    The code above will produce:

    .. code-block:: text

        >>> flipped_idx
        array([0, 0, 0, 2, 2, 2, 2, 1, 1])

    :param idx:
        vector of cluster classifications.
    :param dictionary:
        dictionary specifying instructions for cluster label flipping.

    :raises ValueError:
        if ``idx`` vector contains entries other than integers, is not a list or ``numpy.ndarray``.

    :raises ValueError:
        if any ``key`` or ``value`` is not an integer.

    :raises ValueError:
        if any ``key`` is not found within ``idx``.

    :return:
        - **flipped_idx** - re-labelled vector of cluster classifications.
    """

    if isinstance(idx, list):
        if not all(isinstance(i, int) for i in idx) or any(isinstance(i, bool) for i in idx):
            raise ValueError("Vector of cluster classifications can only contain integers.")
    elif isinstance(idx, np.ndarray):
        if not all(isinstance(i, np.integer) for i in idx):
            raise ValueError("Vector of cluster classifications can only contain integers.")
    else:
        raise ValueError("Vector of cluster classifications should be a list or numpy.ndarray.")

    # Check that keys and values are properly defined:
    for key, value in dictionary.items():

        # Check that all keys are present in the `idx`:
        if key not in np.unique(idx):
            raise ValueError("Key " + str(key) + " does not match an entry in `idx`.")

        # Check that keys are non-negative integers:
        if not isinstance(key, int):
            raise ValueError("Error in key " + str(key) + ". Key must be an integer.")

        # Check that values are non-negative integers:
        if not isinstance(value, int):
            raise ValueError("Error in value " + str(value) + ". Value must be an integer.")

    flipped_idx = []

    for i in idx:
        if i in dictionary.keys():
            flipped_idx.append(dictionary[i])
        else:
            flipped_idx.append(i)

    return(np.asarray(flipped_idx))

def get_centroids(X, idx):
    """
    This function computes the centroids for the clustering specified in the
    ``idx`` vector.

    :param X:
        data set for computing the cluster centroids.
    :param idx:
        vector of cluster classifications.

    :raises ValueError:
        if the number of observations in the data set ``X`` does not match the
        number of elements in the ``idx`` vector.

    :return:
        - **centroids** - matrix of cluster centroids. It has size ``k`` times number of observations.
    """

    # Degrade clusters if needed:
    if (len(np.unique(idx)) != (np.max(idx)+1)) or (np.min(idx) != 0):
        (idx, _) = degrade_clusters(idx, verbose=False)

    (n_observations, n_variables) = np.shape(X)

    # Check if the number of indices in `idx` is the same as the number of observations in a data set:
    if n_observations != len(idx):
        raise ValueError("The number of observations in the data set `X` must match the number of elements in `idx` vector.")

    # Find the number of clusters:
    k = len(np.unique(idx))

    # Initialize the centroids matrix:
    centroids = np.zeros((k, n_variables))

    # Compute the centroids:
    for i in range(0,k):
        indices = [ind for ind, e in enumerate(idx) if e == i]
        centroids[i, :] = np.mean(X[indices,:], axis=0)

    return(centroids)

def get_partition(X, idx, verbose=False):
    """
    This function partitions the observations from the original global data
    set ``X`` into local clusters according to ``idx`` provided. It returns a
    tuple of three variables ``(data_in_clusters, data_idx_in_clusters, k_new)``,
    where ``data_in_clusters`` are the original observations from ``X`` divided
    into clusters, ``data_idx_in_clusters`` are the indices of the original
    observations divided into clusters. If any cluster is empty or has less
    observations assigned to it that the number of variables, that cluster will
    be removed and the observations that were assigned to it will not appear
    in ``data_in_clusters`` nor in ``data_idx_in_clusters``. The new number of
    clusters ``k_new`` is computed taking into account any possibly removed
    clusters.

    :param X:
        data set to partition.
    :param idx:
        vector of cluster classifications.
        The first cluster has index 0.
    :param verbose: (optional)
        boolean for printing details.

    :return:
        - **data_in_clusters** - list of ``k_new`` arrays that contains original data set observations in each cluster.
        - **data_idx_in_clusters** - list of ``k_new`` arrays that contains indices of the original data set observations in each cluster.
        - **k_new** - the updated number of clusters.
    """

    try:
        (n_observations, n_variables) = np.shape(X)
    except:
        (n_observations, ) = np.shape(X)
        n_variables = 1

    # Remove empty clusters from indexing:
    if (len(np.unique(idx)) != (np.max(idx)+1)) or (np.min(idx) != 0):
        (idx, _) = degrade_clusters(idx, verbose)
        if verbose==True:
            print('Empty clusters will be removed.')

    k = len(np.unique(idx))

    idx_clust = []
    n_points = np.zeros(k)
    data_in_clusters = []
    data_idx_in_clusters = []

    for i in range(0,k):

        indices_to_append = np.argwhere(idx==i).ravel()
        idx_clust.append(indices_to_append)
        n_points[i] = len(indices_to_append)

        if ((n_points[i] < n_variables) and (n_points[i] > 0)):
            if verbose==True:
                print('Too few points (' + str(int(n_points[i])) + ') in cluster ' + str(i) + ', cluster will be removed.')

    # Find those cluster numbers where the number of observations is not less than number of variables:
    nz_idx = np.argwhere(n_points >= n_variables).ravel()

    # Compute the new number of clusters taking into account removed clusters:
    k_new = len(nz_idx)

    for i in range(0,k_new):

        # Assign observations to clusters:
        data_idx_in_clusters.append(idx_clust[nz_idx[i]])
        data_in_clusters.append(X[data_idx_in_clusters[i],:])

    return(data_in_clusters, data_idx_in_clusters, k_new)

def get_populations(idx, verbose=False):
    """
    This function computes populations (number of observations) in clusters
    specified in the ``idx`` vector. As an example, if there are 100
    observations in the first cluster and 500 observations in the second cluster
    this function will return a list: ``[100, 500]``.

    **Example:**

    .. code::

        from PCAfold import variable_bins, get_populations
        import numpy as np

        # Generate dummy partitioning:
        x = np.linspace(-1,1,100)
        idx = variable_bins(x, 4, verbose=True)

        # Compute cluster populations:
        populations = get_populations(idx)

    The code above will produce:

    .. code-block:: text

        >>> populations
        [25, 25, 25, 25]

    :param idx:
        vector of cluster classifications.
        The first cluster has index 0.
    :param verbose: (optional)
        boolean for printing details.

    :return:
        - **populations** - list of cluster populations. Each entry referes to one cluster ordered according to ``idx``.
    """

    populations = []

    # Degrade clusters if needed:
    if (len(np.unique(idx)) != (np.max(idx)+1)) or (np.min(idx) != 0):
        (idx, _) = degrade_clusters(idx, verbose)

    # Find the number of clusters:
    k = len(np.unique(idx))

    for i in range(0,k):

        populations.append(int((idx==i).sum()))

    return(populations)

################################################################################
#
# Plotting functions
#
################################################################################

def plot_2d_clustering(x, y, idx, x_label=None, y_label=None, color_map='viridis', first_cluster_index_zero=True, grid_on=False, figure_size=(7,7), title=None, save_filename=None):
    """
    This function plots a 2-dimensional manifold divided into clusters.
    Number of observations in each cluster will be plotted in the legend.

    :param x:
        variable on the :math:`x`-axis.
    :param y:
        variable on the :math:`y`-axis.
    :param idx:
        vector of cluster classifications.
    :param x_label: (optional)
        string specifying :math:`x`-axis label annotation. If set to ``None``
        label will not be plotted.
    :param y_label: (optional)
        string specifying :math:`y`-axis label annotation. If set to ``None``
        label will not be plotted.
    :param color_map: (optional)
        colormap to use as per ``matplotlib.cm``. Default is *viridis*.
    :param first_cluster_index_zero: (optional)
        boolean specifying if the first cluster should be indexed ``0`` on the plot.
        If set to ``False`` the first cluster will be indexed ``1``.
    :param grid_on:
        boolean specifying whether grid should be plotted.
    :param figure_size:
        tuple specifying figure size.
    :param title: (optional)
        string specifying plot title. If set to ``None`` title will not be
        plotted.
    :param save_filename: (optional)
        string specifying plot save location/filename. If set to ``None``
        plot will not be saved.

    :return:
        - **plt** - plot handle.
    """

    from matplotlib import cm

    n_clusters = len(np.unique(idx))
    populations = get_populations(idx, verbose=False)

    x = x.ravel()
    y = y.ravel()

    color_map_colors = cm.get_cmap(color_map, n_clusters)
    cluster_colors = color_map_colors(np.linspace(0, 1, n_clusters))

    figure = plt.figure(figsize=figure_size)

    for k in range(0,n_clusters):
        if first_cluster_index_zero:
            plt.scatter(x[np.where(idx==k)], y[np.where(idx==k)], color=cluster_colors[k], marker='o', s=scatter_point_size, label='$k_{' + str(k) + '}$ - ' + str(populations[k]))
        else:
            plt.scatter(x[np.where(idx==k)], y[np.where(idx==k)], color=cluster_colors[k], marker='o', s=scatter_point_size, label='$k_{' + str(k+1) + '}$ - ' + str(populations[k]))

    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), fancybox=True, shadow=True, ncol=4, fontsize=font_legend, markerscale=marker_scale_legend_clustering)

    plt.xticks(fontsize=font_axes, **csfont), plt.yticks(fontsize=font_axes, **csfont)
    if x_label != None: plt.xlabel(x_label, fontsize=font_labels, **csfont)
    if y_label != None: plt.ylabel(y_label, fontsize=font_labels, **csfont)
    if grid_on: plt.grid(alpha=grid_opacity)

    if title != None: plt.title(title, fontsize=font_title, **csfont)
    if save_filename != None: plt.savefig(save_filename, dpi = 500, bbox_inches='tight')

    return plt

def plot_2d_train_test_samples(x, y, idx, idx_train, idx_test, x_label=None, y_label=None, color_map='viridis', first_cluster_index_zero=True, grid_on=False, figure_size=(14,7), title=None, save_filename=None):
    """
    This function plots a 2-dimensional manifold divided into train and test
    samples. Number of observations in train and test data respectively will be
    plotted in the legend.

    :param x:
        variable on the :math:`x`-axis.
    :param y:
        variable on the :math:`y`-axis.
    :param idx:
        vector of cluster classifications.
    :param idx_train:
        indices of the train data.
    :param idx_test:
        indices of the test data.
    :param x_label: (optional)
        string specifying :math:`x`-axis label annotation. If set to ``None``
        label will not be plotted.
    :param y_label: (optional)
        string specifying :math:`y`-axis label annotation. If set to ``None``
        label will not be plotted.
    :param color_map: (optional)
        colormap to use as per ``matplotlib.cm``. Default is *viridis*.
    :param first_cluster_index_zero: (optional)
        boolean specifying if the first cluster should be indexed ``0`` on the plot.
        If set to ``False`` the first cluster will be indexed ``1``.
    :param grid_on:
        boolean specifying whether grid should be plotted.
    :param figure_size:
        tuple specifying figure size.
    :param title: (optional)
        string specifying plot title. If set to ``None`` title will not be
        plotted.
    :param save_filename: (optional)
        string specifying plot save location/filename. If set to ``None``
        plot will not be saved.

    :return:
        - **plt** - plot handle.
    """

    from matplotlib import cm

    n_clusters = len(np.unique(idx))
    populations = get_populations(idx, verbose=False)

    x = x.ravel()
    y = y.ravel()

    color_map_colors = cm.get_cmap(color_map, n_clusters)
    cluster_colors = color_map_colors(np.linspace(0, 1, n_clusters))

    figure = plt.figure(figsize=figure_size)

    ax1 = plt.subplot(121)
    for k in range(0,n_clusters):
        train_indices = [idxt for idxt in idx_train if idx[idxt,]==k]
        if first_cluster_index_zero:
            ax1.scatter(x[train_indices], y[train_indices], color=cluster_colors[k], marker='o', s=scatter_point_size*2, label='$k_' + str(k) + '$ - ' + str(len(train_indices)))
        else:
            ax1.scatter(x[train_indices], y[train_indices], color=cluster_colors[k], marker='o', s=scatter_point_size*2, label='$k_' + str(k+1) + '$ - ' + str(len(train_indices)))
    ax1.set_xticks([]), ax1.set_yticks([])
    ax1.set_title('Train data', fontsize=font_title)
    if x_label != None: ax1.set_xlabel(x_label, fontsize=font_labels, **csfont)
    if y_label != None: ax1.set_ylabel(y_label, fontsize=font_labels, **csfont)
    if grid_on: ax1.grid(alpha=grid_opacity)
    ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=4, fontsize=font_legend/2, markerscale=marker_scale_legend_clustering/2)

    ax2 = plt.subplot(1,2,2)
    for k in range(0,n_clusters):
        test_indices = [idxt for idxt in idx_test if idx[idxt,]==k]
        if first_cluster_index_zero:
            ax2.scatter(x[test_indices], y[test_indices], color=cluster_colors[k], marker='o', s=scatter_point_size*2, label='$k_' + str(k) + '$ - ' + str(len(test_indices)))
        else:
            ax2.scatter(x[test_indices], y[test_indices], color=cluster_colors[k], marker='o', s=scatter_point_size*2, label='$k_' + str(k+1) + '$ - ' + str(len(test_indices)))
    ax2.set_xticks([]), ax2.set_yticks([])
    ax2.set_title('Test data', fontsize=font_title)
    if x_label != None: ax2.set_xlabel(x_label, fontsize=font_labels, **csfont)
    if y_label != None: ax2.set_ylabel(y_label, fontsize=font_labels, **csfont)
    if grid_on: ax2.grid(alpha=grid_opacity)
    ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=4, fontsize=font_legend/2, markerscale=marker_scale_legend_clustering/2)

    if title != None: figure.suptitle(title, fontsize=font_title, **csfont)
    if save_filename != None: figure.savefig(save_filename, dpi = 500, bbox_inches='tight')

    return plt
