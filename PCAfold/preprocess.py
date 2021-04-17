"""preprocess.py: module for data pre-processing."""

__author__ = "Kamila Zdybal, Elizabeth Armstrong, Alessandro Parente and James C. Sutherland"
__copyright__ = "Copyright (c) 2020, Kamila Zdybal and Elizabeth Armstrong"
__credits__ = ["Department of Chemical Engineering, University of Utah, Salt Lake City, Utah, USA", "Universite Libre de Bruxelles, Aero-Thermo-Mechanics Laboratory, Brussels, Belgium"]
__license__ = "MIT"
__version__ = "1.0.0"
__maintainer__ = ["Kamila Zdybal", "Elizabeth Armstrong"]
__email__ = ["kamilazdybal@gmail.com", "Elizabeth.Armstrong@chemeng.utah.edu", "James.Sutherland@chemeng.utah.edu"]
__status__ = "Production"

import numpy as np
import random
import copy
import operator
import matplotlib.pyplot as plt
from PCAfold.styles import *

################################################################################
#
# Data Manipulation
#
################################################################################

_scalings_list = ['none', '', 'auto', 'std', 'pareto', 'vast', 'range', '0to1', '-1to1', 'level', 'max', 'poisson', 'vast_2', 'vast_3', 'vast_4']

def center_scale(X, scaling, nocenter=False):
    """
    This function centers and scales the data set.

    In the discussion below we understand that :math:`\mathbf{X}_i` is the :math:`i^{th}` column
    of :math:`\mathbf{X}`.

    - Centering is performed by subtracting a center :math:`C_i` from
      :math:`\mathbf{X}_i`, where centers for all columns are stored
      in a vector :math:`\mathbf{C}`:

        .. math::

            \mathbf{X_c} = \mathbf{X} - \mathbf{C}

      Centers for each column are computed as:

        .. math::

            C_i = mean(\mathbf{X}_i)

      The only exceptions are the *0 to 1* and *-1 to 1* scalings which introduce a different
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

    +----------------------------+--------------------------+--------------------------------------------------------------------+
    | Scaling method             | ``scaling``              | Scaling factor :math:`d_i`                                         |
    +============================+==========================+====================================================================+
    | None                       | ``'none'``               | 1                                                                  |
    +----------------------------+--------------------------+--------------------------------------------------------------------+
    | Auto                       | ``'auto'`` or ``'std'``  | :math:`\sigma`                                                     |
    +----------------------------+--------------------------+--------------------------------------------------------------------+
    | Pareto :cite:`Noda2008`    | ``'pareto'``             | :math:`\sqrt{\sigma}`                                              |
    +----------------------------+--------------------------+--------------------------------------------------------------------+
    | VAST :cite:`Keun2003`      | ``'vast'``               | :math:`\sigma^2 / mean(\mathbf{X}_i)`                              |
    +----------------------------+--------------------------+--------------------------------------------------------------------+
    | Range                      | ``'range'``              | :math:`max(\mathbf{X}_i) - min(\mathbf{X}_i)`                      |
    +----------------------------+--------------------------+--------------------------------------------------------------------+
    | | 0 to 1                   | | ``'0to1'``             | | :math:`d_i = max(\mathbf{X}_i) - min(\mathbf{X}_i)`              |
    | |                          | |                        | | :math:`C_i = min(\mathbf{X}_i)`                                  |
    +----------------------------+--------------------------+--------------------------------------------------------------------+
    | | -1 to 1                  | | ``'-1to1'``            | | :math:`d_i = 0.5 \cdot (max(\mathbf{X}_i) - min(\mathbf{X}_i))`  |
    | |                          | |                        | | :math:`C_i = 0.5 \cdot (max(\mathbf{X}_i) + min(\mathbf{X}_i))`  |
    +----------------------------+--------------------------+--------------------------------------------------------------------+
    | Level                      | ``'level'``              | :math:`mean(\mathbf{X}_i)`                                         |
    +----------------------------+--------------------------+--------------------------------------------------------------------+
    | Max                        | ``'max'``                | :math:`max(\mathbf{X}_i)`                                          |
    +----------------------------+--------------------------+--------------------------------------------------------------------+
    | Poisson :cite:`Keenan2004` |``'poisson'``             | :math:`\sqrt{mean(\mathbf{X}_i)}`                                  |
    +----------------------------+--------------------------+--------------------------------------------------------------------+
    | Vast-2                     | ``'vast_2'``             | :math:`\sigma^2 k^2 / mean(\mathbf{X}_i)`                          |
    +----------------------------+--------------------------+--------------------------------------------------------------------+
    | Vast-3                     | ``'vast_3'``             | :math:`\sigma^2 k^2 / max(\mathbf{X}_i)`                           |
    +----------------------------+--------------------------+--------------------------------------------------------------------+
    | Vast-4                     | ``'vast_4'``             | :math:`\sigma^2 k^2 / (max(\mathbf{X}_i) - min(\mathbf{X}_i))`     |
    +----------------------------+--------------------------+--------------------------------------------------------------------+

    where :math:`\sigma` is the standard deviation of :math:`\mathbf{X}_i`
    and :math:`k` is the kurtosis of :math:`\mathbf{X}_i`.

    The effect of data pre-processing (including scaling) on low-dimensional manifolds was studied
    in :cite:`Parente2013`.

    **Example:**

    .. code:: python

        from PCAfold import center_scale
        import numpy as np

        # Generate dummy data set:
        X = np.random.rand(100,20)

        # Center and scale:
        (X_cs, X_center, X_scale) = center_scale(X, 'range', nocenter=False)

    :param X:
        ``numpy.ndarray`` specifying the original data set :math:`\mathbf{X}`. It should be of size ``(n_observations,n_variables)``.
    :param scaling:
        ``str`` specifying the scaling methodology. It can be one of the following:
        ``'none'``, ``''``, ``'auto'``, ``'std'``, ``'pareto'``, ``'vast'``, ``'range'``, ``'0to1'``,
        ``'-1to1'``, ``'level'``, ``'max'``, ``'poisson'``, ``'vast_2'``, ``'vast_3'``, ``'vast_4'``.
    :param nocenter: (optional)
        ``bool`` specifying whether data should be centered by mean. If set to ``True`` data will *not* be centered.

    :return:
        - **X_cs** - ``numpy.ndarray`` specifying the centered and scaled data set :math:`\mathbf{X_{cs}}`. It has size ``(n_observations,n_variables)``.
        - **X_center** - ``numpy.ndarray`` specifying the centers :math:`\mathbf{C}` applied on the original data set :math:`\mathbf{X}`. It has size ``(n_variables,)``.
        - **X_scale** - ``numpy.ndarray`` specifying the scales :math:`\mathbf{D}` applied on the original data set :math:`\mathbf{X}`. It has size ``(n_variables,)``.
    """

    if not isinstance(X, np.ndarray):
        raise ValueError("Parameter `X` has to be of type `numpy.ndarray`.")

    try:
        (n_observations, n_variables) = np.shape(X)
    except:
        raise ValueError("Parameter `X` has to have size `(n_observations,n_variables)`.")

    if not isinstance(scaling, str):
        raise ValueError("Parameter `scaling` has to be a string.")
    else:
        if scaling.lower() not in _scalings_list:
            raise ValueError("Unrecognized scaling method.")

    if not isinstance(nocenter, bool):
        raise ValueError("Parameter `nocenter` has to be a boolean.")

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
    elif scaling == '0TO1':
       X_center = np.min(X, axis=0)
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
        ``numpy.ndarray`` specifying the centered and scaled data set :math:`\mathbf{X_{cs}}`.  It should be of size ``(n_observations,n_variables)``.
    :param X_center:
        ``numpy.ndarray`` specifying the centers :math:`\mathbf{C}` applied on the original data set :math:`\mathbf{X}`. It should be of size ``(n_variables,)``.
    :param X_scale:
        ``numpy.ndarray`` specifying the scales :math:`\mathbf{D}` applied on the original data set :math:`\mathbf{X}`. It should be of size ``(n_variables,)``.

    :return:
        - **X** - ``numpy.ndarray`` specifying the original data set :math:`\mathbf{X}`. It has size ``(n_observations,n_variables)``.
    """

    if not isinstance(X_cs, np.ndarray):
        raise ValueError("Parameter `X_cs` has to be of type `numpy.ndarray`.")

    try:
        (n_observations, n_variables) = np.shape(X_cs)
    except:
        raise ValueError("Parameter `X_cs` has to have size `(n_observations,n_variables)`.")

    if not isinstance(X_center, np.ndarray):
        raise ValueError("Parameter `X_center` has to be of type `numpy.ndarray`.")

    try:
        (n_variables_centers,) = np.shape(X_center)
    except:
        raise ValueError("Parameter `X_center` has to have size `(n_variables,)`.")

    if not isinstance(X_scale, np.ndarray):
        raise ValueError("Parameter `X_scale` has to be of type `numpy.ndarray`.")

    try:
        (n_variables_scales,) = np.shape(X_scale)
    except:
        raise ValueError("Parameter `X_scale` has to have size `(n_variables,)`.")

    if n_variables != n_variables_centers:
        raise ValueError("Parameter `X_center` has different number of variables than parameter `X_cs`.")

    if n_variables != n_variables_scales:
        raise ValueError("Parameter `X_scale` has different number of variables than parameter `X_cs`.")

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

        - **X_removed** - (read only) data set with removed constant columns.
        - **idx_removed** - (read only) the indices of columns removed from :math:`\mathbf{X}`.
        - **idx_retained** - (read only) the indices of columns retained in :math:`\mathbf{X}`.
        - **X_cs** - (read only) centered and scaled data set :math:`\mathbf{X_{cs}}`.
        - **X_center** - (read only) vector of centers :math:`\mathbf{C}` applied on the original data set :math:`\mathbf{X}`.
        - **X_scale** - (read only) vector of scales :math:`\mathbf{D}` applied on the original data set :math:`\mathbf{X}`.
    """

    def __init__(self, X, scaling='none', nocenter=False):

        (self.__X_removed, self.__idx_removed, self.__idx_retained) = remove_constant_vars(X)
        (self.__X_cs, self.__X_center, self.__X_scale) = center_scale(self.X_removed, scaling, nocenter=nocenter)

    @property
    def X_removed(self):
        return self.__X_removed

    @property
    def idx_removed(self):
        return self.__idx_removed

    @property
    def idx_retained(self):
        return self.__idx_retained

    @property
    def X_cs(self):
        return self.__X_cs

    @property
    def X_center(self):
        return self.__X_center

    @property
    def X_scale(self):
        return self.__X_scale

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

def order_variables(X, method='mean', descending=True):
    """
    This function orders variables in the data set using a selected method.

    **Example:**

    .. code::

        from PCAfold import order_variables
        import numpy as np

        # Generate a dummy data set:
        X = np.array([[100, 1, 10],
                      [200, 2, 20],
                      [300, 3, 30]])

        # Order variables by the mean value in the descending order:
        (X_ordered, idx) = order_variables(X, method='mean', descending=True)

    The code above should return an ordered data set:

    .. code-block:: text

        array([[100,  10,   1],
               [200,  20,   2],
               [300,  30,   3]])

    and the list of ordered variable indices:

    .. code-block:: text

        [1, 2, 0]

    :param X:
        original data set :math:`\mathbf{X}`.
    :param method: (optional)
        string specifying the ordering method. Can be ``'mean'``, ``'min'``, ``'max'``, ``'std'`` or ``'var'``.
    :param descending: (optional)
        boolean specifying whether variables should be ordered in the descending order.
        If set to ``False``, variables will be ordered in the ascending order.

    :return:
        - **X_ordered** - original data set with ordered variables.
        - **idx** - list of indices of the ordered variables.
    """

    __method = ['mean', 'min', 'max', 'std', 'var']

    if not isinstance(X, np.ndarray):
        raise ValueError("Parameter `X` has to be a `numpy.ndarray`.")

    if not isinstance(method, str):
        raise ValueError("Parameter `method` has to be a string.")

    if method not in __method:
        raise ValueError("Parameter `method` has to be a 'mean', 'min', 'max', 'std' or 'var'.")

    if not isinstance(descending, bool):
        raise ValueError("Parameter `descending` has to be a boolean.")

    if method == 'mean':
        criterion = np.mean(X, axis=0)
    elif method == 'min':
        criterion = np.min(X, axis=0)
    elif method == 'max':
        criterion = np.max(X, axis=0)
    elif method == 'std':
        criterion = np.std(X, axis=0)
    elif method == 'var':
        criterion = np.var(X, axis=0)

    sorted_pairs = sorted(enumerate(criterion), key=operator.itemgetter(1))
    sorted_indices = [index for index, element in sorted_pairs]

    if descending:
        idx = sorted_indices[::-1]
    else:
        idx = sorted_indices

    X_ordered = X[:,idx]

    return (X_ordered, idx)

def outlier_detection(X, scaling, method='MULTIVARIATE TRIMMING', trimming_threshold=0.5, quantile_threshold=0.9899, verbose=False):
    """
    This function finds outliers in a data set :math:`\mathbf{X}` and returns
    indices of observations without outliers as well as indices of the outliers
    themselves.

    Two options are implemented here:

    - ``'MULTIVARIATE TRIMMING'``

        Outliers are detected based on multivariate Mahalanobis distance :math:`D_M`:

        .. math::

            D_M = \\sqrt{(\mathbf{X} - \mathbf{\\bar{X}})^T \mathbf{S}^{-1} (\mathbf{X} - \mathbf{\\bar{X}})}

        where :math:`\mathbf{\\bar{X}}` is a matrix of the same size as :math:`\mathbf{X}`
        storing in each column a copy of the average value of the same column in :math:`\mathbf{X}`.
        :math:`\mathbf{S}` is the covariance matrix computed as per ``PCA`` class.
        Note that the scaling option selected will affect the covariance matrix :math:`\mathbf{S}`.
        Since Mahalanobis distance takes into account covariance between variables,
        observations with sufficiently large :math:`D_M` can be considered as outliers.
        For more detailed information on Mahalanobis distance the user is referred
        to :cite:`Bishop2006` or :cite:`DeMaesschalck2000`.

        The threshold above which observations will be classified as outliers
        can be specified using ``trimming_threshold`` parameter. Specifically,
        the :math:`i^{th}` observation is classified as an outlier if:

        .. math::

            D_{M, i} > \\verb|trimming_threshold| \\cdot max(D_M)

    - ``'PC CLASSIFIER'``

        Outliers are detected based on major and minor Principal Components (PCs).
        The method of Principal Component classifier (PCC) was first proposed in
        :cite:`Shyu2003`. The application of this technique to combustion data sets
        was studied in :cite:`Parente2013`. Specifically,
        the :math:`i^{th}` observation is classified as an outlier
        if the *first PC classifier* based on :math:`q`-first (major) PCs:

        .. math::

            \sum_{j=1}^{q} \\frac{z_{ij}^2}{L_j} > c_1

        or if the *second PC classifier* based on :math:`(Q-k+1)`-last (minor) PCs:

        .. math::

            \sum_{j=k}^{Q} \\frac{z_{ij}^2}{L_j} > c_2

        where :math:`z_{ij}` is the :math:`i^{th}, j^{th}` element from the Principal
        Components matrix :math:`\mathbf{Z}` and :math:`L_j` is the :math:`j^{th}`
        eigenvalue from :math:`\mathbf{L}` (as per ``PCA`` class).
        Major PCs are selected such that the total variance explained is 50%.
        Minor PCs are selected such that the remaining variance they explain is 20%.

        Coefficients :math:`c_1` and :math:`c_2` are found such that they
        represent the ``quantile_threshold`` (by default 98.99%) quantile of the
        empirical distributions of the first and second PC classifier respectively.

    **Example:**

    .. code::

        from PCAfold import outlier_detection
        import numpy as np

        # Generate dummy data set:
        X = np.random.rand(100,20)

        # Find outliers:
        (idx_outliers_removed, idx_outliers) = outlier_detection(X, scaling='auto', method='MULTIVARIATE TRIMMING', trimming_threshold=0.8, verbose=True)

        # New data set without outliers can be obtained as:
        X_outliers_removed = X[idx_outliers_removed,:]

        # Observations that were classified as outliers can be obtained as:
        X_outliers = X[idx_outliers,:]

    :param X:
        original data set :math:`\mathbf{X}`.
    :param scaling:
        string specifying the scaling methodology as per
        ``preprocess.center_scale`` function.
    :param method: (optional)
        string specifying the outlier detection method to use. It should be
        ``'MULTIVARIATE TRIMMING'`` or ``'PC CLASSIFIER'``.
    :param trimming_threshold: (optional)
        trimming threshold to use in combination with ``'MULTIVARIATE TRIMMING'`` method.
    :param quantile_threshold: (optional)
        quantile threshold to use in combination with ``'PC CLASSIFIER'`` method.
    :param verbose: (optional)
        boolean for printing outlier detection details.

    :raises ValueError:
        if ``scaling`` is not a string or is not consistent with scalings available in the ``preprocess.center_scale`` function.
    :raises ValueError:
        if ``method`` is not ``'MULTIVARIATE TRIMMING'`` or ``'PC CLASSIFIER'``.
    :raises ValueError:
        if ``trimming_threshold`` is not between 0-1.
    :raises ValueError:
        if ``quantile_threshold`` is not between 0-1.
    :raises ValueError:
        if ``verbose`` is not a boolean.

    :return:
        - **idx_outliers_removed** - indices of observations without outliers.
        - **idx_outliers** - indices of observations that were classified as outliers.
    """

    _detection_methods = ['MULTIVARIATE TRIMMING', 'PC CLASSIFIER']

    from PCAfold import PCA

    if not isinstance(scaling, str):
        raise ValueError("Parameter `scaling` has to be a string.")
    else:
        if scaling.lower() not in _scalings_list:
            raise ValueError("Unrecognized scaling method.")

    if not isinstance(method, str):
        raise ValueError("Parameter `method` has to be a string.")
    else:
        if method.upper() not in _detection_methods:
            raise ValueError("Unrecognized outlier detection method.")

    if trimming_threshold < 0 or trimming_threshold > 1:
        raise ValueError("Parameter `trimming_threshold` has to be between 0 and 1.")

    if not isinstance(verbose, bool):
        raise ValueError("Parameter `verbose` has to be a boolean.")

    (n_observations, n_variables) = np.shape(X)

    idx_full = np.arange(0, n_observations)
    idx_outliers_removed = []
    idx_outliers = []

    pca_X = PCA(X, scaling=scaling, n_components=0)

    if method.upper() == 'MULTIVARIATE TRIMMING':

        means_of_X = np.mean(X, axis=0)

        covariance_matrix = pca_X.S

        inverse_covariance_matrix = np.linalg.inv(covariance_matrix)

        mahalanobis_distances = np.zeros((n_observations,))

        for n_obs in range(0, n_observations):

            mahalanobis_distance = np.sqrt(np.dot((X[n_obs,:] - means_of_X), np.dot(inverse_covariance_matrix, (X[n_obs,:] - means_of_X))))
            mahalanobis_distances[n_obs,] = mahalanobis_distance

        minimum_mahalanobis_distance = np.min(mahalanobis_distances)
        maximum_mahalanobis_distance = np.max(mahalanobis_distances)

        range_mahalanobis_distance = maximum_mahalanobis_distance - minimum_mahalanobis_distance

        (idx_outliers, ) = np.where(mahalanobis_distances > trimming_threshold * maximum_mahalanobis_distance)

        idx_outliers_removed = np.setdiff1d(idx_full, idx_outliers)

        if verbose:
            n_outliers = len(idx_outliers)
            print('Number of observations classified as outliers: ' + str(n_outliers))

    elif method.upper() == 'PC CLASSIFIER':

        principal_components = pca_X.transform(X)
        eigenvalues = pca_X.L
        n_components = pca_X.n_components

        # Select major components based on 50% of the original data variance:
        pca_major = pca_X.set_retained_eigenvalues(method='TOTAL VARIANCE', option=0.5)
        n_major_components = pca_major.n_components

        # Select minor components based on 20% of the total variance in the data:
        pca_minor = pca_X.set_retained_eigenvalues(method='TOTAL VARIANCE', option=0.8)
        n_minor_components = pca_minor.n_components

        if verbose:
            print("Major components that will be selected are: " + ', '.join([str(i) for i in range(1, n_major_components+1)]))

        if verbose:
            print("Minor components that will be selected are: " + ', '.join([str(i) for i in range(n_minor_components, n_components+1)]))

        scaled_squared_PCs = np.divide(np.square(principal_components), eigenvalues)

        distances_major = np.sum(scaled_squared_PCs[:,0:n_major_components], axis=1)
        distances_minor = np.sum(scaled_squared_PCs[:,(n_minor_components-1):n_components], axis=1)

        # Threshold coefficient c_1 (for major PCs):
        threshold_coefficient_major = np.quantile(distances_major, quantile_threshold)

        # Threshold coefficient c_2 (for minor PCs):
        threshold_coefficient_minor = np.quantile(distances_minor, quantile_threshold)

        (idx_outliers_major, ) = np.where((distances_major > threshold_coefficient_major))
        (idx_outliers_minor, ) = np.where((distances_minor > threshold_coefficient_minor))

        idx_outliers = np.vstack((idx_outliers_major[:,np.newaxis], idx_outliers_minor[:,np.newaxis]))

        idx_outliers = np.unique(idx_outliers)

        idx_outliers_removed = np.setdiff1d(idx_full, idx_outliers)

        if verbose:
            n_outliers = len(idx_outliers)
            print('Number of observations classified as outliers: ' + str(n_outliers))

    idx_outliers_removed = np.sort(idx_outliers_removed.astype(int))
    idx_outliers = np.sort(idx_outliers.astype(int))

    return (idx_outliers_removed, idx_outliers)

class KernelDensity:
    """
    This class enables kernel density weighting of data sets
    based on *single-variable* or *multi-variable* case as proposed in
    :cite:`Coussement2012`.

    The goal of both cases is to obtain a vector of weights :math:`\mathbf{W_c}` that
    has the same number of elements as there are observations in the original
    data set :math:`\mathbf{X}`.
    Each observation will then get multiplied by the corresponding weight from
    :math:`\mathbf{W_c}`.

    .. note::

        Kernel density weighting technique is usually very expensive, even
        on data sets with relatively small number of observations.
        Since the *single-variable* case is a cheaper option than the *multi-variable*
        case, it is recommended that this technique is tried first on larger data
        sets.

    Gaussian kernel is used in both approaches:

    .. math::

        K_{c, c'} = \sqrt{\\frac{1}{2 \pi h^2}} exp(- \\frac{d^2}{2 h^2})

    :math:`h` is the kernel bandwidth:

    .. math::

        h = \Big( \\frac{4 \hat{\sigma}}{3 n} \Big)^{1/5}

    where :math:`\hat{\sigma}` is the standard deviation of the considered variable
    and :math:`n` is the number of observations in the data set.

    :math:`d` is the distance between two observations :math:`c` and :math:`c'`:

    .. math::

        d = |x_c - x_{c'}|

    **Single-variable**

    If the ``conditioning_variable`` argument is a single vector, weighting will be performed
    according to the *single-variable* case. It begins by summing Gaussian kernels:

    .. math::

        \mathbf{K_c} = \sum_{c' = 1}^{c' = n} \\frac{1}{n} K_{c, c'}

    and weights are then computed as:

    .. math::

        \mathbf{W_c} = \\frac{\\frac{1}{\mathbf{K_c}}}{max(\\frac{1}{\mathbf{K_c}})}

    **Multi-variable**

    If the ``conditioning_variable`` argument is a matrix of multiple variables, weighting will
    be performed according to the *multi-variable* case. It begins by summing
    Gaussian kernels for a :math:`k^{th}` variable:

    .. math::

        \mathbf{K_c}_{, k} = \sum_{c' = 1}^{c' = n} \\frac{1}{n} K_{c, c', k}

    Global density taking into account all variables is then obtained as:

    .. math::

        \mathbf{K_{c}} = \prod_{k=1}^{k=Q} \mathbf{K_c}_{, k}

    where :math:`Q` is the total number of conditioning variables, and weights are computed as:

    .. math::

        \mathbf{W_c} = \\frac{\\frac{1}{\mathbf{K_c}}}{max(\\frac{1}{\mathbf{K_c}})}

    **Example:**

    .. code:: python

        from PCAfold import KernelDensity
        import numpy as np

        # Generate dummy data set:
        X = np.random.rand(100,20)

        # Perform kernel density weighting based on the first variable:
        kerneld = KernelDensity(X, X[:,0])

        # Access the weighted data set:
        X_weighted = kerneld.X_weighted

        # Access the weights used to scale the data set:
        weights = kerneld.weights

    :param X:
        data set :math:`\mathbf{X}` that should be weighted.
    :param conditioning_variable:
        either a single variable or multiple variables to be used as a
        conditioning variable for kernel weighting procedure. Note that it can also
        be passed as the data set :math:`\mathbf{X}`.

    :raises ValueError:
        if the number of observations in ``X`` is different than the number of observations in ``conditioning_variable``.

    **Attributes:**

        - **weights** - vector of computed weights :math:`\mathbf{W_c}`.
        - **X_weighted** - weighted data set (each observation in\
        :math:`\mathbf{X}` is multiplied by the corresponding weight in :math:`\mathbf{W_c}`).
    """

    def __init__(self, X, conditioning_variable, verbose=False):

        (n_observations_X, n_variables_X) = np.shape(X)

        try:
            (n_observations, n_variables) = np.shape(conditioning_variable)
        except:
            (n_observations, n_variables) = np.shape(conditioning_variable[:,np.newaxis])

        if n_observations_X != n_observations:
            raise ValueError("The data set to weight and the conditioning variable should have the same number of observations.")

        if n_variables == 1:

            if verbose: print('Single-variable case will be applied.')

            self.__weights = self.__single_variable_observation_weights(conditioning_variable)

        elif n_variables > 1:

            if verbose: print('Multi-variable case will be applied.')

            self.__weights = self.__multi_variable_observation_weights(conditioning_variable)

        self.__X_weighted = np.multiply(X, self.weights)

    @property
    def weights(self):
        return self.__weights

    @property
    def X_weighted(self):
        return self.__X_weighted

    # Computes eq.(26):
    def __bandwidth(self, n, mean_standard_deviation):
        """
        This function computes kernel bandwidth as:

        .. math::

            h = \Big( \\frac{4 \hat{\sigma}}{3 n} \Big)^{1/5}

        :param n:
            number of observations in a data set or a variable vector.
        :param mean_standard_deviation:
            mean standard deviation in the entire data set or a variable vector.

        :returns:
            - **h** - kernel bandwidth, scalar.
        """

        h = (4*mean_standard_deviation/(3*n))**(1/5)

        return(h)

    # Computes eq.(21):
    def __distance(self, x_1, x_2):
        """
        This function computes distance between two observations:

        .. math::

            d = |x_1 - x_2|

        :param x_1:
            first observation.
        :param x_2:
            second observation.

        :returns:
            - **d** - distance between the first and second observation.
        """

        d = abs(x_1 - x_2)

        return(d)

    # Computes eq.(22):
    def __gaussian_kernel(self, x1, x2, n, mean_standard_deviation):
        """
        This function computes a Gaussian kernel:

        .. math::

            K = \sqrt{\\frac{1}{2 \pi h^2}} exp(- \\frac{d^2}{2 h^2})

        :param x_1:
            first observation.
        :param x_2:
            second observation.
        :param n:
            number of observations in a data set or a variable vector.
        :param mean_standard_deviation:
            mean standard deviation in the entire data set or a variable vector.

        :returns:
            - **K** - Gaussian kernel.
        """

        d = self.__distance(x1, x2)
        h = self.__bandwidth(n, mean_standard_deviation)

        K = (1/(2*np.pi*h**2))**0.5 * np.exp(- d/(2*h**2))

        return(K)

    # Computes eq.(23):
    def __variable_density(self, x, mean_standard_deviation):
        """
        This function computes a vector of variable densities for all observations.

        :param x:
            single variable vector.
        :param mean_standard_deviation:
            mean standard deviation in the entire data set or a variable vector.

        :returns:
            - **Kck** - a vector of variable densities for all observations, it has the same size as the variable vector `x`.
        """

        n = len(x)

        Kck = np.zeros((n,1))

        for i in range(0,n):

            gaussian_kernel_sum = 0

            for j in range(0,n):

                gaussian_kernel_sum = gaussian_kernel_sum + self.__gaussian_kernel(x[i], x[j], n, mean_standard_deviation)

            Kck[i] = 1/n * gaussian_kernel_sum

        return(Kck)

    # Computes eq.(24):
    def __multi_variable_global_density(self, X):
        """
        This function computes a vector of variable global densities for a
        multi-variable case, for all observations.

        :param X:
            multi-variable data set matrix.

        :returns:
            - **Kc** - a vector of global densities for all observations.
        """

        (n, n_vars) = np.shape(X)

        mean_standard_deviation = np.mean(np.std(X, axis=0))

        Kck_matrix = np.zeros((n, n_vars))

        for variable in range(0, n_vars):

            Kck_matrix[:,variable] = np.reshape(self.__variable_density(X[:,variable], mean_standard_deviation), (n,))

        # Compute the global densities vector:
        Kc = np.zeros((n,1))

        K = 1

        for i in range(0,n):

            Kc[i] = K * np.prod(Kck_matrix[i,:])

        return(Kc)

    # Computes eq.(25):
    def __multi_variable_observation_weights(self, X):
        """
        This function computes a vector of observation weights for a
        multi-variable case.

        :param X:
            multi-variable data set matrix.

        :returns:
            - **W_c** - a vector of observation weights.
        """

        (n, n_vars) = np.shape(X)

        W_c = np.zeros((n,1))

        Kc = self.__multi_variable_global_density(X)

        Kc_inv = 1/Kc

        for i in range(0,n):

            W_c[i] = Kc_inv[i] / np.max(Kc_inv)

        return(W_c)

    # Computes eq.(20):
    def __single_variable_observation_weights(self, x):
        """
        This function computes a vector of observation weights for a
        single-variable case.

        :param x:
            single variable vector.

        :returns:
            - **W_c** - a vector of observation weights.
        """

        n = len(x)

        mean_standard_deviation = np.std(x)

        W_c = np.zeros((n,1))

        Kc = self.__variable_density(x, mean_standard_deviation)

        Kc_inv = 1/Kc

        for i in range(0,n):

            W_c[i] = Kc_inv[i] / np.max(Kc_inv)

        return(W_c)

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

        .. image:: ../images/sampling-test-selection-option-number.svg
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

        .. image:: ../images/sampling-test-selection-option-percentage.svg
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

        .. image:: ../images/sampling-test-selection-option-manual.svg
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

        .. image:: ../images/sampling-test-selection-option-random.svg
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

    .. image:: ../images/clustering-variable-bins.svg
      :width: 600
      :align: center

    **Example:**

    .. code::

        from PCAfold import variable_bins
        import numpy as np

        # Generate dummy variable:
        x = np.linspace(-1,1,100)

        # Create partitioning according to bins of x:
        (idx, borders) = variable_bins(x, 4, verbose=True)

    :param var:
        vector of variable values.
    :param k:
        number of clusters to partition the data.
    :param verbose: (optional)
        boolean for printing clustering details.

    :raises ValueError:
        if number of clusters ``k`` is not a positive integer.
    :raises ValueError:
        if ``verbose`` is not a boolean.

    :return:
        - **idx** - vector of cluster classifications.
        - **borders** - list of values that define borders of the clusters.
    """

    # Check that the number of clusters is an integer and is non-zero:
    if not (isinstance(k, int) and k > 0):
        raise ValueError("The number of clusters must be a positive integer.")

    if not isinstance(verbose, bool):
        raise ValueError("Parameter `verbose` has to be a boolean.")

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

    borders = bins_borders

    return (idx, borders)

def predefined_variable_bins(var, split_values, verbose=False):
    """
    This function does clustering by dividing a variable vector ``var`` into
    bins such that splits are done at user-specified values.
    Split values can be specified in the ``split_values`` list.
    In general: ``split_values = [value_1, value_2, ..., value_n]``.

    *Note:*
    When a split is performed at a given ``value_i``, the observation in ``var``
    that takes exactly that value is assigned to the newly created bin.

    An example of how a vector can be partitioned with this function is presented below:

    .. image:: ../images/clustering-predefined-variable-bins.svg
      :width: 600
      :align: center

    **Example:**

    .. code::

        from PCAfold import predefined_variable_bins
        import numpy as np

        # Generate dummy variable:
        x = np.linspace(-1,1,100)

        # Create partitioning according to pre-defined bins of x:
        (idx, borders) = predefined_variable_bins(x, [-0.6, 0.4, 0.8], verbose=True)

    :param var:
        vector of variable values.
    :param split_values:
        list containing values at which the split to bins should be performed.
    :param verbose: (optional)
        boolean for printing clustering details.

    :raises ValueError:
        if any value within ``split_values`` is not within the range of
        vector ``var`` values.
    :raises ValueError:
        if ``verbose`` is not a boolean.

    :return:
        - **idx** - vector of cluster classifications.
        - **borders** - list of values that define borders of the clusters.
    """

    if not isinstance(verbose, bool):
        raise ValueError("Parameter `verbose` has to be a boolean.")

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

    borders = bins_borders

    return (idx, borders)

def mixture_fraction_bins(Z, k, Z_stoich, verbose=False):
    """
    This function does clustering by dividing a mixture fraction vector
    ``Z`` into bins of equal lengths. This technique can be used to partition
    combustion data sets as proposed in :cite:`Parente2009`.
    The vector is first split to lean and rich
    side (according to the stoichiometric mixture fraction ``Z_stoich``) and
    then the sides get divided further into clusters. When ``k`` is odd,
    there will always be one more cluster on the side with larger range in
    mixture fraction space compared to the other side.

    An example of how a vector can be partitioned with this function is presented below:

    .. image:: ../images/clustering-mixture-fraction-bins.svg
      :width: 600
      :align: center

    **Example:**

    .. code::

        from PCAfold import mixture_fraction_bins
        import numpy as np

        # Generate dummy mixture fraction variable:
        Z = np.linspace(0,1,100)

        # Create partitioning according to bins of mixture fraction:
        (idx, borders) = mixture_fraction_bins(Z, 4, 0.4, verbose=True)

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
    :raises ValueError:
        if the stoichiometric mixture fraction is not a number between 0 and 1.
    :raises ValueError:
        if ``verbose`` is not a boolean.

    :return:
        - **idx** - vector of cluster classifications.
        - **borders** - list of values that define borders of the clusters.
    """

    if not isinstance(verbose, bool):
        raise ValueError("Parameter `verbose` has to be a boolean.")

    # Check that the number of clusters is an integer and is non-zero:
    if not (isinstance(k, int) and k > 0):
        raise ValueError("The number of clusters must be a positive integer.")

    if Z_stoich < 0 or Z_stoich > 1:
        raise ValueError("Stoichiometric mixture fraction should be between 0 and 1.")

    # Number of interval borders:
    n_bins_borders = k + 1

    # Minimum and maximum mixture fraction:
    min_Z = np.min(Z)
    max_Z = np.max(Z)

    # Partition the Z-space:
    if k == 1:

        borders = np.linspace(min_Z, max_Z, n_bins_borders)

    else:

        if Z_stoich <= 0.5:

            # Z-space lower than stoichiometric mixture fraction:
            borders_lean = np.linspace(min_Z, Z_stoich, int(np.ceil(n_bins_borders/2)))

            # Z-space higher than stoichiometric mixture fraction:
            borders_rich = np.linspace(Z_stoich, max_Z, int(np.ceil((n_bins_borders+1)/2)))

        else:

            # Z-space lower than stoichiometric mixture fraction:
            borders_lean = np.linspace(min_Z, Z_stoich, int(np.ceil((n_bins_borders+1)/2)))

            # Z-space higher than stoichiometric mixture fraction:
            borders_rich = np.linspace(Z_stoich, max_Z, int(np.ceil(n_bins_borders/2)))

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

    return (idx, borders)

def zero_neighborhood_bins(var, k, zero_offset_percentage=0.1, split_at_zero=False, verbose=False):
    """
    This function aims to separate close-to-zero observations in a vector into one
    cluster (``split_at_zero=False``) or two clusters (``split_at_zero=True``).
    The offset from zero at which splits are performed is computed
    based on the input parameter ``zero_offset_percentage``:

    .. math::

        \\verb|offset| = \\frac{(max(\\verb|var|) - min(\\verb|var|)) \cdot \\verb|zero_offset_percentage|}{100}

    Further clusters are found by splitting positive and negative values in a vector
    alternatingly into bins of equal lengths.

    This clustering technique can be useful for partitioning any variable
    that has many observations clustered around zero value and relatively few
    observations far away from zero on either side.

    Two examples of how a vector can be partitioned with this function are presented below.

    With ``split_at_zero=False``:

    .. image:: ../images/clustering-zero-neighborhood-bins.svg
      :width: 700
      :align: center

    If ``split_at_zero=False`` the smallest allowed number of clusters is 3.
    This is to assure that there are at least three clusters:
    with negative values, with close to zero values, with positive values.

    When ``k`` is even, there will always be one more cluster on the side with
    larger range compared to the other side.

    With ``split_at_zero=True``:

    .. image:: ../images/clustering-zero-neighborhood-bins-zero-split.svg
      :width: 700
      :align: center

    If ``split_at_zero=True`` the smallest allowed number of clusters is 4.
    This is to assure that there are at least four clusters: with negative
    values, with negative values close to zero, with positive values close to
    zero and with positive values.

    When ``k`` is odd, there will always be one more cluster on the side with
    larger range compared to the other side.

    .. note::

        This clustering technique is well suited for partitioning chemical
        source terms :math:`\mathbf{S_X}` or sources of Principal Components
        :math:`\mathbf{S_Z}` (as per :cite:`Sutherland2009`) since it relies on
        unbalanced vectors that have many observations numerically close to zero.
        Using ``split_at_zero=True`` it can further differentiate between
        negative and positive sources which have different physical effect.

    **Example:**

    .. code::

        from PCAfold import zero_neighborhood_bins
        import numpy as np

        # Generate dummy variable:
        x = np.linspace(-100,100,1000)

        # Create partitioning according to bins of x:
        (idx, borders) = zero_neighborhood_bins(x, 4, zero_offset_percentage=10, split_at_zero=True, verbose=True)

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
    :raises ValueError:
        if ``verbose`` is not a boolean.

    :return:
        - **idx** - vector of cluster classifications.
        - **borders** - list of values that define borders of the clusters.
    """

    if not isinstance(verbose, bool):
        raise ValueError("Parameter `verbose` has to be a boolean.")

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
        raise ValueError("Variable vector does not have negative values. Use `predefined_variable_bins` as a clustering technique instead.")

    if not (var_max >= 0):
        raise ValueError("Variable vector does not have positive values. Use `predefined_variable_bins` as a clustering technique instead.")

    if (var_min > -offset) or (var_max < offset):
        raise ValueError("Offset from zero crosses the minimum or maximum value of the variable vector. Consider lowering `zero_offset_percentage`.")

    # Number of interval borders:
    if split_at_zero:
        n_bins_borders = k-1
    else:
        n_bins_borders = k

    if abs(var_min) > abs(var_max):

        # Generate cluster borders on the negative side:
        borders_negative = np.linspace(var_min, -offset, int(np.ceil((n_bins_borders+1)/2)))

        # Generate cluster borders on the positive side:
        borders_positive = np.linspace(offset, var_max, int(np.ceil(n_bins_borders/2)))

    else:

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

    return (idx, borders)

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
    :raises ValueError:
        if ``verbose`` is not a boolean.

    :return:
        - **idx_degraded** degraded vector of cluster classifications. The first cluster has index 0.
        - **k_update** - the updated number of clusters.
    """

    if not isinstance(verbose, bool):
        raise ValueError("Parameter `verbose` has to be a boolean.")

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
    This function computes the centroids for each cluster specified in the
    ``idx`` vector. Centroids :math:`c_i` of cluster :math:`k_i` are computed as:

    .. math::

        c_i = mean(\mathbf{x}_i)

    where :math:`\mathbf{x}_i` are the observations belonging to cluster :math:`k_i`
    of all variables in the data set :math:`\mathbf{X}`. Centroids of all variables
    from all clusters are then stored in the matrix :math:`\mathbf{c}` returned.

    **Example:**

    .. code::

        from PCAfold import get_centroids
        import numpy as np

        # Generate dummy data set:
        X = np.random.rand(100,5)

        # Generate dummy clustering of the data set:
        idx = np.zeros((100,))
        idx[50:80] = 1

        # Compute the centroids of each cluster:
        centroids = get_centroids(X, idx)

    :param X:
        data set :math:`\mathbf{X}` for computing the cluster centroids.
    :param idx:
        vector of cluster classifications.

    :raises ValueError:
        if the number of observations in the data set ``X`` does not match the
        number of elements in the ``idx`` vector.

    :return:
        - **centroids** - matrix of centroids of all :math:`k` clusters. It has size :math:`k` times number of variables.
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

def get_partition(X, idx):
    """
    This function partitions the observations from the original data
    set :math:`\mathbf{X}` into :math:`k` clusters according to ``idx`` provided.

    **Example:**

    .. code::

        from PCAfold import get_partition
        import numpy as np

        # Generate dummy data set:
        X = np.random.rand(100,5)

        # Generate dummy clustering of the data set:
        idx = np.zeros((100,))
        idx[50:80] = 1

        # Generate partitioning of the data set according to idx:
        (X_in_clusters, idx_in_clusters) = get_partition(X, idx)

    :param X:
        data set to partition.
    :param idx:
        vector of cluster classifications.

    :raises ValueError:
        if the number of observations in the data set ``X`` does not match the
        number of elements in the ``idx`` vector.

    :return:
        - **X_in_clusters** - list of :math:`k` arrays that contains original data set observations partitioned to :math:`k` clusters.
        - **idx_in_clusters** - list of :math:`k` arrays that contains indices of the original data set observations partitioned to :math:`k` clusters.
    """

    try:
        (n_observations, n_variables) = np.shape(X)
    except:
        (n_observations, ) = np.shape(X)
        n_variables = 1

    # Check if the number of indices in `idx` is the same as the number of observations in a data set:
    if n_observations != len(idx):
        raise ValueError("The number of observations in the data set `X` must match the number of elements in `idx` vector.")

    # Degrade clusters if needed:
    if (len(np.unique(idx)) != (np.max(idx)+1)) or (np.min(idx) != 0):
        (idx, _) = degrade_clusters(idx, verbose=False)

    k = len(np.unique(idx))

    idx_in_clusters = []
    X_in_clusters = []

    for i in range(0,k):

        indices_to_append = np.argwhere(idx==i).ravel()
        idx_in_clusters.append(indices_to_append)

        if n_variables == 1:
            X_in_clusters.append(X[indices_to_append])
        else:
            X_in_clusters.append(X[indices_to_append,:])

    return(X_in_clusters, idx_in_clusters)

def get_populations(idx):
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

    :return:
        - **populations** - list of cluster populations. Each entry referes to one cluster ordered according to ``idx``.
    """

    populations = []

    # Degrade clusters if needed:
    if (len(np.unique(idx)) != (np.max(idx)+1)) or (np.min(idx) != 0):
        (idx, _) = degrade_clusters(idx, verbose=False)

    # Find the number of clusters:
    k = len(np.unique(idx))

    for i in range(0,k):

        populations.append(int((idx==i).sum()))

    return(populations)

def get_average_centroid_distance(X, idx, weighted=False):
    """
    This function computes the average Euclidean distance between observations
    and the centroids of clusters to which each observation belongs.

    The average can be computed as an arithmetic average from all clusters
    (``weighted=False``) or as a weighted average (``weighted=True``). In the
    latter, the distances are weighted by the number of
    observations in a cluster so that the average centroid distance will approach
    the average distance in the largest cluster.

    **Example:**

    .. code::

        from PCAfold import get_average_centroid_distance
        import numpy as np

        # Generate dummy data set:
        X = np.random.rand(100,5)

        # Generate dummy clustering of the data set:
        idx = np.zeros((100,))
        idx[50:80] = 1

        # Compute average distance from cluster centroids:
        average_centroid_distance = get_average_centroid_distance(X, idx, weighted=False)

    :param X:
        data set to partition.
    :param idx:
        vector of cluster classifications.
    :param weighted: (optional)
        boolean specifying whether distances from centroid should be weighted
        by the number of observations in a cluster.
        If set to ``False``, arithmetic average will be computed.

    :raises ValueError:
        if the number of observations in the data set ``X`` does not match the
        number of elements in the ``idx`` vector.

    :return:
        - **average_centroid_distance** - float specifying the average distance from centroids, averaged over all observations and all clusters.
    """

    try:
        (n_observations, n_variables) = np.shape(X)
    except:
        (n_observations, ) = np.shape(X)

    # Check if the number of indices in `idx` is the same as the number of observations in a data set:
    if n_observations != len(idx):
        raise ValueError("The number of observations in the data set `X` must match the number of elements in `idx` vector.")

    # Degrade clusters if needed:
    if (len(np.unique(idx)) != (np.max(idx)+1)) or (np.min(idx) != 0):
        (idx, _) = degrade_clusters(idx, verbose=False)

    # Find the number of clusters:
    n_clusters = len(np.unique(idx))

    centroids = get_centroids(X, idx)

    average_centroid_distances = np.zeros((n_clusters, 1))

    (X_in_clusters, _) = get_partition(X, idx)

    for cluster in range(0,n_clusters):

        (n_observations_in_cluster, _) = np.shape(X_in_clusters[cluster])

        X_in_clusters_centered = X_in_clusters[cluster] - centroids[cluster]

        X_in_clusters_centered_squared = np.square(X_in_clusters_centered)

        euclidean_distances = np.sqrt(np.sum(X_in_clusters_centered_squared, axis=1))

        if weighted:
            average_centroid_distances[cluster] = np.mean(euclidean_distances)*n_observations_in_cluster
        else:
            average_centroid_distances[cluster] = np.mean(euclidean_distances)

    if weighted:
        average_centroid_distance = np.sum(average_centroid_distances)/n_observations
    else:
        average_centroid_distance = np.mean(average_centroid_distances)

    return average_centroid_distance

################################################################################
#
# Plotting functions
#
################################################################################

def plot_2d_clustering(x, y, idx, x_label=None, y_label=None, color_map='viridis', first_cluster_index_zero=True, grid_on=False, figure_size=(7,7), title=None, save_filename=None):
    """
    This function plots a 2-dimensional manifold divided into clusters.
    Number of observations in each cluster will be plotted in the legend.

    **Example:**

    .. code:: python

        from PCAfold import variable_bins, plot_2d_clustering
        import numpy as np

        # Generate dummy data set:
        x = np.linspace(-1,1,100)
        y = -x**2 + 1

        # Generate dummy clustering of the data set:
        (idx, _) = variable_bins(x, 4, verbose=False)

        # Plot the clustering result:
        plt = plot_2d_clustering(x, y, idx, x_label='$x$', y_label='$y$', color_map='viridis', first_cluster_index_zero=False, grid_on=True, figure_size=(10,6), title='x-y data set', save_filename='clustering.pdf')
        plt.close()

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
        plot will not be saved. You can also set a desired file extension,
        for instance ``.pdf``. If the file extension is not specified, the default
        is ``.png``.

    :return:
        - **plt** - plot handle.
    """

    from matplotlib import cm

    n_clusters = len(np.unique(idx))
    populations = get_populations(idx)

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

def plot_3d_clustering(x, y, z, idx, elev=45, azim=-45, x_label=None, y_label=None, z_label=None, color_map='viridis', first_cluster_index_zero=True, figure_size=(7,7), title=None, save_filename=None):
    """
    This function plots a 3-dimensional manifold divided into clusters.
    Number of observations in each cluster will be plotted in the legend.

    **Example:**

    .. code:: python

        from PCAfold import variable_bins, plot_3d_clustering
        import numpy as np

        # Generate dummy data set:
        x = np.linspace(-1,1,100)
        y = -x**2 + 1
        z = x + 10

        # Generate dummy clustering of the data set:
        (idx, _) = variable_bins(x, 4, verbose=False)

        # Plot the clustering result:
        plt = plot_3d_clustering(x, y, z, idx, x_label='$x$', y_label='$y$', z_label='$z$', color_map='viridis', first_cluster_index_zero=False, grid_on=True, figure_size=(10,6), title='x-y-z data set', save_filename='clustering.pdf')
        plt.close()

    :param x:
        variable on the :math:`x`-axis.
    :param y:
        variable on the :math:`y`-axis.
    :param z:
        variable on the :math:`z`-axis.
    :param idx:
        vector of cluster classifications.
    :param elev: (optional)
        elevation angle.
    :param azim: (optional)
        azimuth angle.
    :param x_label: (optional)
        string specifying :math:`x`-axis label annotation. If set to ``None``
        label will not be plotted.
    :param y_label: (optional)
        string specifying :math:`y`-axis label annotation. If set to ``None``
        label will not be plotted.
    :param z_label: (optional)
        string specifying :math:`z`-axis label annotation. If set to ``None``
        label will not be plotted.
    :param color_map: (optional)
        colormap to use as per ``matplotlib.cm``. Default is *viridis*.
    :param first_cluster_index_zero: (optional)
        boolean specifying if the first cluster should be indexed ``0`` on the plot.
        If set to ``False`` the first cluster will be indexed ``1``.
    :param figure_size:
        tuple specifying figure size.
    :param title: (optional)
        string specifying plot title. If set to ``None`` title will not be
        plotted.
    :param save_filename: (optional)
        string specifying plot save location/filename. If set to ``None``
        plot will not be saved. You can also set a desired file extension,
        for instance ``.pdf``. If the file extension is not specified, the default
        is ``.png``.

    :return:
        - **plt** - plot handle.
    """

    from matplotlib import cm
    from mpl_toolkits.mplot3d import Axes3D

    if not isinstance(x, np.ndarray):
        raise ValueError("Parameter `x` has to be of type `numpy.ndarray`.")

    try:
        (n_x,) = np.shape(x)
        n_var_x = 1
    except:
        (n_x, n_var_x) = np.shape(x)

    if n_var_x != 1:
        raise ValueError("Parameter `x` has to be a 0D or 1D vector.")

    if not isinstance(y, np.ndarray):
        raise ValueError("Parameter `y` has to be of type `numpy.ndarray`.")

    try:
        (n_y,) = np.shape(y)
        n_var_y = 1
    except:
        (n_y, n_var_y) = np.shape(y)

    if n_var_y != 1:
        raise ValueError("Parameter `y` has to be a 0D or 1D vector.")

    if not isinstance(z, np.ndarray):
        raise ValueError("Parameter `z` has to be of type `numpy.ndarray`.")

    try:
        (n_z,) = np.shape(z)
        n_var_z = 1
    except:
        (n_z, n_var_z) = np.shape(z)

    if n_var_z != 1:
        raise ValueError("Parameter `z` has to be a 0D or 1D vector.")

    if n_x != n_y or n_y != n_z:
        raise ValueError("Parameters `x`, `y` and `z` have to have the same number of elements.")

    if n_x != len(idx):
        raise ValueError("Parameter `idx` has to have the same number of elements as `x`, `y` and `z`.")

    n_clusters = len(np.unique(idx))
    populations = get_populations(idx)

    x = x.ravel()
    y = y.ravel()
    z = z.ravel()

    color_map_colors = cm.get_cmap(color_map, n_clusters)
    cluster_colors = color_map_colors(np.linspace(0, 1, n_clusters))

    fig = plt.figure(figsize=figure_size)
    ax = fig.add_subplot(111, projection='3d')

    for k in range(0,n_clusters):
        if first_cluster_index_zero:
            ax.scatter(x[np.where(idx==k)], y[np.where(idx==k)], z[np.where(idx==k)], color=cluster_colors[k], marker='o', s=scatter_point_size, label='$k_{' + str(k) + '}$ - ' + str(populations[k]))
        else:
            ax.scatter(x[np.where(idx==k)], y[np.where(idx==k)], z[np.where(idx==k)], color=cluster_colors[k], marker='o', s=scatter_point_size, label='$k_{' + str(k+1) + '}$ - ' + str(populations[k]))

    plt.legend(bbox_to_anchor=(1.3, 1), fancybox=True, shadow=True, ncol=1, fontsize=font_legend, markerscale=marker_scale_legend_clustering)

    if x_label != None: ax.set_xlabel(x_label, **csfont, fontsize=font_labels, rotation=0, labelpad=20)
    if y_label != None: ax.set_ylabel(y_label, **csfont, fontsize=font_labels, rotation=0, labelpad=20)
    if z_label != None: ax.set_zlabel(z_label, **csfont, fontsize=font_labels, rotation=0, labelpad=20)

    ax.tick_params(pad=5)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('w')
    ax.yaxis.pane.set_edgecolor('w')
    ax.zaxis.pane.set_edgecolor('w')
    ax.view_init(elev=elev, azim=azim)
    ax.grid(alpha=grid_opacity)

    for label in (ax.get_xticklabels()):
        label.set_fontsize(font_axes)
    for label in (ax.get_yticklabels()):
        label.set_fontsize(font_axes)
    for label in (ax.get_zticklabels()):
        label.set_fontsize(font_axes)

    if title != None: ax.set_title(title, **csfont, fontsize=font_title)
    if save_filename != None: plt.savefig(save_filename, dpi = 500, bbox_inches='tight')

    return plt

def plot_2d_train_test_samples(x, y, idx, idx_train, idx_test, x_label=None, y_label=None, color_map='viridis', first_cluster_index_zero=True, grid_on=False, figure_size=(14,7), title=None, save_filename=None):
    """
    This function plots a 2-dimensional manifold divided into train and test
    samples. Number of observations in train and test data respectively will be
    plotted in the legend.

    **Example:**

    .. code:: python

        from PCAfold import variable_bins, DataSampler, plot_2d_train_test_samples
        import numpy as np

        # Generate dummy data set:
        x = np.linspace(-1,1,100)
        y = -x**2 + 1

        # Generate dummy clustering of the data set:
        idx = variable_bins(x, 4, verbose=False)

        # Generate dummy sampling of the data set:
        sample = DataSampler(idx, idx_test=[], random_seed=None, verbose=True)
        (idx_train, idx_test) = sample.number(40, test_selection_option=1)

        # Plot the sampling result:
        plt = plot_2d_train_test_samples(x, y, idx, idx_train, idx_test, x_label='$x$', y_label='$y$', color_map='viridis', first_cluster_index_zero=False, grid_on=True, figure_size=(12,6), title='x-y data set', save_filename='sampling.pdf')
        plt.close()

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
        plot will not be saved. You can also set a desired file extension,
        for instance ``.pdf``. If the file extension is not specified, the default
        is ``.png``.

    :return:
        - **plt** - plot handle.
    """

    from matplotlib import cm

    n_clusters = len(np.unique(idx))
    populations = get_populations(idx)

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
