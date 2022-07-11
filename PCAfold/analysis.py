"""analysis.py: module for manifolds analysis."""

__author__ = "Kamila Zdybal, Elizabeth Armstrong, Alessandro Parente and James C. Sutherland"
__copyright__ = "Copyright (c) 2020-2022, Kamila Zdybal, Elizabeth Armstrong, Alessandro Parente and James C. Sutherland"
__credits__ = ["Department of Chemical Engineering, University of Utah, Salt Lake City, Utah, USA", "Universite Libre de Bruxelles, Aero-Thermo-Mechanics Laboratory, Brussels, Belgium"]
__license__ = "MIT"
__version__ = "1.6.0"
__maintainer__ = ["Kamila Zdybal", "Elizabeth Armstrong"]
__email__ = ["kamilazdybal@gmail.com", "Elizabeth.Armstrong@chemeng.utah.edu", "James.Sutherland@chemeng.utah.edu"]
__status__ = "Production"

import numpy as np
import copy as cp
import multiprocessing as multiproc
from PCAfold import KReg
from scipy.spatial import KDTree
from scipy.optimize import minimize
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import random as rnd
from scipy.interpolate import CubicSpline
from PCAfold.styles import *
from PCAfold import preprocess
from PCAfold import reduction
from termcolor import colored
from matplotlib.colors import ListedColormap
import time

################################################################################
#
# Manifold assessment
#
################################################################################

class VarianceData:
    """
    A class for storing helpful quantities in analyzing dimensionality of manifolds through normalized variance measures.
    This class will be returned by ``compute_normalized_variance``.

    :param bandwidth_values:
        the array of bandwidth values (Gaussian filter widths) used in computing the normalized variance for each variable
    :param normalized_variance:
        dictionary of the normalized variance computed at each of the bandwidth values for each variable
    :param global_variance:
        dictionary of the global variance for each variable
    :param bandwidth_10pct_rise:
        dictionary of the bandwidth value corresponding to a 10% rise in the normalized variance for each variable
    :param variable_names:
        list of the variable names
    :param normalized_variance_limit:
        dictionary of the normalized variance computed as the bandwidth approaches zero (numerically at :math:`10^{-16}`) for each variable
    :param sample_normalized_variance:
        dictionary of the sample normalized variance for every observation, for each bandwidth and for each variable
    """

    def __init__(self, bandwidth_values, norm_var, global_var, bandwidth_10pct_rise, keys, norm_var_limit, sample_norm_var):
        self._bandwidth_values = bandwidth_values.copy()
        self._normalized_variance = norm_var.copy()
        self._global_variance = global_var.copy()
        self._bandwidth_10pct_rise = bandwidth_10pct_rise.copy()
        self._variable_names = keys.copy()
        self._normalized_variance_limit = norm_var_limit.copy()
        self._sample_normalized_variance = sample_norm_var.copy()

    @property
    def bandwidth_values(self):
        """return the bandwidth values (Gaussian filter widths) used in computing the normalized variance for each variable"""
        return self._bandwidth_values.copy()

    @property
    def normalized_variance(self):
        """return a dictionary of the normalized variance computed at each of the bandwidth values for each variable"""
        return self._normalized_variance.copy()

    @property
    def global_variance(self):
        """return a dictionary of the global variance for each variable"""
        return self._global_variance.copy()

    @property
    def bandwidth_10pct_rise(self):
        """return a dictionary of the bandwidth value corresponding to a 10% rise in the normalized variance for each variable"""
        return self._bandwidth_10pct_rise.copy()

    @property
    def variable_names(self):
        """return a list of the variable names"""
        return self._variable_names.copy()

    @property
    def normalized_variance_limit(self):
        """return a dictionary of the normalized variance computed as the
        bandwidth approaches zero (numerically at 1.e-16) for each variable"""
        return self._normalized_variance_limit.copy()

    @property
    def sample_normalized_variance(self):
        """return a dictionary of the sample normalized variances for each bandwidth and for each variable"""
        return self._sample_normalized_variance.copy()

# ------------------------------------------------------------------------------

def compute_normalized_variance(indepvars, depvars, depvar_names, npts_bandwidth=25, min_bandwidth=None,
                                max_bandwidth=None, bandwidth_values=None, scale_unit_box=True, n_threads=None):
    """
    Compute a normalized variance (and related quantities) for analyzing manifold dimensionality.
    The normalized variance is computed as

    .. math::

        \\mathcal{N}(\\sigma) = \\frac{\\sum_{i=1}^n (y_i - \\mathcal{K}(\\hat{x}_i; \\sigma))^2}{\\sum_{i=1}^n (y_i - \\bar{y} )^2}

    where :math:`\\bar{y}` is the average quantity over the whole manifold and :math:`\\mathcal{K}(\\hat{x}_i; \\sigma)` is the
    weighted average quantity calculated using kernel regression with a Gaussian kernel of bandwidth :math:`\\sigma` centered
    around the :math:`i^{th}` observation. :math:`n` is the number of observations.
    :math:`\\mathcal{N}(\\sigma)` is computed for each bandwidth in an array of bandwidth values.
    By default, the ``indepvars`` (:math:`x`) are centered and scaled to reside inside a unit box (resulting in :math:`\\hat{x}`) so that the bandwidths have the
    same meaning in each dimension. Therefore, the bandwidth and its involved calculations are applied in the normalized
    independent variable space. This may be turned off by setting ``scale_unit_box`` to False.
    The bandwidth values may be specified directly through ``bandwidth_values`` or default values will be calculated as a
    logspace from ``min_bandwidth`` to ``max_bandwidth`` with ``npts_bandwidth`` number of values. If left unspecified,
    ``min_bandwidth`` and ``max_bandwidth`` will be calculated as the minimum and maximum nonzero distance between points, respectively.

    More information can be found in :cite:`Armstrong2021`.

    **Example:**

    .. code:: python

        from PCAfold import PCA, compute_normalized_variance
        import numpy as np

        # Generate dummy data set:
        X = np.random.rand(100,5)

        # Perform PCA to obtain the low-dimensional manifold:
        pca_X = PCA(X, n_components=2)
        principal_components = pca_X.transform(X)

        # Compute normalized variance quantities:
        variance_data = compute_normalized_variance(principal_components, X, depvar_names=['A', 'B', 'C', 'D', 'E'], bandwidth_values=np.logspace(-3, 1, 20), scale_unit_box=True)

        # Access bandwidth values:
        variance_data.bandwidth_values

        # Access normalized variance values:
        variance_data.normalized_variance

        # Access normalized variance values for a specific variable:
        variance_data.normalized_variance['B']

    :param indepvars:
        ``numpy.ndarray`` specifying the independent variable values. It should be of size ``(n_observations,n_independent_variables)``.
    :param depvars:
        ``numpy.ndarray`` specifying the dependent variable values. It should be of size ``(n_observations,n_dependent_variables)``.
    :param depvar_names:
        ``list`` of ``str`` corresponding to the names of the dependent variables (for saving values in a dictionary)
    :param npts_bandwidth:
        (optional, default 25) number of points to build a logspace of bandwidth values
    :param min_bandwidth:
        (optional, default to minimum nonzero interpoint distance) minimum bandwidth
    :param max_bandwidth:
        (optional, default to estimated maximum interpoint distance) maximum bandwidth
    :param bandwidth_values:
        (optional) array of bandwidth values, i.e. filter widths for a Gaussian filter, to loop over
    :param scale_unit_box:
        (optional, default True) center/scale the independent variables between [0,1] for computing a normalized variance so the bandwidth values have the same meaning in each dimension
    :param n_threads:
        (optional, default None) number of threads to run this computation. If None, default behavior of multiprocessing.Pool is used, which is to use all available cores on the current system.

    :return:
        - **variance_data** - an object of the ``VarianceData`` class.
    """
    assert indepvars.ndim == 2, "independent variable array must be 2D: n_observations x n_variables."
    assert depvars.ndim == 2, "dependent variable array must be 2D: n_observations x n_variables."
    assert (indepvars.shape[0] == depvars.shape[
        0]), "The number of observations for dependent and independent variables must match."
    assert (len(depvar_names) == depvars.shape[
        1]), "The provided keys do not match the shape of the dependent variables yi."

    if scale_unit_box:
        xi = (indepvars - np.min(indepvars, axis=0)) / (np.max(indepvars, axis=0) - np.min(indepvars, axis=0))
    else:
        xi = indepvars.copy()

    yi = depvars.copy()

    if bandwidth_values is None:
        if min_bandwidth is None:
            tree = KDTree(xi)
            min_bandwidth = np.min(tree.query(xi, k=2)[0][tree.query(xi, k=2)[0][:, 1] > 1.e-16, 1])
        if max_bandwidth is None:
            max_bandwidth = np.linalg.norm(np.max(xi, axis=0) - np.min(xi, axis=0)) * 10.
        bandwidth_values = np.logspace(np.log10(min_bandwidth), np.log10(max_bandwidth), npts_bandwidth)
    else:
        if not isinstance(bandwidth_values, np.ndarray):
            raise ValueError("bandwidth_values must be an array.")

    lvar = np.zeros((bandwidth_values.size, yi.shape[1]))
    kregmod = KReg(xi, yi)  # class for kernel regression evaluations

    # define a list of argments for kregmod_predict
    fcnArgs = [(xi, bandwidth_values[si]) for si in range(bandwidth_values.size) ]

    pool = multiproc.Pool(processes=n_threads)
    kregmodResults = pool.starmap( kregmod.predict, fcnArgs)

    pool.close()
    pool.join()

    for si in range(bandwidth_values.size):
        lvar[si, :] = np.linalg.norm(yi - kregmodResults[si], axis=0) ** 2

    # saving the local variance for each yi...
    local_var = dict({key: lvar[:, idx] for idx, key in enumerate(depvar_names)})
    # saving the global variance for each yi...
    global_var = dict(
        {key: np.linalg.norm(yi[:, idx] - np.mean(yi[:, idx])) ** 2 for idx, key in enumerate(depvar_names)})
    # saving the values of the bandwidth where the normalized variance increases by 10%...
    bandwidth_10pct_rise = dict()
    for key in depvar_names:
        bandwidth_idx = np.argwhere(local_var[key] / global_var[key] >= 0.1)
        if len(bandwidth_idx) == 0.:
            bandwidth_10pct_rise[key] = None
        else:
            bandwidth_10pct_rise[key] = bandwidth_values[bandwidth_idx[0]][0]
    norm_local_var = dict({key: local_var[key] / global_var[key] for key in depvar_names})

    # Computing normalized variance for each individual observation:
    sample_norm_var = {}
    for idx, key in enumerate(depvar_names):
        sample_local_variance = np.zeros((yi.shape[0], bandwidth_values.size))
        for si in range(bandwidth_values.size):
            sample_local_variance[:,si] = (yi[:, idx] - kregmodResults[si][:,idx])**2
        sample_norm_var[key] = sample_local_variance / global_var[key]

    # computing normalized variance as bandwidth approaches zero to check for non-uniqueness
    lvar_limit = kregmod.predict(xi, 1.e-16)
    nlvar_limit = np.linalg.norm(yi - lvar_limit, axis=0) ** 2
    normvar_limit = dict({key: nlvar_limit[idx] for idx, key in enumerate(depvar_names)})

    solution_data = VarianceData(bandwidth_values, norm_local_var, global_var, bandwidth_10pct_rise, depvar_names, normvar_limit, sample_norm_var)
    return solution_data

# ------------------------------------------------------------------------------

def normalized_variance_derivative(variance_data):
    """
    Compute a scaled normalized variance derivative on a logarithmic scale, :math:`\\hat{\\mathcal{D}}(\\sigma)`, from

    .. math::

        \\mathcal{D}(\\sigma) = \\frac{\\mathrm{d}\\mathcal{N}(\\sigma)}{\\mathrm{d}\\log_{10}(\\sigma)} + \lim_{\\sigma \\to 0} \\mathcal{N}(\\sigma)

    and

    .. math::

        \\hat{\\mathcal{D}}(\\sigma) = \\frac{\\mathcal{D}(\\sigma)}{\\max(\\mathcal{D}(\\sigma))}

    This value relays how fast the variance is changing as the bandwidth changes and captures non-uniqueness from
    nonzero values of :math:`\lim_{\\sigma \\to 0} \\mathcal{N}(\\sigma)`. The derivative is approximated
    with central finite differencing and the limit is approximated by :math:`\\mathcal{N}(\\sigma=10^{-16})` using the
    ``normalized_variance_limit`` attribute of the ``VarianceData`` object.

    More information can be found in :cite:`Armstrong2021`.

    **Example:**

    .. code:: python

        from PCAfold import PCA, compute_normalized_variance, normalized_variance_derivative
        import numpy as np

        # Generate dummy data set:
        X = np.random.rand(100,5)

        # Perform PCA to obtain the low-dimensional manifold:
        pca_X = PCA(X, n_components=2)
        principal_components = pca_X.transform(X)

        # Compute normalized variance quantities:
        variance_data = compute_normalized_variance(principal_components, X, depvar_names=['A', 'B', 'C', 'D', 'E'], bandwidth_values=np.logspace(-3, 1, 20), scale_unit_box=True)

        # Compute normalized variance derivative:
        (derivative, bandwidth_values, max_derivative) = normalized_variance_derivative(variance_data)

        # Access normalized variance derivative values for a specific variable:
        derivative['B']

    :param variance_data:
        a ``VarianceData`` class returned from ``compute_normalized_variance``

    :return:
        - **derivative_dict** - a dictionary of :math:`\\hat{\\mathcal{D}}(\\sigma)` for each variable in the provided ``VarianceData`` object
        - **x** - the :math:`\\sigma` values where :math:`\\hat{\\mathcal{D}}(\\sigma)` was computed
        - **max_derivatives_dicts** - a dictionary of :math:`\\max(\\mathcal{D}(\\sigma))` values for each variable in the provided ``VarianceData`` object.
    """
    x_plus = variance_data.bandwidth_values[2:]
    x_minus = variance_data.bandwidth_values[:-2]
    x = variance_data.bandwidth_values[1:-1]
    derivative_dict = {}
    max_derivatives_dict = {}
    for key in variance_data.variable_names:
        y_plus = variance_data.normalized_variance[key][2:]
        y_minus = variance_data.normalized_variance[key][:-2]
        derivative = (y_plus-y_minus)/(np.log10(x_plus)-np.log10(x_minus)) + variance_data.normalized_variance_limit[key]
        scaled_derivative = derivative/np.max(derivative)
        derivative_dict[key] = scaled_derivative
        max_derivatives_dict[key] = np.max(derivative)
    return derivative_dict, x, max_derivatives_dict

# ------------------------------------------------------------------------------

def find_local_maxima(dependent_values, independent_values, logscaling=True, threshold=1.e-2, show_plot=False):
    """
    Finds and returns locations and values of local maxima in a dependent variable given a set of observations.
    The functional form of the dependent variable is approximated with a cubic spline for smoother approximations to local maxima.

    :param dependent_values:
        observations of a single dependent variable such as :math:`\\hat{\\mathcal{D}}` from ``normalized_variance_derivative`` (for a single variable).
    :param independent_values:
        observations of a single independent variable such as :math:`\\sigma` returned by ``normalized_variance_derivative``
    :param logscaling:
        (optional, default True) this logarithmically scales ``independent_values`` before finding local maxima. This is needed for scaling :math:`\\sigma` appropriately before finding peaks in :math:`\\hat{\\mathcal{D}}`.
    :param threshold:
        (optional, default :math:`10^{-2}`) local maxima found below this threshold will be ignored.
    :param show_plot:
        (optional, default False) when True, a plot of the ``dependent_values`` over ``independent_values`` (logarithmically scaled if ``logscaling`` is True) with the local maxima highlighted will be shown.

    :return:
        - the locations of local maxima in ``dependent_values``
        - the local maxima values
    """
    if logscaling:
        independent_values = np.log10(independent_values.copy())
    zero_indices = []
    upslope = True
    npts = independent_values.size
    for i in range(1, npts):
        if upslope and dependent_values[i] - dependent_values[i - 1] <= 0:
            if dependent_values[i] > threshold:
                zero_indices.append(i - 1)
            upslope = False
        if not upslope and dependent_values[i] - dependent_values[i - 1] >= 0:
            upslope = True

    zero_locations = []
    zero_Dvalues = []
    for idx in zero_indices:
        if idx < 1:
            indices = [idx, idx + 1, idx + 2, idx + 3]
        elif idx < 2:
            indices = [idx - 1, idx, idx + 1, idx + 2]
        elif idx > npts - 1:
            indices = [idx - 3, idx - 2, idx - 1, idx]
        else:
            indices = [idx - 2, idx - 1, idx, idx + 1]
        Dspl = CubicSpline(independent_values[indices], dependent_values[indices])
        sigma_max = minimize(lambda s: -Dspl(s), independent_values[idx])
        zero_locations.append(sigma_max.x[0])
        zero_Dvalues.append(Dspl(sigma_max.x[0]))
    if show_plot:
        plt.plot(independent_values, dependent_values, 'k-')
        plt.plot(zero_locations, zero_Dvalues, 'r*')
        plt.xlim([np.min(independent_values),np.max(independent_values)])
        plt.ylim([0., 1.05])
        plt.grid()
        if logscaling:
            plt.xlabel('log$_{10}$(independent variable)')
        else:
            plt.xlabel('independent variable')
        plt.ylabel('dependent variable')
        plt.show()
    if logscaling:
        zero_locations = 10. ** np.array(zero_locations)
    return np.array(zero_locations, dtype=float), np.array(zero_Dvalues, dtype=float)

# ------------------------------------------------------------------------------

def random_sampling_normalized_variance(sampling_percentages, indepvars, depvars, depvar_names,
                                        n_sample_iterations=1, verbose=True, npts_bandwidth=25, min_bandwidth=None,
                                        max_bandwidth=None, bandwidth_values=None, scale_unit_box=True, n_threads=None):
    """
    Compute the normalized variance derivatives :math:`\\hat{\\mathcal{D}}(\\sigma)` for random samples of the provided
    data specified using ``sampling_percentages``. These will be averaged over ``n_sample_iterations`` iterations. Analyzing
    the shift in peaks of :math:`\\hat{\\mathcal{D}}(\\sigma)` due to sampling can distinguish between characteristic
    features and non-uniqueness due to a transformation/reduction of manifold coordinates. True features should not show
    significant sensitivity to sampling while non-uniqueness/folds in the manifold will.

    More information can be found in :cite:`Armstrong2021`.

    :param sampling_percentages:
        list or 1D array of fractions (between 0 and 1) of the provided data to sample for computing the normalized variance
    :param indepvars:
        independent variable values (size: n_observations x n_independent variables)
    :param depvars:
        dependent variable values (size: n_observations x n_dependent variables)
    :param depvar_names:
        list of strings corresponding to the names of the dependent variables (for saving values in a dictionary)
    :param n_sample_iterations:
        (optional, default 1) how many iterations for each ``sampling_percentages`` to average the normalized variance derivative over
    :param verbose:
        (optional, default True) when True, progress statements are printed
    :param npts_bandwidth:
        (optional, default 25) number of points to build a logspace of bandwidth values
    :param min_bandwidth:
        (optional, default to minimum nonzero interpoint distance) minimum bandwidth
    :param max_bandwidth:
        (optional, default to estimated maximum interpoint distance) maximum bandwidth
    :param bandwidth_values:
        (optional) array of bandwidth values, i.e. filter widths for a Gaussian filter, to loop over
    :param scale_unit_box:
        (optional, default True) center/scale the independent variables between [0,1] for computing a normalized variance so the bandwidth values have the same meaning in each dimension
    :param n_threads:
        (optional, default None) number of threads to run this computation. If None, default behavior of multiprocessing.Pool is used, which is to use all available cores on the current system.

    :return:
        - a dictionary of the normalized variance derivative (:math:`\\hat{\\mathcal{D}}(\\sigma)`) for each sampling percentage in ``sampling_percentages`` averaged over ``n_sample_iterations`` iterations
        - the :math:`\\sigma` values used for computing :math:`\\hat{\\mathcal{D}}(\\sigma)`
        - a dictionary of the ``VarianceData`` objects for each sampling percentage and iteration in ``sampling_percentages`` and ``n_sample_iterations``
    """
    assert indepvars.ndim == 2, "independent variable array must be 2D: n_observations x n_variables."
    assert depvars.ndim == 2, "dependent variable array must be 2D: n_observations x n_variables."

    if isinstance(sampling_percentages, list):
        for p in sampling_percentages:
            assert p > 0., "sampling percentages must be between 0 and 1"
            assert p <= 1., "sampling percentages must be between 0 and 1"
    elif isinstance(sampling_percentages, np.ndarray):
        assert sampling_percentages.ndim ==1, "sampling_percentages must be given as a list or 1D array"
        for p in sampling_percentages:
            assert p > 0., "sampling percentages must be between 0 and 1"
            assert p <= 1., "sampling percentages must be between 0 and 1"
    else:
        raise ValueError("sampling_percentages must be given as a list or 1D array.")

    normvar_data = {}
    avg_der_data = {}

    for p in sampling_percentages:
        if verbose:
            print('sampling', p * 100., '% of the data')
        nv_data = {}
        avg_der = {}

        for it in range(n_sample_iterations):
            if verbose:
                print('  iteration', it + 1, 'of', n_sample_iterations)
            rnd.seed(it)
            idxsample = rnd.sample(list(np.arange(0, indepvars.shape[0])), int(p * indepvars.shape[0]))
            nv_data[it] = compute_normalized_variance(indepvars[idxsample, :], depvars[idxsample, :], depvar_names,
                                                      npts_bandwidth=npts_bandwidth, min_bandwidth=min_bandwidth,
                                                      max_bandwidth=max_bandwidth, bandwidth_values=bandwidth_values,
                                                      scale_unit_box=scale_unit_box, n_threads=n_threads)

            der, xder, _ = normalized_variance_derivative(nv_data[it])
            for key in der.keys():
                if it == 0:
                    avg_der[key] = der[key] / np.float(n_sample_iterations)
                else:
                    avg_der[key] += der[key] / np.float(n_sample_iterations)

        avg_der_data[p] = avg_der
        normvar_data[p] = nv_data
    return avg_der_data, xder, normvar_data

# ------------------------------------------------------------------------------

def average_knn_distance(indepvars, n_neighbors=10, verbose=False):
    """
    Computes an average Euclidean distances to :math:`k` nearest neighbors on
    a manifold defined by the independent variables.

    **Example:**

    .. code:: python

        from PCAfold import PCA, average_knn_distance
        import numpy as np

        # Generate dummy data set:
        X = np.random.rand(100,20)

        # Instantiate PCA class object:
        pca_X = PCA(X, scaling='none', n_components=2, use_eigendec=True, nocenter=False)

        # Calculate the principal components:
        principal_components = pca_X.transform(X)

        # Compute average distances on a manifold defined by the PCs:
        average_distances = average_knn_distance(principal_components, n_neighbors=10, verbose=True)

    With ``verbose=True``, minimum, maximum and average distance will be printed:

    .. code-block:: text

        Minimum distance:	0.1388300829487847
        Maximum distance:	0.4689587542132183
        Average distance:	0.20824964953425693
        Median distance:	0.18333873029179215

    .. note::

        This function requires the ``scikit-learn`` module. You can install it through:

        ``pip install scikit-learn``

    :param indepvars:
        ``numpy.ndarray`` specifying the independent variable values. It should be of size ``(n_observations,n_independent_variables)``.
    :param n_neighbors: (optional)
        ``int`` specifying the number of nearest neighbors, :math:`k`.
    :param verbose: (optional)
        ``bool`` for printing verbose details.

    :return:
        - **average_distances** - ``numpy.ndarray`` specifying the vector of average distances for every observation in a data set to its :math:`k` nearest neighbors. It has size ``(n_observations,)``.
    """

    if not isinstance(indepvars, np.ndarray):
        raise ValueError("Parameter `indepvars` has to be of type `numpy.ndarray`.")

    try:
        (n_observations, n_independent_variables) = np.shape(indepvars)
    except:
        raise ValueError("Parameter `indepvars` has to have size `(n_observations,n_independent_variables)`.")

    if not isinstance(n_neighbors, int):
        raise ValueError("Parameter `n_neighbors` has to be of type int.")

    if n_neighbors < 2:
        raise ValueError("Parameter `n_neighbors` cannot be smaller than 2.")

    if not isinstance(verbose, bool):
        raise ValueError("Parameter `verbose` has to be a boolean.")

    try:
        from sklearn.neighbors import NearestNeighbors
    except:
        raise ValueError("Nearest neighbors search requires the `sklearn` module: `pip install scikit-learn`.")

    (n_observations, n_independent_variables) = np.shape(indepvars)

    knn_model = NearestNeighbors(n_neighbors=n_neighbors+1)
    knn_model.fit(indepvars)

    average_distances = np.zeros((n_observations,))

    for query_point in range(0,n_observations):

        (distances_neigh, idx_neigh) = knn_model.kneighbors(indepvars[query_point,:][None,:], n_neighbors=n_neighbors+1, return_distance=True)
        query_point_idx = np.where(idx_neigh.ravel()==query_point)
        distances_neigh = np.delete(distances_neigh.ravel(), np.s_[query_point_idx])
        idx_neigh = np.delete(idx_neigh.ravel(), np.s_[query_point_idx])
        average_distances[query_point] = np.mean(distances_neigh)

    if verbose:
        print('Minimum distance:\t' + str(np.min(average_distances)))
        print('Maximum distance:\t' + str(np.max(average_distances)))
        print('Average distance:\t' + str(np.mean(average_distances)))
        print('Median distance:\t' + str(np.median(average_distances)))

    return average_distances

# ------------------------------------------------------------------------------

def cost_function_normalized_variance_derivative(variance_data, penalty_function=None, power=1, vertical_shift=1, norm=None, integrate_to_peak=False):
    """
    Defines a cost function for manifold topology assessment based on the areas, or weighted (penalized) areas, under
    the normalized variance derivatives curves, :math:`\\hat{\\mathcal{D}}(\\sigma)`, for the selected :math:`n_{dep}` dependent variables.

    An individual area, :math:`A_i`, for the :math:`i^{th}` dependent variable, is computed by directly integrating the function :math:`\\hat{\\mathcal{D}}_i(\\sigma)``
    in the :math:`\\log_{10}` space of bandwidths :math:`\\sigma`. Integration is performed using the composite trapezoid rule.

    When ``integrate_to_peak=False``, the bounds of integration go from the minimum bandwidth, :math:`\\sigma_{min, i}`,
    to the maximum bandwidth, :math:`\\sigma_{max, i}`:

    .. math::

        A_i = \\int_{\\sigma_{min, i}}^{\\sigma_{max, i}} \\hat{\\mathcal{D}}_i(\\sigma) d \\log_{10} \\sigma

    .. image:: ../images/cost-function-D-hat.svg
        :width: 600
        :align: center

    When ``integrate_to_peak=True``, the bounds of integration go from the minimum bandwidth, :math:`\\sigma_{min, i}`,
    to the bandwidth for which the rightmost peak happens in :math:`\\hat{\\mathcal{D}}_i(\\sigma)``, :math:`\\sigma_{peak, i}`:

    .. math::

        A_i = \\int_{\\sigma_{min, i}}^{\\sigma_{peak, i}} \\hat{\\mathcal{D}}_i(\\sigma) d \\log_{10} \\sigma

    .. image:: ../images/cost-function-D-hat-to-peak.svg
        :width: 600
        :align: center

    In addition, each individual area, :math:`A_i`, can be weighted. The following weighting options are available:

    - If ``penalty_function='peak'``, :math:`A_i` is weighted by the inverse of the rightmost peak location:

    .. math::

        A_i = \\frac{1}{\\sigma_{peak, i}} \\cdot \\int \\hat{\\mathcal{D}}_i(\\sigma) d(\\log_{10} \\sigma)

    This creates a constant penalty:

    .. image:: ../images/cost-function-peak.svg
        :width: 600
        :align: center

    - If ``penalty_function='sigma'``, :math:`A_i` is weighted continuously by the bandwidth:

    .. math::

        A_i = \\int \\frac{1}{\\sigma^r} \\cdot \\hat{\\mathcal{D}}_i(\\sigma) d(\\log_{10} \\sigma)

    where :math:`r` is a hyper-parameter that can be controlled by the user. This \
    type of weighting *strongly* penalizes the area happening at lower bandwidth values.

    For instance, when :math:`r=0.2`:

    .. image:: ../images/cost-function-sigma-penalty-r02.svg
        :width: 600
        :align: center

    When :math:`r=1` (with the penalty corresponding to :math:`r=0.2` plotted in gray in the background):

    .. image:: ../images/cost-function-sigma-penalty-r1.svg
        :width: 600
        :align: center

    - If ``penalty_function='log-sigma-over-peak'``, :math:`A_i` is weighted continuously by the :math:`\\log_{10}` -transformed bandwidth\
    and takes into account information about the rightmost peak location.

    .. math::

        A_i = \\int \\Big(  \\big| \\log_{10} \\Big( \\frac{\\sigma}{\\sigma_{peak, i}} \\Big) \\big|^r + b \\cdot \\frac{\\log_{10} \\sigma_{max, i} - \\log_{10} \\sigma_{min, i}}{\\log_{10} \\sigma_{peak, i} - \\log_{10} \\sigma_{min, i}} \\Big) \\cdot \\hat{\\mathcal{D}}_i(\\sigma) d(\\log_{10} \\sigma)

    This type of weighting creates a more gentle penalty for the area happening further from the rightmost peak location.
    By increasing :math:`b`, the user can increase the amount of penalty applied to smaller feature sizes over larger ones.
    By increasing :math:`r`, the user can penalize non-uniqueness more strongly.

    For instance, when :math:`r=1`:

    .. image:: ../images/cost-function-log-sigma-over-peak-penalty-r1.svg
        :width: 600
        :align: center

    When :math:`r=2` (with the penalty corresponding to :math:`r=1` plotted in gray in the background):

    .. image:: ../images/cost-function-log-sigma-over-peak-penalty-r2.svg
        :width: 600
        :align: center

    If ``norm=None``, a list of costs for all dependent variables is returned.
    Otherwise, the final cost, :math:`\\mathcal{L}`, can be computed from all :math:`A_i` in a few ways,
    where :math:`n_{dep}` is the number of dependent variables stored in the ``variance_data`` object:

    - If ``norm='average'``, :math:`\\mathcal{L} = \\frac{1}{n_{dep}} \\sum_{i = 1}^{n_{dep}} A_i`.

    - If ``norm='cumulative'``, :math:`\\mathcal{L} = \\sum_{i = 1}^{n_{dep}} A_i`.

    - If ``norm='max'``, :math:`\\mathcal{L} = \\text{max} (A_i)`.

    - If ``norm='median'``, :math:`\\mathcal{L} = \\text{median} (A_i)`.

    - If ``norm='min'``, :math:`\\mathcal{L} = \\text{min} (A_i)`.

    **Example:**

    .. code:: python

        from PCAfold import PCA, compute_normalized_variance, cost_function_normalized_variance_derivative
        import numpy as np

        # Generate dummy data set:
        X = np.random.rand(100,10)

        # Specify variables names
        variable_names = ['X_' + str(i) for i in range(0,10)]

        # Perform PCA to obtain the low-dimensional manifold:
        pca_X = PCA(X, n_components=2)
        principal_components = pca_X.transform(X)

        # Specify the bandwidth values:
        bandwidth_values = np.logspace(-4, 2, 50)

        # Compute normalized variance quantities:
        variance_data = compute_normalized_variance(principal_components,
                                                    X,
                                                    depvar_names=variable_names,
                                                    bandwidth_values=bandwidth_values)

        # Compute the cost for the current manifold:
        cost = cost_function_normalized_variance_derivative(variance_data,
                                                            penalty_function='sigma',
                                                            power=0.5,
                                                            vertical_shift=1,
                                                            norm='max',
                                                            integrate_to_peak=True)

    :param variance_data:
        an object of ``VarianceData`` class.
    :param penalty_function: (optional)
        ``str`` specifying the weighting (penalty) applied to each area.
        Set ``penalty_function='peak'`` to weight each area by the rightmost peak location, :math:`\\sigma_{peak, i}`, for the :math:`i^{th}` dependent variable.
        Set ``penalty_function='sigma'`` to weight each area continuously by the bandwidth.
        Set ``penalty_function='log-sigma-over-peak'`` to weight each area continuously by the :math:`\\log_{10}` -transformed bandwidth, normalized by the right most peak location, :math:`\\sigma_{peak, i}`.
        If ``penalty_function=None``, the area is not weighted.
    :param power: (optional)
        ``float`` or ``int`` specifying the power, :math:`r`. It can be used to control how much penalty should be applied to variance happening at the smallest length scales.
    :param vertical_shift: (optional)
        ``float`` or ``int`` specifying the vertical shift multiplier, :math:`b`. It can be used to control how much penalty should be applied to feature sizes.
    :param norm: (optional)
        ``str`` specifying the norm to apply for all areas :math:`A_i`. ``norm='average'`` uses an arithmetic average, ``norm='max'`` uses the :math:`L_{\\infty}` norm,
        ``norm='median'`` uses a median area, ``norm='cumulative'`` uses a cumulative area and ``norm='min'`` uses a minimum area. If ``norm=None``, a list of costs for all depedent variables is returned.
    :param integrate_to_peak: (optional)
        ``bool`` specifying whether an individual area for the :math:`i^{th}` dependent variable should be computed only up the the rightmost peak location.

    :return:
        - **cost** - ``float`` specifying the normalized cost, :math:`\\mathcal{L}`, or, if ``norm=None``, a list of costs, :math:`A_i`, for each dependent variable.
    """

    __penalty_functions = ['peak', 'sigma', 'log-sigma-over-peak']
    __norms = ['average', 'cumulative', 'max', 'median', 'min']

    if penalty_function is not None:

        if not isinstance(penalty_function, str):
            raise ValueError("Parameter `penalty_function` has to be of type `str`.")

        if penalty_function not in __penalty_functions:
            raise ValueError("Parameter `penalty_function` has to be one of the following: 'peak', 'sigma', 'log-sigma-over-peak'.")

    if not isinstance(vertical_shift, int) and not isinstance(vertical_shift, float):
        raise ValueError("Parameter `vertical_shift` has to be of type `float` or `int`.")

    if not isinstance(power, int) and not isinstance(power, float):
        raise ValueError("Parameter `power` has to be of type `float` or `int`.")

    if norm is not None:

        if not isinstance(norm, str):
            raise ValueError("Parameter `norm` has to be of type `str`.")

        if norm not in __norms:
            raise ValueError("Parameter `norm` has to be one of the following: 'average', 'cumulative', 'max', 'median', 'min'.")

    if not isinstance(integrate_to_peak, bool):
        raise ValueError("Parameter `integrate_to_peak` has to be of type `bool`.")

    derivative, sigma, _ = normalized_variance_derivative(variance_data)

    costs = []

    for variable in variance_data.variable_names:

        idx_peaks, _ = find_peaks(derivative[variable], height=0)
        idx_rightmost_peak = idx_peaks[-1]
        rightmost_peak_location = sigma[idx_rightmost_peak]

        (indices_to_the_left_of_peak, ) = np.where(sigma<=rightmost_peak_location)

        if integrate_to_peak:

            if penalty_function is None:
                cost = np.trapz(derivative[variable][indices_to_the_left_of_peak], np.log10(sigma[indices_to_the_left_of_peak]))
                costs.append(cost)

            elif penalty_function == 'peak':
                cost = 1. / (rightmost_peak_location) * np.trapz(derivative[variable][indices_to_the_left_of_peak], np.log10(sigma[indices_to_the_left_of_peak]))
                costs.append(cost)

            elif penalty_function == 'sigma':
                penalty_sigma = 1. / (sigma[indices_to_the_left_of_peak]**power)
                cost = np.trapz(derivative[variable][indices_to_the_left_of_peak]*penalty_sigma, np.log10(sigma[indices_to_the_left_of_peak]))
                costs.append(cost)

            elif penalty_function == 'log-sigma-over-peak':
                normalized_sigma, _, _ = preprocess.center_scale(np.log10(sigma[:,None]), scaling='0to1')
                addition = normalized_sigma[idx_rightmost_peak][0]
                penalty_log_sigma_peak = (abs(np.log10(sigma[indices_to_the_left_of_peak]/rightmost_peak_location)))**power + vertical_shift * 1. / addition
                cost = np.trapz(derivative[variable][indices_to_the_left_of_peak]*penalty_log_sigma_peak, np.log10(sigma[indices_to_the_left_of_peak]))
                costs.append(cost)

        else:

            if penalty_function is None:
                cost = np.trapz(derivative[variable], np.log10(sigma))
                costs.append(cost)

            elif penalty_function == 'peak':
                cost = 1. / (rightmost_peak_location) * np.trapz(derivative[variable], np.log10(sigma))
                costs.append(cost)

            elif penalty_function == 'sigma':
                penalty_sigma = 1. / (sigma**power)
                cost = np.trapz(derivative[variable]*penalty_sigma, np.log10(sigma))
                costs.append(cost)

            elif penalty_function == 'log-sigma-over-peak':
                normalized_sigma, _, _ = preprocess.center_scale(np.log10(sigma[:,None]), scaling='0to1')
                addition = normalized_sigma[idx_rightmost_peak][0]
                penalty_log_sigma_peak = (abs(np.log10(sigma/rightmost_peak_location)))**power + vertical_shift * 1. / addition
                cost = np.trapz(derivative[variable]*penalty_log_sigma_peak, np.log10(sigma))
                costs.append(cost)

    if norm is None:

        return costs

    else:

        if norm == 'max':

            # Take L-infinity norm over all costs:
            normalized_cost = np.max(costs)

        elif norm == 'average':

            # Take the arithmetic average norm over all costs:
            normalized_cost = np.mean(costs)

        elif norm == 'min':

            # Take the minimum norm over all costs:
            normalized_cost = np.min(costs)

        elif norm == 'median':

            # Take the median norm over all costs:
            normalized_cost = np.median(costs)

        elif norm == 'cumulative':

            # Take the cumulative sum over all costs:
            normalized_cost = np.sum(costs)

        return normalized_cost

# ------------------------------------------------------------------------------

def manifold_informed_feature_selection(X, X_source, variable_names, scaling, bandwidth_values, target_variables=None, add_transformed_source=True, target_manifold_dimensionality=3, bootstrap_variables=None, penalty_function=None, norm='max', integrate_to_peak=False, verbose=False):
    """
    Manifold-informed feature selection algorithm based on forward feature addition. The goal of the algorithm is to
    select a meaningful subset of the original variables such that
    undesired behaviors on a PCA-derived manifold of a given dimensionality are minimized.
    The algorithm uses the cost function, :math:`\\mathcal{L}`, based on minimizing the area under the normalized variance derivatives curves, :math:`\\hat{\\mathcal{D}}(\\sigma)`,
    for the selected :math:`n_{dep}` dependent variables (as per ``cost_function_normalized_variance_derivative`` function).
    The algorithm can be bootstrapped in two ways:

    - Automatic bootstrap when ``bootstrap_variables=None``: the first best variable is selected automatically as the one that gives the lowest cost.

    - User-defined bootstrap when ``bootstrap_variables`` is set to a user-defined list of the bootstrap variables.

    The algorithm iterates, adding a new variable that exhibits the lowest cost at each iteration.
    The original variables in a data set get ordered according to their effect
    on the manifold topology. Assuming that the original data set is composed of :math:`Q` variables,
    the first output is a list of indices of the ordered
    original variables, :math:`\\mathbf{X} = [X_1, X_2, \\dots, X_Q]`. The second output is a list of indices of the selected
    subset of the original variables, :math:`\\mathbf{X}_S = [X_1, X_2, \\dots, X_n]`, that correspond to the minimum cost, :math:`\\mathcal{L}`.

    .. note::

        The algorithm can be very expensive (for large data sets) due to multiple computations of the normalized variance derivative.
        Try running it on multiple cores or on a sampled data set.

        In case the algorithm breaks when not being able to determine the peak
        location, try increasing the range in the ``bandwidth_values`` parameter.

    **Example:**

    .. code:: python

        from PCAfold import manifold_informed_feature_selection
        import numpy as np

        # Generate dummy data set:
        X = np.random.rand(100,10)
        X_source = np.random.rand(100,10)

        # Define original variables to add to the optimization:
        target_variables = X[:,0:3]

        # Specify variables names
        variable_names = ['X_' + str(i) for i in range(0,10)]

        # Specify the bandwidth values to compute the optimization on:
        bandwidth_values = np.logspace(-4, 2, 50)

        # Run the subset selection algorithm:
        (ordered, selected, costs) = manifold_informed_feature_selection(X,
                                                                         X_source,
                                                                         variable_names,
                                                                         scaling='auto',
                                                                         bandwidth_values=bandwidth_values,
                                                                         target_variables=target_variables,
                                                                         add_transformed_source=True,
                                                                         target_manifold_dimensionality=2,
                                                                         bootstrap_variables=None,
                                                                         penalty_function='peak',
                                                                         norm='max',
                                                                         integrate_to_peak=True,
                                                                         verbose=True)

    :param X:
        ``numpy.ndarray`` specifying the original data set, :math:`\\mathbf{X}`. It should be of size ``(n_observations,n_variables)``.
    :param X_source:
        ``numpy.ndarray`` specifying the source terms, :math:`\\mathbf{S_X}`, corresponding to the state-space
        variables in :math:`\\mathbf{X}`. This parameter is applicable to data sets
        representing reactive flows. More information can be found in :cite:`Sutherland2009`. It should be of size ``(n_observations,n_variables)``.
    :param variable_names:
        ``list`` of ``str`` specifying variables names.
    :param scaling: (optional)
        ``str`` specifying the scaling methodology. It can be one of the following:
        ``'none'``, ``''``, ``'auto'``, ``'std'``, ``'pareto'``, ``'vast'``, ``'range'``, ``'0to1'``,
        ``'-1to1'``, ``'level'``, ``'max'``, ``'poisson'``, ``'vast_2'``, ``'vast_3'``, ``'vast_4'``.
    :param bandwidth_values:
        ``numpy.ndarray`` specifying the bandwidth values, :math:`\\sigma`, for :math:`\\hat{\\mathcal{D}}(\\sigma)` computation.
    :param target_variables: (optional)
        ``numpy.ndarray`` specifying the dependent variables that should be used in :math:`\\hat{\\mathcal{D}}(\\sigma)` computation. It should be of size ``(n_observations,n_target_variables)``.
    :param add_transformed_source: (optional)
        ``bool`` specifying if the PCA-transformed source terms of the state-space variables should be added in :math:`\\hat{\\mathcal{D}}(\\sigma)` computation, alongside the user-defined dependent variables.
    :param target_manifold_dimensionality: (optional)
        ``int`` specifying the target dimensionality of the PCA manifold.
    :param bootstrap_variables: (optional)
        ``list`` specifying the user-selected variables to bootstrap the algorithm with. If set to ``None``, automatic bootstrapping is performed.
    :param penalty_function: (optional)
        ``str`` specifying the weighting applied to each area.
        Set ``penalty_function='peak'`` to weight each area by the rightmost peak location, :math:`\\sigma_{peak, i}`, for the :math:`i^{th}` dependent variable.
        Set ``penalty_function='sigma'`` to weight each area continuously by the bandwidth.
        Set ``penalty_function='log-sigma-over-peak'`` to weight each area continuously by the :math:`\\log_{10}` -transformed bandwidth, normalized by the right most peak location, :math:`\\sigma_{peak, i}`.
        If ``penalty_function=None``, the area is not weighted.
    :param norm: (optional)
        ``str`` specifying the norm to apply for all areas :math:`A_i`. ``norm='average'`` uses an arithmetic average, ``norm='max'`` uses the :math:`L_{\\infty}` norm,
        ``norm='median'`` uses a median area, ``norm='cumulative'`` uses a cumulative area and ``norm='min'`` uses a minimum area.
    :param integrate_to_peak: (optional)
        ``bool`` specifying whether an individual area for the :math:`i^{th}` dependent variable should be computed only up the the rightmost peak location.
    :param verbose: (optional)
        ``bool`` for printing verbose details.

    :return:
        - **ordered_variables** - ``list`` specifying the indices of the ordered variables.
        - **selected_variables** - ``list`` specifying the indices of the selected variables that correspond to the minimum cost :math:`\\mathcal{L}`.
        - **costs** - ``list`` specifying the costs, :math:`\\mathcal{L}`, from each iteration.
    """

    __penalty_functions = ['peak', 'sigma', 'log-sigma-over-peak']
    __norms = ['average', 'cumulative', 'max', 'median', 'min']

    if not isinstance(X, np.ndarray):
        raise ValueError("Parameter `X` has to be of type `numpy.ndarray`.")

    try:
        (n_observations, n_variables) = np.shape(X)
    except:
        raise ValueError("Parameter `X` has to have shape `(n_observations,n_variables)`.")

    if not isinstance(X_source, np.ndarray):
        raise ValueError("Parameter `X_source` has to be of type `numpy.ndarray`.")

    try:
        (n_observations_source, n_variables_source) = np.shape(X_source)
    except:
        raise ValueError("Parameter `X_source` has to have shape `(n_observations,n_variables)`.")

    if n_variables_source != n_variables:
        raise ValueError("Parameter `X_source` has different number of variables than `X`.")

    if n_observations_source != n_observations:
        raise ValueError("Parameter `X_source` has different number of observations than `X`.")

    if not isinstance(variable_names, list):
        raise ValueError("Parameter `variable_names` has to be of type `list`.")

    if len(variable_names) != n_variables:
        raise ValueError("Parameter `variable_names` has different number of variables than `X`.")

    if not isinstance(scaling, str):
        raise ValueError("Parameter `scaling` has to be of type `str`.")

    if not isinstance(bandwidth_values, np.ndarray):
        raise ValueError("Parameter `bandwidth_values` has to be of type `numpy.ndarray`.")

    if target_variables is not None:
        if not isinstance(target_variables, np.ndarray):
            raise ValueError("Parameter `target_variables` has to be of type `numpy.ndarray`.")

        try:
            (n_d_hat_observations, n_target_variables) = np.shape(target_variables)
            target_variables_names = ['X' + str(i) for i in range(0,n_target_variables)]
        except:
            raise ValueError("Parameter `target_variables` has to have shape `(n_observations,n_target_variables)`.")

        if n_d_hat_observations != n_observations_source:
            raise ValueError("Parameter `target_variables` has different number of observations than `X_source`.")

    if not isinstance(add_transformed_source, bool):
        raise ValueError("Parameter `add_transformed_source` has to be of type `bool`.")

    if target_variables is None:
        if not add_transformed_source:
            raise ValueError("Either `target_variables` has to be specified or `add_transformed_source` has to be set to True.")

    if not isinstance(target_manifold_dimensionality, int):
        raise ValueError("Parameter `target_manifold_dimensionality` has to be of type `int`.")

    if bootstrap_variables is not None:
        if not isinstance(bootstrap_variables, list):
            raise ValueError("Parameter `bootstrap_variables` has to be of type `list`.")

    if penalty_function is not None:

        if not isinstance(penalty_function, str):
            raise ValueError("Parameter `penalty_function` has to be of type `str`.")

        if penalty_function not in __penalty_functions:
            raise ValueError("Parameter `penalty_function` has to be one of the following: 'peak', 'sigma', 'log-sigma-over-peak'.")

    if not isinstance(norm, str):
        raise ValueError("Parameter `norm` has to be of type `str`.")

    if norm not in __norms:
        raise ValueError("Parameter `norm` has to be one of the following: 'average', 'cumulative', 'max', 'median', 'min'.")

    if not isinstance(integrate_to_peak, bool):
        raise ValueError("Parameter `integrate_to_peak` has to be of type `bool`.")

    if not isinstance(verbose, bool):
        raise ValueError("Parameter `verbose` has to be of type `bool`.")

    variables_indices = [i for i in range(0,n_variables)]

    costs = []

    # Automatic bootstrapping: -------------------------------------------------
    if bootstrap_variables is None:

        if verbose: print('Automatic bootstrapping...\n')

        bootstrap_cost_function = []

        bootstrap_tic = time.perf_counter()

        for i_variable in variables_indices:

            if verbose: print('\tCurrently checking variable:\t' + variable_names[i_variable])

            PCs = X[:,[i_variable]]
            PC_sources = X_source[:,[i_variable]]

            if target_variables is None:
                depvars = cp.deepcopy(PC_sources)
                depvar_names = ['SZ1']
            else:
                if add_transformed_source:
                    depvars = np.hstack((PC_sources, target_variables))
                    depvar_names = ['SZ1'] + target_variables_names
                else:
                    depvars = target_variables
                    depvar_names = target_variables_names

            bootstrap_variance_data = compute_normalized_variance(PCs, depvars, depvar_names=depvar_names, bandwidth_values=bandwidth_values)

            bootstrap_area = cost_function_normalized_variance_derivative(bootstrap_variance_data, penalty_function=penalty_function, norm=norm, integrate_to_peak=integrate_to_peak)
            if verbose: print('\tCost:\t%.4f' % bootstrap_area)
            bootstrap_cost_function.append(bootstrap_area)

        # Find a single best variable to bootstrap with:
        (best_bootstrap_variable_index, ) = np.where(np.array(bootstrap_cost_function)==np.min(bootstrap_cost_function))
        best_bootstrap_variable_index = int(best_bootstrap_variable_index)

        costs.append(np.min(bootstrap_cost_function))

        bootstrap_variables = [best_bootstrap_variable_index]

        if verbose: print('\n\tVariable ' + variable_names[best_bootstrap_variable_index] + ' will be used as bootstrap.\n\tCost:\t%.4f' % np.min(bootstrap_cost_function) + '\n')

        bootstrap_toc = time.perf_counter()
        if verbose: print(f'Boostrapping time: {(bootstrap_toc - bootstrap_tic)/60:0.1f} minutes.' + '\n' + '-'*50)

    # Use user-defined bootstrapping: -----------------------------------------
    else:

        # Manifold dimensionality needs a fix here!
        if verbose: print('User-defined bootstrapping...\n')

        bootstrap_cost_function = []

        bootstrap_tic = time.perf_counter()

        if len(bootstrap_variables) < target_manifold_dimensionality:
            n_components = len(bootstrap_variables)
        else:
            n_components = cp.deepcopy(target_manifold_dimensionality)

        if verbose: print('\tUser-defined bootstrapping will be performed for a ' + str(n_components) + '-dimensional manifold.')

        bootstrap_pca = reduction.PCA(X[:,bootstrap_variables], scaling=scaling, n_components=n_components)
        PCs = bootstrap_pca.transform(X[:,bootstrap_variables])
        PC_sources = bootstrap_pca.transform(X_source[:,bootstrap_variables], nocenter=True)

        if target_variables is None:
            depvars = cp.deepcopy(PC_sources)
            depvar_names = ['SZ' + str(i) for i in range(0,n_components)]
        else:
            if add_transformed_source:
                depvars = np.hstack((PC_sources, target_variables))
                depvar_names = depvar_names = ['SZ' + str(i) for i in range(0,n_components)] + target_variables_names
            else:
                depvars = target_variables
                depvar_names = target_variables_names

        bootstrap_variance_data = compute_normalized_variance(PCs, depvars, depvar_names=depvar_names, bandwidth_values=bandwidth_values)

        bootstrap_area = cost_function_normalized_variance_derivative(bootstrap_variance_data, penalty_function=penalty_function, norm=norm, integrate_to_peak=integrate_to_peak)
        bootstrap_cost_function.append(bootstrap_area)
        costs.append(bootstrap_area)

        if verbose: print('\n\tVariable(s) ' + ', '.join([variable_names[i] for i in bootstrap_variables]) + ' will be used as bootstrap\n\tCost:\t%.4f' % np.min(bootstrap_area) + '\n')

        bootstrap_toc = time.perf_counter()
        if verbose: print(f'Boostrapping time: {(bootstrap_toc - bootstrap_tic)/60:0.1f} minutes.' + '\n' + '-'*50)

    # Iterate the algorithm starting from the bootstrap selection: -------------
    if verbose: print('Optimizing...\n')

    total_tic = time.perf_counter()

    ordered_variables = [i for i in bootstrap_variables]

    remaining_variables_list = [i for i in range(0,n_variables) if i not in bootstrap_variables]
    previous_area = np.min(bootstrap_cost_function)

    loop_counter = 0

    while len(remaining_variables_list) > 0:

        iteration_tic = time.perf_counter()

        loop_counter += 1

        if verbose:
            print('Iteration No.' + str(loop_counter))
            print('Currently adding variables from the following list: ')
            print([variable_names[i] for i in remaining_variables_list])

        current_cost_function = []

        for i_variable in remaining_variables_list:

            if len(ordered_variables) < target_manifold_dimensionality:
                n_components = len(ordered_variables) + 1
            else:
                n_components = cp.deepcopy(target_manifold_dimensionality)

            if verbose: print('\tCurrently added variable: ' + variable_names[i_variable])

            current_variables_list = ordered_variables + [i_variable]

            pca = reduction.PCA(X[:,current_variables_list], scaling=scaling, n_components=n_components)
            PCs = pca.transform(X[:,current_variables_list])
            PC_sources = pca.transform(X_source[:,current_variables_list], nocenter=True)

            if target_variables is None:
                depvars = cp.deepcopy(PC_sources)
                depvar_names = ['SZ' + str(i) for i in range(0,n_components)]
            else:
                if add_transformed_source:
                    depvars = np.hstack((PC_sources, target_variables))
                    depvar_names = depvar_names = ['SZ' + str(i) for i in range(0,n_components)] + target_variables_names
                else:
                    depvars = target_variables
                    depvar_names = target_variables_names

            current_variance_data = compute_normalized_variance(PCs, depvars, depvar_names=depvar_names, bandwidth_values=bandwidth_values)
            current_derivative, current_sigma, _ = normalized_variance_derivative(current_variance_data)

            current_area = cost_function_normalized_variance_derivative(current_variance_data, penalty_function=penalty_function, norm=norm, integrate_to_peak=integrate_to_peak)
            if verbose: print('\tCost:\t%.4f' % current_area)
            current_cost_function.append(current_area)

            if current_area <= previous_area:
                if verbose: print(colored('\tSAME OR BETTER', 'green'))
            else:
                if verbose: print(colored('\tWORSE', 'red'))

        min_area = np.min(current_cost_function)
        (best_variable_index, ) = np.where(np.array(current_cost_function)==min_area)
        try:
            best_variable_index = int(best_variable_index)
        except:
            best_variable_index = int(best_variable_index[0])

        if verbose: print('\n\tVariable ' + variable_names[remaining_variables_list[best_variable_index]] + ' is added.\n\tCost:\t%.4f' % min_area + '\n')
        ordered_variables.append(remaining_variables_list[best_variable_index])
        remaining_variables_list = [i for i in range(0,n_variables) if i not in ordered_variables]
        if min_area <= previous_area:
            previous_area = min_area
        costs.append(min_area)

        iteration_toc = time.perf_counter()
        if verbose: print(f'\tIteration time: {(iteration_toc - iteration_tic)/60:0.1f} minutes.' + '\n' + '-'*50)

    # Compute the optimal subset where the cost is minimized: ------------------
    (min_cost_function_index, ) = np.where(costs==np.min(costs))
    try:
        min_cost_function_index = int(min_cost_function_index)
    except:
        min_cost_function_index = int(min_cost_function_index[0])

    if min_cost_function_index+1 < target_manifold_dimensionality:
        selected_variables = list(np.array(ordered_variables)[0:target_manifold_dimensionality])
    else:
        selected_variables = list(np.array(ordered_variables)[0:min_cost_function_index+1])

    if verbose:

        print('Ordered variables:')
        print(', '.join([variable_names[i] for i in ordered_variables]))
        print(ordered_variables)
        print('Final cost: %.4f' % min_area)

        print('\nSelected variables:')
        print(', '.join([variable_names[i] for i in selected_variables]))
        print(selected_variables)
        print('Lowest cost: %.4f' % previous_area)

    total_toc = time.perf_counter()
    if verbose: print(f'\nOptimization time: {(total_toc - total_tic)/60:0.1f} minutes.' + '\n' + '-'*50)

    return ordered_variables, selected_variables, costs

# ------------------------------------------------------------------------------

def manifold_informed_backward_elimination(X, X_source, variable_names, scaling, bandwidth_values, target_variables=None, add_transformed_source=True, source_space=None, target_manifold_dimensionality=3, penalty_function=None, norm='max', integrate_to_peak=False, verbose=False):
    """
    Manifold-informed feature selection algorithm based on backward elimination. The goal of the algorithm is to
    select a meaningful subset of the original variables such that
    undesired behaviors on a PCA-derived manifold of a given dimensionality are minimized.
    The algorithm uses the cost function, :math:`\\mathcal{L}`, based on minimizing the area under the normalized variance derivatives curves, :math:`\\hat{\\mathcal{D}}(\\sigma)`,
    for the selected :math:`n_{dep}` dependent variables (as per ``cost_function_normalized_variance_derivative`` function).

    The algorithm iterates, removing another variable that has an effect of decreasing the cost the most at each iteration.
    The original variables in a data set get ordered according to their effect
    on the manifold topology. Assuming that the original data set is composed of :math:`Q` variables,
    the first output is a list of indices of the ordered
    original variables, :math:`\\mathbf{X} = [X_1, X_2, \\dots, X_Q]`. The second output is a list of indices of the selected
    subset of the original variables, :math:`\\mathbf{X}_S = [X_1, X_2, \\dots, X_n]`, that correspond to the minimum cost, :math:`\\mathcal{L}`.

    .. note::

        The algorithm can be very expensive (for large data sets) due to multiple computations of the normalized variance derivative.
        Try running it on multiple cores or on a sampled data set.

        In case the algorithm breaks when not being able to determine the peak
        location, try increasing the range in the ``bandwidth_values`` parameter.

    **Example:**

    .. code:: python

        from PCAfold import manifold_informed_backward_elimination
        import numpy as np

        # Generate dummy data set:
        X = np.random.rand(100,10)
        X_source = np.random.rand(100,10)

        # Define original variables to add to the optimization:
        target_variables = X[:,0:3]

        # Specify variables names
        variable_names = ['X_' + str(i) for i in range(0,10)]

        # Specify the bandwidth values to compute the optimization on:
        bandwidth_values = np.logspace(-4, 2, 50)

        # Run the subset selection algorithm:
        (ordered, selected, costs) = manifold_informed_backward_elimination(X,
                                                                            X_source,
                                                                            variable_names,
                                                                            scaling='auto',
                                                                            bandwidth_values=bandwidth_values,
                                                                            target_variables=target_variables,
                                                                            add_transformed_source=True,
                                                                            target_manifold_dimensionality=2,
                                                                            penalty_function='peak',
                                                                            norm='max',
                                                                            integrate_to_peak=True,
                                                                            verbose=True)

    :param X:
        ``numpy.ndarray`` specifying the original data set, :math:`\\mathbf{X}`. It should be of size ``(n_observations,n_variables)``.
    :param X_source:
        ``numpy.ndarray`` specifying the source terms, :math:`\\mathbf{S_X}`, corresponding to the state-space
        variables in :math:`\\mathbf{X}`. This parameter is applicable to data sets
        representing reactive flows. More information can be found in :cite:`Sutherland2009`. It should be of size ``(n_observations,n_variables)``.
    :param variable_names:
        ``list`` of ``str`` specifying variables names. Order of names in the ``variable_names`` list should match the order of variables (columns) in ``X``.
    :param scaling: (optional)
        ``str`` specifying the scaling methodology. It can be one of the following:
        ``'none'``, ``''``, ``'auto'``, ``'std'``, ``'pareto'``, ``'vast'``, ``'range'``, ``'0to1'``,
        ``'-1to1'``, ``'level'``, ``'max'``, ``'poisson'``, ``'vast_2'``, ``'vast_3'``, ``'vast_4'``.
    :param bandwidth_values:
        ``numpy.ndarray`` specifying the bandwidth values, :math:`\\sigma`, for :math:`\\hat{\\mathcal{D}}(\\sigma)` computation.
    :param target_variables: (optional)
        ``numpy.ndarray`` specifying the dependent variables that should be used in :math:`\\hat{\\mathcal{D}}(\\sigma)` computation. It should be of size ``(n_observations,n_target_variables)``.
    :param add_transformed_source: (optional)
        ``bool`` specifying if the PCA-transformed source terms of the state-space variables should be added in :math:`\\hat{\\mathcal{D}}(\\sigma)` computation, alongside the user-defined dependent variables.
    :param source_space: (optional)
        ``str`` specifying the space to which the PC source terms should be transformed before computing the cost. It can be one of the following: ``symlog``, ``continuous-symlog``, ``original-and-symlog``, ``original-and-continuous-symlog``. If set to ``None``, PC source terms are kept in their original PCA-space.
    :param target_manifold_dimensionality: (optional)
        ``int`` specifying the target dimensionality of the PCA manifold.
    :param penalty_function: (optional)
        ``str`` specifying the weighting applied to each area.
        Set ``penalty_function='peak'`` to weight each area by the rightmost peak location, :math:`\\sigma_{peak, i}`, for the :math:`i^{th}` dependent variable.
        Set ``penalty_function='sigma'`` to weight each area continuously by the bandwidth.
        Set ``penalty_function='log-sigma-over-peak'`` to weight each area continuously by the :math:`\\log_{10}` -transformed bandwidth, normalized by the right most peak location, :math:`\\sigma_{peak, i}`.
        If ``penalty_function=None``, the area is not weighted.
    :param norm: (optional)
        ``str`` specifying the norm to apply for all areas :math:`A_i`. ``norm='average'`` uses an arithmetic average, ``norm='max'`` uses the :math:`L_{\\infty}` norm,
        ``norm='median'`` uses a median area, ``norm='cumulative'`` uses a cumulative area and ``norm='min'`` uses a minimum area.
    :param integrate_to_peak: (optional)
        ``bool`` specifying whether an individual area for the :math:`i^{th}` dependent variable should be computed only up the the rightmost peak location.
    :param verbose: (optional)
        ``bool`` for printing verbose details.

    :return:
        - **ordered_variables** - ``list`` specifying the indices of the ordered variables.
        - **selected_variables** - ``list`` specifying the indices of the selected variables that correspond to the minimum cost :math:`\\mathcal{L}`.
        - **optimized_cost** - ``float`` specifying the cost corresponding to the optimized subset.
        - **costs** - ``list`` specifying the costs, :math:`\\mathcal{L}`, from each iteration.
    """

    __penalty_functions = ['peak', 'sigma', 'log-sigma-over-peak']
    __norms = ['average', 'cumulative', 'max', 'median', 'min']
    __source_spaces = ['symlog', 'continuous-symlog', 'original-and-symlog', 'original-and-continuous-symlog']

    if not isinstance(X, np.ndarray):
        raise ValueError("Parameter `X` has to be of type `numpy.ndarray`.")

    try:
        (n_observations, n_variables) = np.shape(X)
    except:
        raise ValueError("Parameter `X` has to have shape `(n_observations,n_variables)`.")

    if not isinstance(X_source, np.ndarray):
        raise ValueError("Parameter `X_source` has to be of type `numpy.ndarray`.")

    try:
        (n_observations_source, n_variables_source) = np.shape(X_source)
    except:
        raise ValueError("Parameter `X_source` has to have shape `(n_observations,n_variables)`.")

    if n_variables_source != n_variables:
        raise ValueError("Parameter `X_source` has different number of variables than `X`.")

    # TODO: In the future, we might want to allow different number of observations, there is no reason why they should be equal.
    if n_observations_source != n_observations:
        raise ValueError("Parameter `X_source` has different number of observations than `X`.")

    if not isinstance(variable_names, list):
        raise ValueError("Parameter `variable_names` has to be of type `list`.")

    if len(variable_names) != n_variables:
        raise ValueError("Parameter `variable_names` has different number of variables than `X`.")

    if not isinstance(scaling, str):
        raise ValueError("Parameter `scaling` has to be of type `str`.")

    if not isinstance(bandwidth_values, np.ndarray):
        raise ValueError("Parameter `bandwidth_values` has to be of type `numpy.ndarray`.")

    if target_variables is not None:
        if not isinstance(target_variables, np.ndarray):
            raise ValueError("Parameter `target_variables` has to be of type `numpy.ndarray`.")
        try:
            (n_d_hat_observations, n_target_variables) = np.shape(target_variables)
            target_variables_names = ['X' + str(i) for i in range(0,n_target_variables)]
        except:
            raise ValueError("Parameter `target_variables` has to have shape `(n_observations,n_target_variables)`.")

        if n_d_hat_observations != n_observations_source:
            raise ValueError("Parameter `target_variables` has different number of observations than `X_source`.")

    if not isinstance(add_transformed_source, bool):
        raise ValueError("Parameter `add_transformed_source` has to be of type `bool`.")

    if target_variables is None:
        if not add_transformed_source:
            raise ValueError("Either `target_variables` has to be specified or `add_transformed_source` has to be set to True.")

    if source_space is not None:
        if not isinstance(source_space, str):
            raise ValueError("Parameter `source_space` has to be of type `str`.")
        if source_space.lower() not in __source_spaces:
            raise ValueError("Parameter `source_space` has to be one of the following: symlog`, `continuous-symlog`.")

    if not isinstance(target_manifold_dimensionality, int):
        raise ValueError("Parameter `target_manifold_dimensionality` has to be of type `int`.")

    if penalty_function is not None:
        if not isinstance(penalty_function, str):
            raise ValueError("Parameter `penalty_function` has to be of type `str`.")
        if penalty_function not in __penalty_functions:
            raise ValueError("Parameter `penalty_function` has to be one of the following: 'peak', 'sigma', 'log-sigma-over-peak'.")

    if not isinstance(norm, str):
        raise ValueError("Parameter `norm` has to be of type `str`.")

    if norm not in __norms:
        raise ValueError("Parameter `norm` has to be one of the following: 'average', 'cumulative', 'max', 'median', 'min'.")

    if not isinstance(integrate_to_peak, bool):
        raise ValueError("Parameter `integrate_to_peak` has to be of type `bool`.")

    if not isinstance(verbose, bool):
        raise ValueError("Parameter `verbose` has to be of type `bool`.")

    costs = []

    if verbose: print('Optimizing...\n')

    if verbose:
        if add_transformed_source is not None:
            if source_space is not None:
                print('PC source terms will be assessed in the ' + source_space + ' space.\n')

    total_tic = time.perf_counter()

    remaining_variables_list = [i for i in range(0,n_variables)]

    ordered_variables = []

    loop_counter = 0

    # Iterate the algorithm: ---------------------------------------------------
    while len(remaining_variables_list) > target_manifold_dimensionality:

        iteration_tic = time.perf_counter()

        loop_counter += 1

        if verbose:
            print('Iteration No.' + str(loop_counter))
            print('Currently eliminating variable from the following list: ')
            print([variable_names[i] for i in remaining_variables_list])

        current_cost_function = []

        for i_variable in remaining_variables_list:

            if verbose: print('\tCurrently eliminated variable: ' + variable_names[i_variable])

            # Consider a subset with all variables but the currently eliminated one:
            current_variables_list = [i for i in remaining_variables_list if i != i_variable]

            if verbose:
                print('\tRunning PCA for a subset:')
                print('\t' + ', '.join([variable_names[i] for i in current_variables_list]))

            pca = reduction.PCA(X[:,current_variables_list], scaling=scaling, n_components=target_manifold_dimensionality)
            PCs = pca.transform(X[:,current_variables_list])
            (PCs, _, _) = preprocess.center_scale(PCs, '-1to1')

            if add_transformed_source:
                PC_sources = pca.transform(X_source[:,current_variables_list], nocenter=True)
                if source_space is not None:
                    if source_space == 'original-and-symlog':
                        transformed_PC_sources = preprocess.log_transform(PC_sources, method='symlog', threshold=1.e-4)
                    elif source_space == 'original-and-continuous-symlog':
                        transformed_PC_sources = preprocess.log_transform(PC_sources, method='continuous-symlog', threshold=1.e-4)
                    else:
                        transformed_PC_sources = preprocess.log_transform(PC_sources, method=source_space, threshold=1.e-4)

            if target_variables is None:
                if source_space == 'original-and-symlog' or source_space == 'original-and-continuous-symlog':
                    depvars = np.hstack((PC_sources, transformed_PC_sources))
                    depvar_names = ['SZ' + str(i) for i in range(0,target_manifold_dimensionality)] + ['symlog-SZ' + str(i) for i in range(0,target_manifold_dimensionality)]
                elif source_space == 'symlog' or source_space == 'continuous-symlog':
                    depvars = cp.deepcopy(transformed_PC_sources)
                    depvar_names = ['symlog-SZ' + str(i) for i in range(0,target_manifold_dimensionality)]
                else:
                    depvars = cp.deepcopy(PC_sources)
                    depvar_names = ['SZ' + str(i) for i in range(0,target_manifold_dimensionality)]
            else:
                if add_transformed_source:
                    if source_space == 'original-and-symlog' or source_space == 'original-and-continuous-symlog':
                        depvars = np.hstack((PC_sources, transformed_PC_sources, target_variables))
                        depvar_names = ['SZ' + str(i) for i in range(0,target_manifold_dimensionality)] + ['symlog-SZ' + str(i) for i in range(0,target_manifold_dimensionality)] + target_variables_names
                    elif source_space == 'symlog' or source_space == 'continuous-symlog':
                        depvars = np.hstack((transformed_PC_sources, target_variables))
                        depvar_names = ['symlog-SZ' + str(i) for i in range(0,target_manifold_dimensionality)] + target_variables_names
                    else:
                        depvars = np.hstack((PC_sources, target_variables))
                        depvar_names = ['SZ' + str(i) for i in range(0,target_manifold_dimensionality)] + target_variables_names
                else:
                    depvars = cp.deepcopy(target_variables)
                    depvar_names = cp.deepcopy(target_variables_names)

            current_variance_data = compute_normalized_variance(PCs, depvars, depvar_names=depvar_names, scale_unit_box = False, bandwidth_values=bandwidth_values)
            current_area = cost_function_normalized_variance_derivative(current_variance_data, penalty_function=penalty_function, norm=norm, integrate_to_peak=integrate_to_peak)
            if verbose: print('\tCost:\t%.4f' % current_area)
            current_cost_function.append(current_area)

            # Starting from the second iteration, we can make a comparison with the previous iteration's results:
            if loop_counter > 1:
                if current_area <= previous_area:
                    if verbose: print(colored('\tSAME OR BETTER', 'green'))
                else:
                    if verbose: print(colored('\tWORSE', 'red'))

        min_area = np.min(current_cost_function)
        costs.append(min_area)

        # Search for the variable whose removal will decrease the cost the most:
        (worst_variable_index, ) = np.where(np.array(current_cost_function)==min_area)

        # This handles cases where there are multiple minima with the same minimum cost value:
        try:
            worst_variable_index = int(worst_variable_index)
        except:
            worst_variable_index = int(worst_variable_index[0])

        if verbose: print('\n\tVariable ' + variable_names[remaining_variables_list[worst_variable_index]] + ' is removed.\n\tCost:\t%.4f' % min_area + '\n')

        # Append removed variable in the ascending order, this list is later flipped to have variables ordered from most to least important:
        ordered_variables.append(remaining_variables_list[worst_variable_index])

        # Create a new list of variables to loop over at the next iteration:
        remaining_variables_list = [i for i in range(0,n_variables) if i not in ordered_variables]

        if loop_counter > 1:
            if min_area <= previous_area:
                previous_area = min_area
        else:
            previous_area = min_area

        iteration_toc = time.perf_counter()
        if verbose: print(f'\tIteration time: {(iteration_toc - iteration_tic)/60:0.1f} minutes.' + '\n' + '-'*50)

    # Compute the optimal subset where the overal cost from all iterations is minimized: ------------------

    # One last time remove the worst variable:
    del current_cost_function[worst_variable_index]

    for i in remaining_variables_list:
        ordered_variables.append(i)

    for i in range(0,len(remaining_variables_list)):
        costs.append(current_cost_function[i])

    # Invert lists to have variables ordered from most to least important:
    ordered_variables = ordered_variables[::-1]
    costs = costs[::-1]

    (min_cost_function_index, ) = np.where(costs==np.min(costs))

    # This handles cases where there are multiple minima with the same minimum cost value:
    try:
        min_cost_function_index = int(min_cost_function_index)
    except:
        min_cost_function_index = int(min_cost_function_index[0])

    selected_variables = list(np.array(ordered_variables)[0:min_cost_function_index])

    optimized_cost = costs[min_cost_function_index]

    if verbose:

        print('Ordered variables:')
        print(', '.join([variable_names[i] for i in ordered_variables]))
        print(ordered_variables)
        print('Final cost: %.4f' % min_area)

        print('\nSelected variables:')
        print(', '.join([variable_names[i] for i in selected_variables]))
        print(selected_variables)
        print('Lowest cost: %.4f' % optimized_cost)

    total_toc = time.perf_counter()
    if verbose: print(f'\nOptimization time: {(total_toc - total_tic)/60:0.1f} minutes.' + '\n' + '-'*50)

    return ordered_variables, selected_variables, optimized_cost, costs

################################################################################
#
# Regression assessment
#
################################################################################

class RegressionAssessment:
    """
    Wrapper class for storing all regression assessment metrics for a given
    regression solution given by the observed dependent variables, :math:`\\pmb{\\phi}_o`,
    and the predicted dependent variables, :math:`\\pmb{\\phi}_p`.

    **Example:**

    .. code:: python

        from PCAfold import PCA, RegressionAssessment
        import numpy as np

        # Generate dummy data set:
        X = np.random.rand(100,3)

        # Instantiate PCA class object:
        pca_X = PCA(X, scaling='auto', n_components=2)

        # Approximate the data set:
        X_rec = pca_X.reconstruct(pca_X.transform(X))

        # Instantiate RegressionAssessment class object:
        regression_metrics = RegressionAssessment(X, X_rec)

        # Access mean absolute error values:
        MAE = regression_metrics.mean_absolute_error

    In addition, all stratified regression metrics can be computed on a single variable:

    .. code:: python

        from PCAfold import variable_bins

        # Generate bins:
        (idx, bins_borders) = variable_bins(X[:,0], k=5, verbose=False)

        # Instantiate RegressionAssessment class object:
        stratified_regression_metrics = RegressionAssessment(X[:,0], X_rec[:,0], idx=idx)

        # Access stratified mean absolute error values:
        stratified_MAE = stratified_regression_metrics.stratified_mean_absolute_error

    :param observed:
        ``numpy.ndarray`` specifying the observed values of dependent variables, :math:`\\pmb{\\phi}_o`. It should be of size ``(n_observations,)`` or ``(n_observations,n_variables)``.
    :param predicted:
        ``numpy.ndarray`` specifying the predicted values of dependent variables, :math:`\\pmb{\\phi}_p`. It should be of size ``(n_observations,)`` or ``(n_observations,n_variables)``.
    :param idx:
        ``numpy.ndarray`` of cluster classifications. It should be of size ``(n_observations,)`` or ``(n_observations,1)``.
    :param variable_names: (optional)
        ``list`` of ``str`` specifying variable names.
    :param use_global_mean: (optional)
        ``bool`` specifying if global mean of the observed variable should be used as a reference in :math:`R^2` calculation.
    :param norm:
        ``str`` specifying the normalization, :math:`d_{norm}`, for NRMSE computation. It can be one of the following: ``std``, ``range``, ``root_square_mean``, ``root_square_range``, ``root_square_std``, ``abs_mean``.
    :param use_global_norm: (optional)
        ``bool`` specifying if global norm of the observed variable should be used in NRMSE calculation.
    :param tolerance:
        ``float`` specifying the tolerance for GDE computation.

    **Attributes:**

    - **coefficient_of_determination** - (read only) ``numpy.ndarray`` specifying the coefficient of determination, :math:`R^2`, values. It has size ``(1,n_variables)``.
    - **mean_absolute_error** - (read only) ``numpy.ndarray`` specifying the mean absolute error (MAE) values. It has size ``(1,n_variables)``.
    - **mean_squared_error** - (read only) ``numpy.ndarray`` specifying the mean squared error (MSE) values. It has size ``(1,n_variables)``.
    - **root_mean_squared_error** - (read only) ``numpy.ndarray`` specifying the root mean squared error (RMSE) values. It has size ``(1,n_variables)``.
    - **normalized_root_mean_squared_error** - (read only) ``numpy.ndarray`` specifying the normalized root mean squared error (NRMSE) values. It has size ``(1,n_variables)``.
    - **good_direction_estimate** - (read only) ``float`` specifying the good direction estimate (GDE) value, treating the entire :math:`\\pmb{\\phi}_o` and :math:`\\pmb{\\phi}_p` as vectors. Note that if a single dependent variable is passed, GDE cannot be computed and is set to ``NaN``.

    If ``idx`` has been specified:

    - **stratified_coefficient_of_determination** - (read only) ``numpy.ndarray`` specifying the coefficient of determination, :math:`R^2`, values. It has size ``(1,n_variables)``.
    - **stratified_mean_absolute_error** - (read only) ``numpy.ndarray`` specifying the mean absolute error (MAE) values. It has size ``(1,n_variables)``.
    - **stratified_mean_squared_error** - (read only) ``numpy.ndarray`` specifying the mean squared error (MSE) values. It has size ``(1,n_variables)``.
    - **stratified_root_mean_squared_error** - (read only) ``numpy.ndarray`` specifying the root mean squared error (RMSE) values. It has size ``(1,n_variables)``.
    - **stratified_normalized_root_mean_squared_error** - (read only) ``numpy.ndarray`` specifying the normalized root mean squared error (NRMSE) values. It has size ``(1,n_variables)``.
    """

    def __init__(self, observed, predicted, idx=None, variable_names=None, use_global_mean=False, norm='std', use_global_norm=False, tolerance=0.05):

        if not isinstance(observed, np.ndarray):
            raise ValueError("Parameter `observed` has to be of type `numpy.ndarray`.")

        try:
            (n_observed,) = np.shape(observed)
            n_var_observed = 1
            observed = observed[:,None]
        except:
            (n_observed, n_var_observed) = np.shape(observed)

        if not isinstance(predicted, np.ndarray):
            raise ValueError("Parameter `predicted` has to be of type `numpy.ndarray`.")

        try:
            (n_predicted,) = np.shape(predicted)
            n_var_predicted = 1
            predicted = predicted[:,None]
        except:
            (n_predicted, n_var_predicted) = np.shape(predicted)

        if n_observed != n_predicted:
            raise ValueError("Parameter `observed` has different number of elements than `predicted`.")

        if n_var_observed != n_var_predicted:
            raise ValueError("Parameter `observed` has different number of elements than `predicted`.")

        self.__n_variables = n_var_observed

        if idx is not None:

            if isinstance(idx, np.ndarray):
                if not all(isinstance(i, np.integer) for i in idx.ravel()):
                    raise ValueError("Parameter `idx` can only contain integers.")
            else:
                raise ValueError("Parameter `idx` has to be of type `numpy.ndarray`.")

            try:
                (n_observations_idx, ) = np.shape(idx)
                n_idx = 1
            except:
                (n_observations_idx, n_idx) = np.shape(idx)

            if n_idx != 1:
                raise ValueError("Parameter `idx` has to have size `(n_observations,)` or `(n_observations,1)`.")

            if n_observations_idx != n_observed:
                raise ValueError('Vector of cluster classifications `idx` has different number of observations than the original data set `X`.')

            if n_var_observed != 1:
                raise ValueError('Stratified regression metrics can only be computed on a single vector.')

            self.__n_clusters = len(np.unique(idx))
            self.__cluster_populations = preprocess.get_populations(idx)

            self.__cluster_min = []
            self.__cluster_max = []
            for i in range(0,self.__n_clusters):
                (cluster_indices, ) = np.where(idx==i)
                self.__cluster_min.append(np.min(observed[cluster_indices,:]))
                self.__cluster_max.append(np.max(observed[cluster_indices,:]))

        if not isinstance(use_global_mean, bool):
            raise ValueError("Parameter `use_global_mean` has to be a boolean.")

        if variable_names is not None:
            if not isinstance(variable_names, list):
                raise ValueError("Parameter `variable_names` has to be of type `list`.")
            else:
                if self.__n_variables != len(variable_names):
                    raise ValueError("Parameter `variable_names` has different number of variables than `observed` and `predicted`.")
        else:
            variable_names = []
            for i in range(0,self.__n_variables):
                variable_names.append('X' + str(i+1))

        self.__variable_names = variable_names

        self.__coefficient_of_determination_matrix = np.ones((1,self.__n_variables))
        self.__mean_absolute_error_matrix = np.ones((1,self.__n_variables))
        self.__mean_squared_error_matrix = np.ones((1,self.__n_variables))
        self.__root_mean_squared_error_matrix = np.ones((1,self.__n_variables))
        self.__normalized_root_mean_squared_error_matrix = np.ones((1,self.__n_variables))

        if n_var_observed > 1:
            _, self.__good_direction_estimate_value = good_direction_estimate(observed, predicted, tolerance=tolerance)
            self.__good_direction_estimate_matrix = self.__good_direction_estimate_value * np.ones((1,self.__n_variables))
        else:
            self.__good_direction_estimate_value = np.NAN
            self.__good_direction_estimate_matrix = self.__good_direction_estimate_value * np.ones((1,self.__n_variables))

        for i in range(0,self.__n_variables):

            self.__coefficient_of_determination_matrix[0,i] = coefficient_of_determination(observed[:,i], predicted[:,i])
            self.__mean_absolute_error_matrix[0,i] = mean_absolute_error(observed[:,i], predicted[:,i])
            self.__mean_squared_error_matrix[0,i] = mean_squared_error(observed[:,i], predicted[:,i])
            self.__root_mean_squared_error_matrix[0,i] = root_mean_squared_error(observed[:,i], predicted[:,i])
            self.__normalized_root_mean_squared_error_matrix[0,i] = normalized_root_mean_squared_error(observed[:,i], predicted[:,i], norm=norm)

        if idx is not None:

            self.__stratified_coefficient_of_determination = stratified_coefficient_of_determination(observed, predicted, idx=idx, use_global_mean=use_global_mean)
            self.__stratified_mean_absolute_error = stratified_mean_absolute_error(observed, predicted, idx=idx)
            self.__stratified_mean_squared_error = stratified_mean_squared_error(observed, predicted, idx=idx)
            self.__stratified_root_mean_squared_error = stratified_root_mean_squared_error(observed, predicted, idx=idx)
            self.__stratified_normalized_root_mean_squared_error = stratified_normalized_root_mean_squared_error(observed, predicted, idx=idx, norm=norm, use_global_norm=use_global_norm)

        else:

            self.__stratified_coefficient_of_determination = None
            self.__stratified_mean_absolute_error = None
            self.__stratified_mean_squared_error = None
            self.__stratified_root_mean_squared_error = None
            self.__stratified_normalized_root_mean_squared_error = None

    @property
    def coefficient_of_determination(self):
        return self.__coefficient_of_determination_matrix

    @property
    def mean_absolute_error(self):
        return self.__mean_absolute_error_matrix

    @property
    def mean_squared_error(self):
        return self.__mean_squared_error_matrix

    @property
    def root_mean_squared_error(self):
        return self.__root_mean_squared_error_matrix

    @property
    def normalized_root_mean_squared_error(self):
        return self.__normalized_root_mean_squared_error_matrix

    @property
    def good_direction_estimate(self):
        return self.__good_direction_estimate_value

    @property
    def stratified_coefficient_of_determination(self):
        return self.__stratified_coefficient_of_determination

    @property
    def stratified_mean_absolute_error(self):
        return self.__stratified_mean_absolute_error

    @property
    def stratified_mean_squared_error(self):
        return self.__stratified_mean_squared_error

    @property
    def stratified_root_mean_squared_error(self):
        return self.__stratified_root_mean_squared_error

    @property
    def stratified_normalized_root_mean_squared_error(self):
        return self.__stratified_normalized_root_mean_squared_error

# ------------------------------------------------------------------------------

    def print_metrics(self, table_format=['raw'], float_format='.4f', metrics=None, comparison=None):
        """
        Prints regression assessment metrics as raw text, in ``tex`` format and/or as ``pandas.DataFrame``.

        **Example:**

        .. code:: python

            from PCAfold import PCA, RegressionAssessment
            import numpy as np

            # Generate dummy data set:
            X = np.random.rand(100,3)

            # Instantiate PCA class object:
            pca_X = PCA(X, scaling='auto', n_components=2)

            # Approximate the data set:
            X_rec = pca_X.reconstruct(pca_X.transform(X))

            # Instantiate RegressionAssessment class object:
            regression_metrics = RegressionAssessment(X, X_rec)

            # Print regression metrics:
            regression_metrics.print_metrics(table_format=['raw', 'tex', 'pandas'],
                                             float_format='.4f',
                                             metrics=['R2', 'NRMSE', 'GDE'])

        .. note::

            Adding ``'raw'`` to the ``table_format`` list will result in printing:

            .. code-block:: text

                -------------------------
                X1
                R2:	0.9900
                NRMSE:	0.0999
                GDE:	70.0000
                -------------------------
                X2
                R2:	0.6126
                NRMSE:	0.6224
                GDE:	70.0000
                -------------------------
                X3
                R2:	0.6368
                NRMSE:	0.6026
                GDE:	70.0000

            Adding ``'tex'`` to the ``table_format`` list will result in printing:

            .. code-block:: text

                \\begin{table}[h!]
                \\begin{center}
                \\begin{tabular}{llll} \\toprule
                 & \\textit{X1} & \\textit{X2} & \\textit{X3} \\\\ \\midrule
                R2 & 0.9900 & 0.6126 & 0.6368 \\\\
                NRMSE & 0.0999 & 0.6224 & 0.6026 \\\\
                GDE & 70.0000 & 70.0000 & 70.0000 \\\\
                \\end{tabular}
                \\caption{}\\label{}
                \\end{center}
                \\end{table}

            Adding ``'pandas'`` to the ``table_format`` list (works well in Jupyter notebooks) will result in printing:

            .. image:: ../images/generate-pandas-table.png
                :width: 300
                :align: center

        Additionally, the current object of ``RegressionAssessment`` class can be compared with another object:

        .. code:: python

            from PCAfold import PCA, RegressionAssessment
            import numpy as np

            # Generate dummy data set:
            X = np.random.rand(100,3)
            Y = np.random.rand(100,3)

            # Instantiate PCA class object:
            pca_X = PCA(X, scaling='auto', n_components=2)
            pca_Y = PCA(Y, scaling='auto', n_components=2)

            # Approximate the data set:
            X_rec = pca_X.reconstruct(pca_X.transform(X))
            Y_rec = pca_Y.reconstruct(pca_Y.transform(Y))

            # Instantiate RegressionAssessment class object:
            regression_metrics_X = RegressionAssessment(X, X_rec)
            regression_metrics_Y = RegressionAssessment(Y, Y_rec)

            # Print regression metrics:
            regression_metrics_X.print_metrics(table_format=['raw', 'pandas'],
                                               float_format='.4f',
                                               metrics=['R2', 'NRMSE', 'GDE'],
                                               comparison=regression_metrics_Y)

        .. note::

            Adding ``'raw'`` to the ``table_format`` list will result in printing:

            .. code-block:: text

                -------------------------
                X1
                R2:	0.9133	BETTER
                NRMSE:	0.2944	BETTER
                GDE:	67.0000	WORSE
                -------------------------
                X2
                R2:	0.5969	WORSE
                NRMSE:	0.6349	WORSE
                GDE:	67.0000	WORSE
                -------------------------
                X3
                R2:	0.6175	WORSE
                NRMSE:	0.6185	WORSE
                GDE:	67.0000	WORSE

            Adding ``'pandas'`` to the ``table_format`` list (works well in Jupyter notebooks) will result in printing:

            .. image:: ../images/generate-pandas-table-comparison.png
                :width: 300
                :align: center

        :param table_format: (optional)
            ``list`` of ``str`` specifying the format(s) in which the table should be printed.
            Strings can only be ``'raw'``, ``'tex'`` and/or ``'pandas'``.
        :param float_format: (optional)
            ``str`` specifying the display format for the numerical entries inside the
            table. By default it is set to ``'.4f'``.
        :param metrics: (optional)
            ``list`` of ``str`` specifying which metrics should be printed. Strings can only be ``'R2'``, ``'MAE'``, ``'MSE'``, ``'RMSE'``, ``'NRMSE'``, ``'GDE'``.
            If metrics is set to ``None``, all available metrics will be printed.
        :param comparison: (optional)
            object of ``RegressionAssessment`` class specifying the metrics that should be compared with the current regression metrics.
        """

        __table_formats = ['raw', 'tex', 'pandas']
        __metrics_names = ['R2', 'MAE', 'MSE', 'RMSE', 'NRMSE', 'GDE']
        __metrics_dict = {'R2': self.__coefficient_of_determination_matrix,
                          'MAE': self.__mean_absolute_error_matrix,
                          'MSE': self.__mean_squared_error_matrix,
                          'RMSE': self.__root_mean_squared_error_matrix,
                          'NRMSE': self.__normalized_root_mean_squared_error_matrix,
                          'GDE': self.__good_direction_estimate_matrix}
        if comparison is not None:
            __comparison_metrics_dict = {'R2': comparison.coefficient_of_determination,
                                         'MAE': comparison.mean_absolute_error,
                                         'MSE': comparison.mean_squared_error,
                                         'RMSE': comparison.root_mean_squared_error,
                                         'NRMSE': comparison.normalized_root_mean_squared_error,
                                         'GDE': comparison.good_direction_estimate * np.ones_like(comparison.coefficient_of_determination)}

        if not isinstance(table_format, list):
            raise ValueError("Parameter `table_format` has to be of type `list`.")

        for item in table_format:
            if item not in __table_formats:
                raise ValueError("Parameter `table_format` can only contain 'raw', 'tex' and/or 'pandas'.")

        if not isinstance(float_format, str):
            raise ValueError("Parameter `float_format` has to be of type `str`.")

        if metrics is not None:
            if not isinstance(metrics, list):
                raise ValueError("Parameter `metrics` has to be of type `list`.")

            for item in metrics:
                if item not in __metrics_names:
                    raise ValueError("Parameter `metrics` can only be: 'R2', 'MAE', 'MSE', 'RMSE', 'NRMSE', 'GDE'.")
        else:
            metrics = __metrics_names

        if comparison is None:

            for item in set(table_format):

                if item=='raw':

                    for i in range(0,self.__n_variables):

                        print('-'*25 + '\n' + self.__variable_names[i])

                        metrics_to_print = []
                        for metric in metrics:
                            metrics_to_print.append(__metrics_dict[metric][0,i])

                        for j in range(0,len(metrics)):
                            print(metrics[j] + ':\t' + ('%' + float_format) % metrics_to_print[j])

                if item=='tex':

                    import pandas as pd

                    metrics_to_print = np.zeros_like(self.__coefficient_of_determination_matrix)
                    for metric in metrics:
                        metrics_to_print = np.vstack((metrics_to_print, __metrics_dict[metric]))
                    metrics_to_print = metrics_to_print[1::,:]

                    metrics_table = pd.DataFrame(metrics_to_print, columns=self.__variable_names, index=metrics)
                    generate_tex_table(metrics_table, float_format=float_format)

                if item=='pandas':

                    import pandas as pd
                    from IPython.display import display
                    pandas_format = '{:,' + float_format + '}'

                    metrics_to_print = np.zeros_like(self.__coefficient_of_determination_matrix.T)
                    for metric in metrics:
                        metrics_to_print = np.hstack((metrics_to_print, __metrics_dict[metric].T))
                    metrics_to_print = metrics_to_print[:,1::]

                    metrics_table = pd.DataFrame(metrics_to_print, columns=metrics, index=self.__variable_names)
                    formatted_table = metrics_table.style.format(pandas_format)
                    display(formatted_table)

        else:

            for item in set(table_format):

                if item=='raw':

                    for i in range(0,self.__n_variables):

                        print('-'*25 + '\n' + self.__variable_names[i])

                        metrics_to_print = []
                        comparison_metrics_to_print = []
                        for metric in metrics:
                            metrics_to_print.append(__metrics_dict[metric][0,i])
                            comparison_metrics_to_print.append(__comparison_metrics_dict[metric][0,i])

                        for j, metric in enumerate(metrics):

                            if metric == 'R2' or metric == 'GDE':
                                if metrics_to_print[j] > comparison_metrics_to_print[j]:
                                    print(metrics[j] + ':\t' + ('%' + float_format) % metrics_to_print[j] + colored('\tBETTER', 'green'))
                                elif metrics_to_print[j] < comparison_metrics_to_print[j]:
                                    print(metrics[j] + ':\t' + ('%' + float_format) % metrics_to_print[j] + colored('\tWORSE', 'red'))
                                elif metrics_to_print[j] == comparison_metrics_to_print[j]:
                                    print(metrics[j] + ':\t' + ('%' + float_format) % metrics_to_print[j] + '\tSAME')
                            else:
                                if metrics_to_print[j] > comparison_metrics_to_print[j]:
                                    print(metrics[j] + ':\t' + ('%' + float_format) % metrics_to_print[j] + colored('\tWORSE', 'red'))
                                elif metrics_to_print[j] < comparison_metrics_to_print[j]:
                                    print(metrics[j] + ':\t' + ('%' + float_format) % metrics_to_print[j] + colored('\tBETTER', 'green'))
                                elif metrics_to_print[j] == comparison_metrics_to_print[j]:
                                    print(metrics[j] + ':\t' + ('%' + float_format) % metrics_to_print[j] + '\tSAME')

                if item=='pandas':

                    import pandas as pd
                    from IPython.display import display
                    pandas_format = '{:,' + float_format + '}'

                    metrics_to_print = np.zeros_like(self.__coefficient_of_determination_matrix.T)
                    comparison_metrics_to_print = np.zeros_like(comparison.coefficient_of_determination.T)
                    for metric in metrics:
                        metrics_to_print = np.hstack((metrics_to_print, __metrics_dict[metric].T))
                        comparison_metrics_to_print = np.hstack((comparison_metrics_to_print, __comparison_metrics_dict[metric].T))
                    metrics_to_print = metrics_to_print[:,1::]
                    comparison_metrics_to_print = comparison_metrics_to_print[:,1::]

                    def highlight_better(data, data_comparison, color='lightgreen'):

                        attr = 'background-color: {}'.format(color)

                        is_better = False * data

                        # Lower value is better (MAE, MSE, RMSE, NRMSE):
                        try:
                            is_better['MAE'] = data['MAE'].astype(float) < data_comparison['MAE']
                        except:
                            pass
                        try:
                            is_better['MSE'] = data['MSE'].astype(float) < data_comparison['MSE']
                        except:
                            pass
                        try:
                            is_better['RMSE'] = data['RMSE'].astype(float) < data_comparison['RMSE']
                        except:
                            pass
                        try:
                            is_better['NRMSE'] = data['NRMSE'].astype(float) < data_comparison['NRMSE']
                        except:
                            pass

                        # Higher value is better (R2 and GDE):
                        try:
                            is_better['R2'] = data['R2'].astype(float) > data_comparison['R2']
                        except:
                            pass
                        try:
                            is_better['GDE'] = data['GDE'].astype(float) > data_comparison['GDE']
                        except:
                            pass

                        formatting = [attr if v else '' for v in is_better]

                        formatting = pd.DataFrame(np.where(is_better, attr, ''), index=data.index, columns=data.columns)

                        return formatting

                    def highlight_worse(data, data_comparison, color='salmon'):

                        attr = 'background-color: {}'.format(color)

                        is_worse = False * data

                        # Higher value is worse (MAE, MSE, RMSE, NRMSE):
                        try:
                            is_worse['MAE'] = data['MAE'].astype(float) > data_comparison['MAE']
                        except:
                            pass
                        try:
                            is_worse['MSE'] = data['MSE'].astype(float) > data_comparison['MSE']
                        except:
                            pass
                        try:
                            is_worse['RMSE'] = data['RMSE'].astype(float) > data_comparison['RMSE']
                        except:
                            pass
                        try:
                            is_worse['NRMSE'] = data['NRMSE'].astype(float) > data_comparison['NRMSE']
                        except:
                            pass

                        # Lower value is worse (R2 and GDE):
                        try:
                            is_worse['R2'] = data['R2'].astype(float) < data_comparison['R2']
                        except:
                            pass
                        try:
                            is_worse['GDE'] = data['GDE'].astype(float) < data_comparison['GDE']
                        except:
                            pass

                        formatting = [attr if v else '' for v in is_worse]

                        formatting = pd.DataFrame(np.where(is_worse, attr, ''), index=data.index, columns=data.columns)

                        return formatting

                    metrics_table = pd.DataFrame(metrics_to_print, columns=metrics, index=self.__variable_names)
                    comparison_metrics_table = pd.DataFrame(comparison_metrics_to_print, columns=metrics, index=self.__variable_names)

                    formatted_table = metrics_table.style.apply(highlight_better, data_comparison=comparison_metrics_table, axis=None)\
                                                         .apply(highlight_worse, data_comparison=comparison_metrics_table, axis=None)\
                                                         .format(pandas_format)

                    display(formatted_table)

# ------------------------------------------------------------------------------

    def print_stratified_metrics(self, table_format=['raw'], float_format='.4f', metrics=None, comparison=None):
        """
        Prints stratified regression assessment metrics as raw text, in ``tex`` format and/or as ``pandas.DataFrame``.
        In each cluster, in addition to the regression metrics, number of observations is printed,
        along with the minimum and maximum values of the observed variable in that cluster.

        **Example:**

        .. code:: python

            from PCAfold import PCA, variable_bins, RegressionAssessment
            import numpy as np

            # Generate dummy data set:
            X = np.random.rand(100,3)

            # Instantiate PCA class object:
            pca_X = PCA(X, scaling='auto', n_components=2)

            # Approximate the data set:
            X_rec = pca_X.reconstruct(pca_X.transform(X))

            # Generate bins:
            (idx, bins_borders) = variable_bins(X[:,0], k=3, verbose=False)

            # Instantiate RegressionAssessment class object:
            stratified_regression_metrics = RegressionAssessment(X[:,0], X_rec[:,0], idx=idx)

            # Print regression metrics:
            stratified_regression_metrics.print_stratified_metrics(table_format=['raw', 'tex', 'pandas'],
                                                                   float_format='.4f',
                                                                   metrics=['R2', 'MAE', 'NRMSE'])

        .. note::

            Adding ``'raw'`` to the ``table_format`` list will result in printing:

            .. code-block:: text

                -------------------------
                k1
                Observations:	31
                Min:	0.0120
                Max:	0.3311
                R2:	-3.3271
                MAE:	0.1774
                NRMSE:	2.0802
                -------------------------
                k2
                Observations:	38
                Min:	0.3425
                Max:	0.6665
                R2:	-1.4608
                MAE:	0.1367
                NRMSE:	1.5687
                -------------------------
                k3
                Observations:	31
                Min:	0.6853
                Max:	0.9959
                R2:	-3.7319
                MAE:	0.1743
                NRMSE:	2.1753

            Adding ``'tex'`` to the ``table_format`` list will result in printing:

            .. code-block:: text

                \\begin{table}[h!]
                \\begin{center}
                \\begin{tabular}{llll} \\toprule
                 & \\textit{k1} & \\textit{k2} & \\textit{k3} \\\\ \\midrule
                Observations & 31.0000 & 38.0000 & 31.0000 \\\\
                Min & 0.0120 & 0.3425 & 0.6853 \\\\
                Max & 0.3311 & 0.6665 & 0.9959 \\\\
                R2 & -3.3271 & -1.4608 & -3.7319 \\\\
                MAE & 0.1774 & 0.1367 & 0.1743 \\\\
                NRMSE & 2.0802 & 1.5687 & 2.1753 \\\\
                \\end{tabular}
                \\caption{}\\label{}
                \\end{center}
                \\end{table}

            Adding ``'pandas'`` to the ``table_format`` list (works well in Jupyter notebooks) will result in printing:

            .. image:: ../images/generate-pandas-table-stratified.png
                :width: 500
                :align: center

        Additionally, the current object of ``RegressionAssessment`` class can be compared with another object:

        .. code:: python

            from PCAfold import PCA, variable_bins, RegressionAssessment
            import numpy as np

            # Generate dummy data set:
            X = np.random.rand(100,3)

            # Instantiate PCA class object:
            pca_X = PCA(X, scaling='auto', n_components=2)

            # Approximate the data set:
            X_rec = pca_X.reconstruct(pca_X.transform(X))

            # Generate bins:
            (idx, bins_borders) = variable_bins(X[:,0], k=3, verbose=False)

            # Instantiate RegressionAssessment class object:
            stratified_regression_metrics_0 = RegressionAssessment(X[:,0], X_rec[:,0], idx=idx)
            stratified_regression_metrics_1 = RegressionAssessment(X[:,1], X_rec[:,1], idx=idx)

            # Print regression metrics:
            stratified_regression_metrics_0.print_stratified_metrics(table_format=['raw', 'pandas'],
                                                                     float_format='.4f',
                                                                     metrics=['R2', 'MAE', 'NRMSE'],
                                                                     comparison=stratified_regression_metrics_1)

        .. note::

            Adding ``'raw'`` to the ``table_format`` list will result in printing:

            .. code-block:: text

                -------------------------
                k1
                Observations:	39
                Min:	0.0013
                Max:	0.3097
                R2:	0.9236	BETTER
                MAE:	0.0185	BETTER
                NRMSE:	0.2764	BETTER
                -------------------------
                k2
                Observations:	29
                Min:	0.3519
                Max:	0.6630
                R2:	0.9380	BETTER
                MAE:	0.0179	BETTER
                NRMSE:	0.2491	BETTER
                -------------------------
                k3
                Observations:	32
                Min:	0.6663
                Max:	0.9943
                R2:	0.9343	BETTER
                MAE:	0.0194	BETTER
                NRMSE:	0.2563	BETTER

            Adding ``'pandas'`` to the ``table_format`` list (works well in Jupyter notebooks) will result in printing:

            .. image:: ../images/generate-pandas-table-comparison-stratified.png
                :width: 500
                :align: center

        :param table_format: (optional)
            ``list`` of ``str`` specifying the format(s) in which the table should be printed.
            Strings can only be ``'raw'``, ``'tex'`` and/or ``'pandas'``.
        :param float_format: (optional)
            ``str`` specifying the display format for the numerical entries inside the
            table. By default it is set to ``'.4f'``.
        :param metrics: (optional)
            ``list`` of ``str`` specifying which metrics should be printed. Strings can only be ``'R2'``, ``'MAE'``, ``'MSE'``, ``'RMSE'``, ``'NRMSE'``.
            If metrics is set to ``None``, all available metrics will be printed.
        :param comparison: (optional)
            object of ``RegressionAssessment`` class specifying the metrics that should be compared with the current regression metrics.
        """

        __table_formats = ['raw', 'tex', 'pandas']
        __metrics_names = ['R2', 'MAE', 'MSE', 'RMSE', 'NRMSE']
        __clusters_names = ['k' + str(i) for i in range(1,self.__n_clusters+1)]
        __metrics_dict = {'R2': self.__stratified_coefficient_of_determination,
                          'MAE': self.__stratified_mean_absolute_error,
                          'MSE': self.__stratified_mean_squared_error,
                          'RMSE': self.__stratified_root_mean_squared_error,
                          'NRMSE': self.__stratified_normalized_root_mean_squared_error}
        if comparison is not None:
            __comparison_metrics_dict = {'R2': comparison.stratified_coefficient_of_determination,
                                         'MAE': comparison.stratified_mean_absolute_error,
                                         'MSE': comparison.stratified_mean_squared_error,
                                         'RMSE': comparison.stratified_root_mean_squared_error,
                                         'NRMSE': comparison.stratified_normalized_root_mean_squared_error}

        if not isinstance(table_format, list):
            raise ValueError("Parameter `table_format` has to be of type `str`.")

        for item in table_format:
            if item not in __table_formats:
                raise ValueError("Parameter `table_format` can only contain 'raw', 'tex' and/or 'pandas'.")

        if not isinstance(float_format, str):
            raise ValueError("Parameter `float_format` has to be of type `str`.")

        if metrics is not None:
            if not isinstance(metrics, list):
                raise ValueError("Parameter `metrics` has to be of type `list`.")

            for item in metrics:
                if item not in __metrics_names:
                    raise ValueError("Parameter `metrics` can only be: 'R2', 'MAE', 'MSE', 'RMSE', 'NRMSE'.")
        else:
            metrics = __metrics_names

        if comparison is None:

            for item in set(table_format):

                if item=='raw':

                    for i in range(0,self.__n_clusters):

                        print('-'*25 + '\n' + __clusters_names[i])

                        metrics_to_print = [self.__cluster_populations[i], self.__cluster_min[i], self.__cluster_max[i]]
                        for metric in metrics:
                            metrics_to_print.append(__metrics_dict[metric][i])

                        print('Observations' + ':\t' + str(metrics_to_print[0]))
                        print('Min' + ':\t' + ('%' + float_format) % metrics_to_print[1])
                        print('Max' + ':\t' + ('%' + float_format) % metrics_to_print[2])
                        for j in range(0,len(metrics)):
                            print(metrics[j] + ':\t' + ('%' + float_format) % metrics_to_print[j+3])

                if item=='tex':

                    import pandas as pd

                    metrics_to_print = np.vstack((self.__cluster_populations, self.__cluster_min, self.__cluster_max))

                    for metric in metrics:
                        metrics_to_print = np.vstack((metrics_to_print, __metrics_dict[metric]))

                    metrics_table = pd.DataFrame(metrics_to_print, columns=__clusters_names, index=['Observations', 'Min', 'Max'] +  metrics)
                    generate_tex_table(metrics_table, float_format=float_format)

                if item=='pandas':

                    import pandas as pd
                    from IPython.display import display
                    pandas_format = '{:,' + float_format + '}'

                    metrics_to_print = np.hstack((np.array(self.__cluster_populations)[:,None], np.array(self.__cluster_min)[:,None], np.array(self.__cluster_max)[:,None]))

                    for metric in metrics:
                        metrics_to_print = np.hstack((metrics_to_print, np.array(__metrics_dict[metric])[:,None]))

                    metrics_table = pd.DataFrame(metrics_to_print, columns=['Observations', 'Min', 'Max'] + metrics, index=__clusters_names)

                    metrics_table['Observations'] = metrics_table['Observations'].astype(int)
                    metrics_table['Min'] = metrics_table['Min'].map(pandas_format.format)
                    metrics_table['Max'] = metrics_table['Max'].map(pandas_format.format)
                    for metric in metrics:
                        metrics_table[metric] = metrics_table[metric].map(pandas_format.format)

                    display(metrics_table)

        else:

            for item in set(table_format):

                if item=='raw':

                    for i in range(0,self.__n_clusters):

                        print('-'*25 + '\n' + __clusters_names[i])

                        metrics_to_print = [self.__cluster_populations[i], self.__cluster_min[i], self.__cluster_max[i]]
                        comparison_metrics_to_print = [self.__cluster_populations[i], self.__cluster_min[i], self.__cluster_max[i]]
                        for metric in metrics:
                            metrics_to_print.append(__metrics_dict[metric][i])
                            comparison_metrics_to_print.append(__comparison_metrics_dict[metric][i])

                        print('Observations' + ':\t' + str(metrics_to_print[0]))
                        print('Min' + ':\t' + ('%' + float_format) % metrics_to_print[1])
                        print('Max' + ':\t' + ('%' + float_format) % metrics_to_print[2])
                        for j, metric in enumerate(metrics):

                            if metric=='R2':
                                if metrics_to_print[j+3] > comparison_metrics_to_print[j+3]:
                                    print(metric + ':\t' + ('%' + float_format) % metrics_to_print[j+3] + colored('\tBETTER', 'green'))
                                elif metrics_to_print[j+3] < comparison_metrics_to_print[j+3]:
                                    print(metric + ':\t' + ('%' + float_format) % metrics_to_print[j+3] + colored('\tWORSE', 'red'))
                                elif metrics_to_print[j+3] == comparison_metrics_to_print[j+3]:
                                    print(metric + ':\t' + ('%' + float_format) % metrics_to_print[j+3] + '\tSAME')
                            else:
                                if metrics_to_print[j+3] > comparison_metrics_to_print[j+3]:
                                    print(metric + ':\t' + ('%' + float_format) % metrics_to_print[j+3] + colored('\tWORSE', 'red'))
                                elif metrics_to_print[j+3] < comparison_metrics_to_print[j+3]:
                                    print(metric + ':\t' + ('%' + float_format) % metrics_to_print[j+3] + colored('\tBETTER', 'green'))
                                elif metrics_to_print[j+3] == comparison_metrics_to_print[j+3]:
                                    print(metric + ':\t' + ('%' + float_format) % metrics_to_print[j+3] + '\tSAME')

                if item=='pandas':

                    import pandas as pd
                    from IPython.display import display
                    pandas_format = '{:,' + float_format + '}'

                    metrics_to_print = np.hstack((np.array(self.__cluster_populations)[:,None], np.array(self.__cluster_min)[:,None], np.array(self.__cluster_max)[:,None]))
                    comparison_metrics_to_print = np.hstack((np.array(self.__cluster_populations)[:,None], np.array(self.__cluster_min)[:,None], np.array(self.__cluster_max)[:,None]))
                    for metric in metrics:
                        metrics_to_print = np.hstack((metrics_to_print, np.array(__metrics_dict[metric])[:,None]))
                        comparison_metrics_to_print = np.hstack((comparison_metrics_to_print, np.array(__comparison_metrics_dict[metric])[:,None]))

                    def highlight_better(data, data_comparison, color='lightgreen'):

                        attr = 'background-color: {}'.format(color)

                        is_better = False * data

                        # Lower value is better (MAE, MSE, RMSE, NRMSE):
                        try:
                            is_better['MAE'] = data['MAE'].astype(float) < data_comparison['MAE']
                        except:
                            pass
                        try:
                            is_better['MSE'] = data['MSE'].astype(float) < data_comparison['MSE']
                        except:
                            pass
                        try:
                            is_better['RMSE'] = data['RMSE'].astype(float) < data_comparison['RMSE']
                        except:
                            pass
                        try:
                            is_better['NRMSE'] = data['NRMSE'].astype(float) < data_comparison['NRMSE']
                        except:
                            pass

                        # Higher value is better (R2):
                        try:
                            is_better['R2'] = data['R2'].astype(float) > data_comparison['R2']
                        except:
                            pass

                        formatting = [attr if v else '' for v in is_better]

                        formatting = pd.DataFrame(np.where(is_better, attr, ''), index=data.index, columns=data.columns)

                        return formatting

                    def highlight_worse(data, data_comparison, color='salmon'):

                        attr = 'background-color: {}'.format(color)

                        is_worse = False * data

                        # Higher value is worse (MAE, MSE, RMSE, NRMSE):
                        try:
                            is_worse['MAE'] = data['MAE'].astype(float) > data_comparison['MAE']
                        except:
                            pass
                        try:
                            is_worse['MSE'] = data['MSE'].astype(float) > data_comparison['MSE']
                        except:
                            pass
                        try:
                            is_worse['RMSE'] = data['RMSE'].astype(float) > data_comparison['RMSE']
                        except:
                            pass
                        try:
                            is_worse['NRMSE'] = data['NRMSE'].astype(float) > data_comparison['NRMSE']
                        except:
                            pass

                        # Lower value is worse (R2):
                        try:
                            is_worse['R2'] = data['R2'].astype(float) < data_comparison['R2']
                        except:
                            pass

                        formatting = [attr if v else '' for v in is_worse]

                        formatting = pd.DataFrame(np.where(is_worse, attr, ''), index=data.index, columns=data.columns)

                        return formatting

                    metrics_table = pd.DataFrame(metrics_to_print, columns=['Observations', 'Min', 'Max'] + metrics, index=__clusters_names)
                    comparison_metrics_table = pd.DataFrame(comparison_metrics_to_print, columns=['Observations', 'Min', 'Max'] + metrics, index=__clusters_names)

                    metrics_table['Observations'] = metrics_table['Observations'].astype(int)
                    metrics_table['Min'] = metrics_table['Min'].map(pandas_format.format)
                    metrics_table['Max'] = metrics_table['Max'].map(pandas_format.format)
                    for metric in metrics:
                        metrics_table[metric] = metrics_table[metric].map(pandas_format.format)

                    formatted_table = metrics_table.style.apply(highlight_better, data_comparison=comparison_metrics_table, axis=None)\
                                                         .apply(highlight_worse, data_comparison=comparison_metrics_table, axis=None)

                    display(formatted_table)

# ------------------------------------------------------------------------------

def coefficient_of_determination(observed, predicted):
    """
    Computes the coefficient of determination, :math:`R^2`, value:

    .. math::

        R^2 = 1 - \\frac{\\sum_{i=1}^N (\\phi_{o,i} - \\phi_{p,i})^2}{\\sum_{i=1}^N (\\phi_{o,i} - \\mathrm{mean}(\\phi_{o,i}))^2}

    where :math:`N` is the number of observations, :math:`\\phi_o` is the observed and
    :math:`\\phi_p` is the predicted dependent variable.

    **Example:**

    .. code:: python

        from PCAfold import PCA, coefficient_of_determination
        import numpy as np

        # Generate dummy data set:
        X = np.random.rand(100,3)

        # Instantiate PCA class object:
        pca_X = PCA(X, scaling='auto', n_components=2)

        # Approximate the data set:
        X_rec = pca_X.reconstruct(pca_X.transform(X))

        # Compute the coefficient of determination for the first variable:
        r2 = coefficient_of_determination(X[:,0], X_rec[:,0])

    :param observed:
        ``numpy.ndarray`` specifying the observed values of a single dependent variable, :math:`\\phi_o`. It should be of size ``(n_observations,)`` or ``(n_observations, 1)``.
    :param predicted:
        ``numpy.ndarray`` specifying the predicted values of a single dependent variable, :math:`\\phi_p`. It should be of size ``(n_observations,)`` or ``(n_observations, 1)``.

    :return:
        - **r2** - coefficient of determination, :math:`R^2`.
    """

    if not isinstance(observed, np.ndarray):
        raise ValueError("Parameter `observed` has to be of type `numpy.ndarray`.")

    try:
        (n_observed,) = np.shape(observed)
        n_var_observed = 1
    except:
        (n_observed, n_var_observed) = np.shape(observed)

    if n_var_observed != 1:
        raise ValueError("Parameter `observed` has to be a 0D or 1D vector.")

    if not isinstance(predicted, np.ndarray):
        raise ValueError("Parameter `predicted` has to be of type `numpy.ndarray`.")

    try:
        (n_predicted,) = np.shape(predicted)
        n_var_predicted = 1
    except:
        (n_predicted, n_var_predicted) = np.shape(predicted)

    if n_var_predicted != 1:
        raise ValueError("Parameter `predicted` has to be a 0D or 1D vector.")

    if n_observed != n_predicted:
        raise ValueError("Parameter `observed` has different number of elements than `predicted`.")

    __observed = observed.ravel()
    __predicted = predicted.ravel()

    r2 = 1. - np.sum((__observed - __predicted) * (__observed - __predicted)) / np.sum(
        (__observed - np.mean(__observed)) * (__observed - np.mean(__observed)))

    return r2

# ------------------------------------------------------------------------------

def stratified_coefficient_of_determination(observed, predicted, idx, use_global_mean=True, verbose=False):
    """
    Computes the stratified coefficient of determination,
    :math:`R^2`, values. Stratified :math:`R^2` is computed separately in each
    bin (cluster) of an observed dependent variable, :math:`\\phi_o`.

    :math:`R_j^2` in the :math:`j^{th}` bin can be computed in two ways:

    - If ``use_global_mean=True``, the mean of the entire observed variable is used as a reference:

    .. math::

        R_j^2 = 1 - \\frac{\\sum_{i=1}^{N_j} (\\phi_{o,i}^{j} - \\phi_{p,i}^{j})^2}{\\sum_{i=1}^{N_j} (\\phi_{o,i}^{j} - \\mathrm{mean}(\\phi_o))^2}

    - If ``use_global_mean=False``, the mean of the considered :math:`j^{th}` bin is used as a reference:

    .. math::

        R_j^2 = 1 - \\frac{\\sum_{i=1}^{N_j} (\\phi_{o,i}^{j} - \\phi_{p,i}^{j})^2}{\\sum_{i=1}^{N_j} (\\phi_{o,i}^{j} - \\mathrm{mean}(\\phi_o^{j}))^2}

    where :math:`N_j` is the number of observations in the :math:`j^{th}` bin and
    :math:`\\phi_p` is the predicted dependent variable.

    .. note::

        After running this function you can call
        ``analysis.plot_stratified_coefficient_of_determination(r2_in_bins, bins_borders)`` on the
        function outputs and it will visualize how stratified :math:`R^2` changes across bins.

    .. warning::

        The stratified :math:`R^2` metric can be misleading if there are large
        variations in point density in an observed variable. For instance, below is a data set
        composed of lines of points that have uniform spacing on the :math:`x` axis
        but become more and more sparse in the direction of increasing :math:`\\phi`
        due to an increasing gradient of :math:`\\phi`.
        If bins are narrow enough (number of bins is high enough), a single bin
        (like the bin bounded by the red dashed lines) can contain only one of
        those lines of points for high value of :math:`\\phi`. :math:`R^2` will then be computed
        for constant, or almost constant observations, even though globally those
        observations lie in a location of a large gradient of the observed variable!

        .. image:: ../images/stratified-r2.png
            :width: 500
            :align: center

    **Example:**

    .. code:: python

        from PCAfold import PCA, variable_bins, stratified_coefficient_of_determination, plot_stratified_coefficient_of_determination
        import numpy as np

        # Generate dummy data set:
        X = np.random.rand(100,10)

        # Instantiate PCA class object:
        pca_X = PCA(X, scaling='auto', n_components=2)

        # Approximate the data set:
        X_rec = pca_X.reconstruct(pca_X.transform(X))

        # Generate bins:
        (idx, bins_borders) = variable_bins(X[:,0], k=10, verbose=False)

        # Compute stratified R2 in 10 bins of the first variable in a data set:
        r2_in_bins = stratified_coefficient_of_determination(X[:,0], X_rec[:,0], idx=idx, use_global_mean=True, verbose=True)

        # Plot the stratified R2 values:
        plot_stratified_coefficient_of_determination(r2_in_bins, bins_borders)

    :param observed:
        ``numpy.ndarray`` specifying the observed values of a single dependent variable, :math:`\\phi_o`. It should be of size ``(n_observations,)`` or ``(n_observations, 1)``.
    :param predicted:
        ``numpy.ndarray`` specifying the predicted values of a single dependent variable, :math:`\\phi_p`. It should be of size ``(n_observations,)`` or ``(n_observations, 1)``.
    :param idx:
        ``numpy.ndarray`` of cluster classifications. It should be of size ``(n_observations,)`` or ``(n_observations,1)``.
    :param use_global_mean: (optional)
            ``bool`` specifying if global mean of the observed variable should be used as a reference in :math:`R^2` calculation.
    :param verbose: (optional)
        ``bool`` for printing sizes (number of observations) and :math:`R^2` values in each bin.

    :return:
        - **r2_in_bins** - ``list`` specifying the coefficients of determination :math:`R^2` in each bin. It has length ``k``.
    """

    if not isinstance(observed, np.ndarray):
        raise ValueError("Parameter `observed` has to be of type `numpy.ndarray`.")

    try:
        (n_observed,) = np.shape(observed)
        n_var_observed = 1
    except:
        (n_observed, n_var_observed) = np.shape(observed)

    if n_var_observed != 1:
        raise ValueError("Parameter `observed` has to be a 0D or 1D vector.")

    if not isinstance(predicted, np.ndarray):
        raise ValueError("Parameter `predicted` has to be of type `numpy.ndarray`.")

    try:
        (n_predicted,) = np.shape(predicted)
        n_var_predicted = 1
    except:
        (n_predicted, n_var_predicted) = np.shape(predicted)

    if n_var_predicted != 1:
        raise ValueError("Parameter `predicted` has to be a 0D or 1D vector.")

    if n_observed != n_predicted:
        raise ValueError("Parameter `observed` has different number of elements than `predicted`.")

    if isinstance(idx, np.ndarray):
        if not all(isinstance(i, np.integer) for i in idx.ravel()):
            raise ValueError("Parameter `idx` can only contain integers.")
    else:
        raise ValueError("Parameter `idx` has to be of type `numpy.ndarray`.")

    try:
        (n_observations_idx, ) = np.shape(idx)
        n_idx = 1
    except:
        (n_observations_idx, n_idx) = np.shape(idx)

    if n_idx != 1:
        raise ValueError("Parameter `idx` has to have size `(n_observations,)` or `(n_observations,1)`.")

    if n_observations_idx != n_observed:
        raise ValueError('Vector of cluster classifications `idx` has different number of observations than the original data set `X`.')

    if not isinstance(use_global_mean, bool):
        raise ValueError("Parameter `use_global_mean` has to be a boolean.")

    if not isinstance(verbose, bool):
        raise ValueError("Parameter `verbose` has to be a boolean.")

    __observed = observed.ravel()
    __predicted = predicted.ravel()

    r2_in_bins = []

    if use_global_mean:
        global_mean = np.mean(__observed)

    for cl in np.unique(idx):

        (idx_bin,) = np.where(idx==cl)

        if use_global_mean:
            r2 = 1. - np.sum((__observed[idx_bin] - __predicted[idx_bin]) * (__observed[idx_bin] - __predicted[idx_bin])) / np.sum(
                (__observed[idx_bin] - global_mean) * (__observed[idx_bin] - global_mean))
        else:
            r2 = coefficient_of_determination(__observed[idx_bin], __predicted[idx_bin])

        constant_bin_metric_min = np.min(__observed[idx_bin])/np.mean(__observed[idx_bin])
        constant_bin_metric_max = np.max(__observed[idx_bin])/np.mean(__observed[idx_bin])

        if verbose:
            if (abs(constant_bin_metric_min - 1) < 0.01) and (abs(constant_bin_metric_max - 1) < 0.01):
                print('Bin\t' + str(cl+1) + '\t| size\t ' + str(len(idx_bin)) + '\t| R2\t' + str(round(r2,6)) + '\t| ' + colored('This bin has almost constant values.', 'red'))
            else:
                print('Bin\t' + str(cl+1) + '\t| size\t ' + str(len(idx_bin)) + '\t| R2\t' + str(round(r2,6)))

        r2_in_bins.append(r2)

    return r2_in_bins

# ------------------------------------------------------------------------------

def mean_absolute_error(observed, predicted):
    """
    Computes the mean absolute error (MAE):

    .. math::

        \\mathrm{MAE} = \\frac{1}{N} \\sum_{i=1}^N | \\phi_{o,i} - \\phi_{p,i} |

    where :math:`N` is the number of observations, :math:`\\phi_o` is the observed and
    :math:`\\phi_p` is the predicted dependent variable.

    **Example:**

    .. code:: python

        from PCAfold import PCA, mean_absolute_error
        import numpy as np

        # Generate dummy data set:
        X = np.random.rand(100,3)

        # Instantiate PCA class object:
        pca_X = PCA(X, scaling='auto', n_components=2)

        # Approximate the data set:
        X_rec = pca_X.reconstruct(pca_X.transform(X))

        # Compute the mean absolute error for the first variable:
        mae = mean_absolute_error(X[:,0], X_rec[:,0])

    :param observed:
        ``numpy.ndarray`` specifying the observed values of a single dependent variable, :math:`\\phi_o`. It should be of size ``(n_observations,)`` or ``(n_observations, 1)``.
    :param predicted:
        ``numpy.ndarray`` specifying the predicted values of a single dependent variable, :math:`\\phi_p`. It should be of size ``(n_observations,)`` or ``(n_observations, 1)``.

    :return:
        - **mae** - mean absolute error (MAE).
    """

    if not isinstance(observed, np.ndarray):
        raise ValueError("Parameter `observed` has to be of type `numpy.ndarray`.")

    try:
        (n_observed,) = np.shape(observed)
        n_var_observed = 1
    except:
        (n_observed, n_var_observed) = np.shape(observed)

    if n_var_observed != 1:
        raise ValueError("Parameter `observed` has to be a 0D or 1D vector.")

    if not isinstance(predicted, np.ndarray):
        raise ValueError("Parameter `predicted` has to be of type `numpy.ndarray`.")

    try:
        (n_predicted,) = np.shape(predicted)
        n_var_predicted = 1
    except:
        (n_predicted, n_var_predicted) = np.shape(predicted)

    if n_var_predicted != 1:
        raise ValueError("Parameter `predicted` has to be a 0D or 1D vector.")

    if n_observed != n_predicted:
        raise ValueError("Parameter `observed` has different number of elements than `predicted`.")

    __observed = observed.ravel()
    __predicted = predicted.ravel()

    mae = np.sum(abs(__observed - __predicted)) / n_observed

    return mae

# ------------------------------------------------------------------------------

def stratified_mean_absolute_error(observed, predicted, idx, verbose=False):
    """
    Computes the stratified mean absolute error (MAE) values. Stratified MAE is computed separately in each
    bin (cluster) of an observed dependent variable, :math:`\\phi_o`.

    MAE in the :math:`j^{th}` bin can be computed as:

    .. math::

        \\mathrm{MAE}_j = \\frac{1}{N_j} \\sum_{i=1}^{N_j} | \\phi_{o,i}^j - \\phi_{p,i}^j |

    where :math:`N_j` is the number of observations in the :math:`j^{th}` bin, :math:`\\phi_o` is the observed and
    :math:`\\phi_p` is the predicted dependent variable.

    **Example:**

    .. code:: python

        from PCAfold import PCA, variable_bins, stratified_mean_absolute_error
        import numpy as np

        # Generate dummy data set:
        X = np.random.rand(100,10)

        # Instantiate PCA class object:
        pca_X = PCA(X, scaling='auto', n_components=2)

        # Approximate the data set:
        X_rec = pca_X.reconstruct(pca_X.transform(X))

        # Generate bins:
        (idx, bins_borders) = variable_bins(X[:,0], k=10, verbose=False)

        # Compute stratified MAE in 10 bins of the first variable in a data set:
        mae_in_bins = stratified_mean_absolute_error(X[:,0], X_rec[:,0], idx=idx, verbose=True)

    :param observed:
        ``numpy.ndarray`` specifying the observed values of a single dependent variable, :math:`\\phi_o`. It should be of size ``(n_observations,)`` or ``(n_observations, 1)``.
    :param predicted:
        ``numpy.ndarray`` specifying the predicted values of a single dependent variable, :math:`\\phi_p`. It should be of size ``(n_observations,)`` or ``(n_observations, 1)``.
    :param idx:
        ``numpy.ndarray`` of cluster classifications. It should be of size ``(n_observations,)`` or ``(n_observations,1)``.
    :param verbose: (optional)
        ``bool`` for printing sizes (number of observations) and MAE values in each bin.

    :return:
        - **mae_in_bins** - ``list`` specifying the mean absolute error (MAE) in each bin. It has length ``k``.
    """

    if not isinstance(observed, np.ndarray):
        raise ValueError("Parameter `observed` has to be of type `numpy.ndarray`.")

    try:
        (n_observed,) = np.shape(observed)
        n_var_observed = 1
    except:
        (n_observed, n_var_observed) = np.shape(observed)

    if n_var_observed != 1:
        raise ValueError("Parameter `observed` has to be a 0D or 1D vector.")

    if not isinstance(predicted, np.ndarray):
        raise ValueError("Parameter `predicted` has to be of type `numpy.ndarray`.")

    try:
        (n_predicted,) = np.shape(predicted)
        n_var_predicted = 1
    except:
        (n_predicted, n_var_predicted) = np.shape(predicted)

    if n_var_predicted != 1:
        raise ValueError("Parameter `predicted` has to be a 0D or 1D vector.")

    if n_observed != n_predicted:
        raise ValueError("Parameter `observed` has different number of elements than `predicted`.")

    if isinstance(idx, np.ndarray):
        if not all(isinstance(i, np.integer) for i in idx.ravel()):
            raise ValueError("Parameter `idx` can only contain integers.")
    else:
        raise ValueError("Parameter `idx` has to be of type `numpy.ndarray`.")

    try:
        (n_observations_idx, ) = np.shape(idx)
        n_idx = 1
    except:
        (n_observations_idx, n_idx) = np.shape(idx)

    if n_idx != 1:
        raise ValueError("Parameter `idx` has to have size `(n_observations,)` or `(n_observations,1)`.")

    if n_observations_idx != n_observed:
        raise ValueError('Vector of cluster classifications `idx` has different number of observations than the original data set `X`.')

    if not isinstance(verbose, bool):
        raise ValueError("Parameter `verbose` has to be a boolean.")

    __observed = observed.ravel()
    __predicted = predicted.ravel()

    mae_in_bins = []

    for cl in np.unique(idx):

        (idx_bin,) = np.where(idx==cl)

        mae = mean_absolute_error(__observed[idx_bin], __predicted[idx_bin])

        constant_bin_metric_min = np.min(__observed[idx_bin])/np.mean(__observed[idx_bin])
        constant_bin_metric_max = np.max(__observed[idx_bin])/np.mean(__observed[idx_bin])

        if verbose:
            if (abs(constant_bin_metric_min - 1) < 0.01) and (abs(constant_bin_metric_max - 1) < 0.01):
                print('Bin\t' + str(cl+1) + '\t| size\t ' + str(len(idx_bin)) + '\t| MAE\t' + str(round(mae,6)) + '\t| ' + colored('This bin has almost constant values.', 'red'))
            else:
                print('Bin\t' + str(cl+1) + '\t| size\t ' + str(len(idx_bin)) + '\t| MAE\t' + str(round(mae,6)))

        mae_in_bins.append(mae)

    return mae_in_bins

# ------------------------------------------------------------------------------

def max_absolute_error(observed, predicted):
    """
    Computes the maximum absolute error (MaxAE):

    .. math::

        \\mathrm{MaxAE} = \\mathrm{max}( | \\phi_{o,i} - \\phi_{p,i} | )

    where :math:`\\phi_o` is the observed and
    :math:`\\phi_p` is the predicted dependent variable.

    **Example:**

    .. code:: python

        from PCAfold import PCA, max_absolute_error
        import numpy as np

        # Generate dummy data set:
        X = np.random.rand(100,3)

        # Instantiate PCA class object:
        pca_X = PCA(X, scaling='auto', n_components=2)

        # Approximate the data set:
        X_rec = pca_X.reconstruct(pca_X.transform(X))

        # Compute the maximum absolute error for the first variable:
        maxae = max_absolute_error(X[:,0], X_rec[:,0])

    :param observed:
        ``numpy.ndarray`` specifying the observed values of a single dependent variable, :math:`\\phi_o`. It should be of size ``(n_observations,)`` or ``(n_observations, 1)``.
    :param predicted:
        ``numpy.ndarray`` specifying the predicted values of a single dependent variable, :math:`\\phi_p`. It should be of size ``(n_observations,)`` or ``(n_observations, 1)``.

    :return:
        - **maxae** - max absolute error (MaxAE).
    """

    if not isinstance(observed, np.ndarray):
        raise ValueError("Parameter `observed` has to be of type `numpy.ndarray`.")

    try:
        (n_observed,) = np.shape(observed)
        n_var_observed = 1
    except:
        (n_observed, n_var_observed) = np.shape(observed)

    if n_var_observed != 1:
        raise ValueError("Parameter `observed` has to be a 0D or 1D vector.")

    if not isinstance(predicted, np.ndarray):
        raise ValueError("Parameter `predicted` has to be of type `numpy.ndarray`.")

    try:
        (n_predicted,) = np.shape(predicted)
        n_var_predicted = 1
    except:
        (n_predicted, n_var_predicted) = np.shape(predicted)

    if n_var_predicted != 1:
        raise ValueError("Parameter `predicted` has to be a 0D or 1D vector.")

    if n_observed != n_predicted:
        raise ValueError("Parameter `observed` has different number of elements than `predicted`.")

    __observed = observed.ravel()
    __predicted = predicted.ravel()

    maxae = np.max(abs(__observed - __predicted))

    return maxae

# ------------------------------------------------------------------------------

def mean_squared_error(observed, predicted):
    """
    Computes the mean squared error (MSE):

    .. math::

        \\mathrm{MSE} = \\frac{1}{N} \\sum_{i=1}^N (\\phi_{o,i} - \\phi_{p,i}) ^2

    where :math:`N` is the number of observations, :math:`\\phi_o` is the observed and
    :math:`\\phi_p` is the predicted dependent variable.

    **Example:**

    .. code:: python

        from PCAfold import PCA, mean_squared_error
        import numpy as np

        # Generate dummy data set:
        X = np.random.rand(100,3)

        # Instantiate PCA class object:
        pca_X = PCA(X, scaling='auto', n_components=2)

        # Approximate the data set:
        X_rec = pca_X.reconstruct(pca_X.transform(X))

        # Compute the mean squared error for the first variable:
        mse = mean_squared_error(X[:,0], X_rec[:,0])

    :param observed:
        ``numpy.ndarray`` specifying the observed values of a single dependent variable, :math:`\\phi_o`. It should be of size ``(n_observations,)`` or ``(n_observations, 1)``.
    :param predicted:
        ``numpy.ndarray`` specifying the predicted values of a single dependent variable, :math:`\\phi_p`. It should be of size ``(n_observations,)`` or ``(n_observations, 1)``.

    :return:
        - **mse** - mean squared error (MSE).
    """

    if not isinstance(observed, np.ndarray):
        raise ValueError("Parameter `observed` has to be of type `numpy.ndarray`.")

    try:
        (n_observed,) = np.shape(observed)
        n_var_observed = 1
    except:
        (n_observed, n_var_observed) = np.shape(observed)

    if n_var_observed != 1:
        raise ValueError("Parameter `observed` has to be a 0D or 1D vector.")

    if not isinstance(predicted, np.ndarray):
        raise ValueError("Parameter `predicted` has to be of type `numpy.ndarray`.")

    try:
        (n_predicted,) = np.shape(predicted)
        n_var_predicted = 1
    except:
        (n_predicted, n_var_predicted) = np.shape(predicted)

    if n_var_predicted != 1:
        raise ValueError("Parameter `predicted` has to be a 0D or 1D vector.")

    if n_observed != n_predicted:
        raise ValueError("Parameter `observed` has different number of elements than `predicted`.")

    __observed = observed.ravel()
    __predicted = predicted.ravel()

    mse = 1.0 / n_observed * np.sum((__observed - __predicted) * (__observed - __predicted))

    return mse

# ------------------------------------------------------------------------------

def stratified_mean_squared_error(observed, predicted, idx, verbose=False):
    """
    Computes the stratified mean squared error (MSE) values. Stratified MSE is computed separately in each
    bin (cluster) of an observed dependent variable, :math:`\\phi_o`.

    MSE in the :math:`j^{th}` bin can be computed as:

    .. math::

        \\mathrm{MSE}_j = \\frac{1}{N_j} \\sum_{i=1}^{N_j} (\\phi_{o,i}^j - \\phi_{p,i}^j) ^2

    where :math:`N_j` is the number of observations in the :math:`j^{th}` bin, :math:`\\phi_o` is the observed and
    :math:`\\phi_p` is the predicted dependent variable.

    **Example:**

    .. code:: python

        from PCAfold import PCA, variable_bins, stratified_mean_squared_error
        import numpy as np

        # Generate dummy data set:
        X = np.random.rand(100,10)

        # Instantiate PCA class object:
        pca_X = PCA(X, scaling='auto', n_components=2)

        # Approximate the data set:
        X_rec = pca_X.reconstruct(pca_X.transform(X))

        # Generate bins:
        (idx, bins_borders) = variable_bins(X[:,0], k=10, verbose=False)

        # Compute stratified MSE in 10 bins of the first variable in a data set:
        mse_in_bins = stratified_mean_squared_error(X[:,0], X_rec[:,0], idx=idx, verbose=True)

    :param observed:
        ``numpy.ndarray`` specifying the observed values of a single dependent variable, :math:`\\phi_o`. It should be of size ``(n_observations,)`` or ``(n_observations, 1)``.
    :param predicted:
        ``numpy.ndarray`` specifying the predicted values of a single dependent variable, :math:`\\phi_p`. It should be of size ``(n_observations,)`` or ``(n_observations, 1)``.
    :param idx:
        ``numpy.ndarray`` of cluster classifications. It should be of size ``(n_observations,)`` or ``(n_observations,1)``.
    :param verbose: (optional)
        ``bool`` for printing sizes (number of observations) and MSE values in each bin.

    :return:
        - **mse_in_bins** - ``list`` specifying the mean squared error (MSE) in each bin. It has length ``k``.
    """

    if not isinstance(observed, np.ndarray):
        raise ValueError("Parameter `observed` has to be of type `numpy.ndarray`.")

    try:
        (n_observed,) = np.shape(observed)
        n_var_observed = 1
    except:
        (n_observed, n_var_observed) = np.shape(observed)

    if n_var_observed != 1:
        raise ValueError("Parameter `observed` has to be a 0D or 1D vector.")

    if not isinstance(predicted, np.ndarray):
        raise ValueError("Parameter `predicted` has to be of type `numpy.ndarray`.")

    try:
        (n_predicted,) = np.shape(predicted)
        n_var_predicted = 1
    except:
        (n_predicted, n_var_predicted) = np.shape(predicted)

    if n_var_predicted != 1:
        raise ValueError("Parameter `predicted` has to be a 0D or 1D vector.")

    if n_observed != n_predicted:
        raise ValueError("Parameter `observed` has different number of elements than `predicted`.")

    if isinstance(idx, np.ndarray):
        if not all(isinstance(i, np.integer) for i in idx.ravel()):
            raise ValueError("Parameter `idx` can only contain integers.")
    else:
        raise ValueError("Parameter `idx` has to be of type `numpy.ndarray`.")

    try:
        (n_observations_idx, ) = np.shape(idx)
        n_idx = 1
    except:
        (n_observations_idx, n_idx) = np.shape(idx)

    if n_idx != 1:
        raise ValueError("Parameter `idx` has to have size `(n_observations,)` or `(n_observations,1)`.")

    if n_observations_idx != n_observed:
        raise ValueError('Vector of cluster classifications `idx` has different number of observations than the original data set `X`.')

    if not isinstance(verbose, bool):
        raise ValueError("Parameter `verbose` has to be a boolean.")

    __observed = observed.ravel()
    __predicted = predicted.ravel()

    mse_in_bins = []

    for cl in np.unique(idx):

        (idx_bin,) = np.where(idx==cl)

        mse = mean_squared_error(__observed[idx_bin], __predicted[idx_bin])

        constant_bin_metric_min = np.min(__observed[idx_bin])/np.mean(__observed[idx_bin])
        constant_bin_metric_max = np.max(__observed[idx_bin])/np.mean(__observed[idx_bin])

        if verbose:
            if (abs(constant_bin_metric_min - 1) < 0.01) and (abs(constant_bin_metric_max - 1) < 0.01):
                print('Bin\t' + str(cl+1) + '\t| size\t ' + str(len(idx_bin)) + '\t| MSE\t' + str(round(mse,6)) + '\t| ' + colored('This bin has almost constant values.', 'red'))
            else:
                print('Bin\t' + str(cl+1) + '\t| size\t ' + str(len(idx_bin)) + '\t| MSE\t' + str(round(mse,6)))

        mse_in_bins.append(mse)

    return mse_in_bins

# ------------------------------------------------------------------------------

def root_mean_squared_error(observed, predicted):
    """
    Computes the root mean squared error (RMSE):

    .. math::

        \\mathrm{RMSE} = \\sqrt{\\frac{1}{N} \\sum_{i=1}^N (\\phi_{o,i} - \\phi_{p,i}) ^2}

    where :math:`N` is the number of observations, :math:`\\phi_o` is the observed and
    :math:`\\phi_p` is the predicted dependent variable.

    **Example:**

    .. code:: python

        from PCAfold import PCA, root_mean_squared_error
        import numpy as np

        # Generate dummy data set:
        X = np.random.rand(100,3)

        # Instantiate PCA class object:
        pca_X = PCA(X, scaling='auto', n_components=2)

        # Approximate the data set:
        X_rec = pca_X.reconstruct(pca_X.transform(X))

        # Compute the root mean squared error for the first variable:
        rmse = root_mean_squared_error(X[:,0], X_rec[:,0])

    :param observed:
        ``numpy.ndarray`` specifying the observed values of a single dependent variable, :math:`\\phi_o`. It should be of size ``(n_observations,)`` or ``(n_observations, 1)``.
    :param predicted:
        ``numpy.ndarray`` specifying the predicted values of a single dependent variable, :math:`\\phi_p`. It should be of size ``(n_observations,)`` or ``(n_observations, 1)``.

    :return:
        - **rmse** - root mean squared error (RMSE).
    """

    if not isinstance(observed, np.ndarray):
        raise ValueError("Parameter `observed` has to be of type `numpy.ndarray`.")

    try:
        (n_observed,) = np.shape(observed)
        n_var_observed = 1
    except:
        (n_observed, n_var_observed) = np.shape(observed)

    if n_var_observed != 1:
        raise ValueError("Parameter `observed` has to be a 0D or 1D vector.")

    if not isinstance(predicted, np.ndarray):
        raise ValueError("Parameter `predicted` has to be of type `numpy.ndarray`.")

    try:
        (n_predicted,) = np.shape(predicted)
        n_var_predicted = 1
    except:
        (n_predicted, n_var_predicted) = np.shape(predicted)

    if n_var_predicted != 1:
        raise ValueError("Parameter `predicted` has to be a 0D or 1D vector.")

    if n_observed != n_predicted:
        raise ValueError("Parameter `observed` has different number of elements than `predicted`.")

    __observed = observed.ravel()
    __predicted = predicted.ravel()

    rmse = (mean_squared_error(__observed, __predicted))**0.5

    return rmse

# ------------------------------------------------------------------------------

def stratified_root_mean_squared_error(observed, predicted, idx, verbose=False):
    """
    Computes the stratified root mean squared error (RMSE) values. Stratified RMSE is computed separately in each
    bin (cluster) of an observed dependent variable, :math:`\\phi_o`.

    RMSE in the :math:`j^{th}` bin can be computed as:

    .. math::

        \\mathrm{RMSE}_j = \\sqrt{\\frac{1}{N_j} \\sum_{i=1}^{N_j} (\\phi_{o,i}^j - \\phi_{p,i}^j) ^2}

    where :math:`N_j` is the number of observations in the :math:`j^{th}` bin, :math:`\\phi_o` is the observed and
    :math:`\\phi_p` is the predicted dependent variable.

    **Example:**

    .. code:: python

        from PCAfold import PCA, variable_bins, stratified_root_mean_squared_error
        import numpy as np

        # Generate dummy data set:
        X = np.random.rand(100,10)

        # Instantiate PCA class object:
        pca_X = PCA(X, scaling='auto', n_components=2)

        # Approximate the data set:
        X_rec = pca_X.reconstruct(pca_X.transform(X))

        # Generate bins:
        (idx, bins_borders) = variable_bins(X[:,0], k=10, verbose=False)

        # Compute stratified RMSE in 10 bins of the first variable in a data set:
        rmse_in_bins = stratified_root_mean_squared_error(X[:,0], X_rec[:,0], idx=idx, verbose=True)

    :param observed:
        ``numpy.ndarray`` specifying the observed values of a single dependent variable, :math:`\\phi_o`. It should be of size ``(n_observations,)`` or ``(n_observations, 1)``.
    :param predicted:
        ``numpy.ndarray`` specifying the predicted values of a single dependent variable, :math:`\\phi_p`. It should be of size ``(n_observations,)`` or ``(n_observations, 1)``.
    :param idx:
        ``numpy.ndarray`` of cluster classifications. It should be of size ``(n_observations,)`` or ``(n_observations,1)``.
    :param verbose: (optional)
        ``bool`` for printing sizes (number of observations) and RMSE values in each bin.

    :return:
        - **rmse_in_bins** - ``list`` specifying the mean squared error (RMSE) in each bin. It has length ``k``.
    """

    if not isinstance(observed, np.ndarray):
        raise ValueError("Parameter `observed` has to be of type `numpy.ndarray`.")

    try:
        (n_observed,) = np.shape(observed)
        n_var_observed = 1
    except:
        (n_observed, n_var_observed) = np.shape(observed)

    if n_var_observed != 1:
        raise ValueError("Parameter `observed` has to be a 0D or 1D vector.")

    if not isinstance(predicted, np.ndarray):
        raise ValueError("Parameter `predicted` has to be of type `numpy.ndarray`.")

    try:
        (n_predicted,) = np.shape(predicted)
        n_var_predicted = 1
    except:
        (n_predicted, n_var_predicted) = np.shape(predicted)

    if n_var_predicted != 1:
        raise ValueError("Parameter `predicted` has to be a 0D or 1D vector.")

    if n_observed != n_predicted:
        raise ValueError("Parameter `observed` has different number of elements than `predicted`.")

    if isinstance(idx, np.ndarray):
        if not all(isinstance(i, np.integer) for i in idx.ravel()):
            raise ValueError("Parameter `idx` can only contain integers.")
    else:
        raise ValueError("Parameter `idx` has to be of type `numpy.ndarray`.")

    try:
        (n_observations_idx, ) = np.shape(idx)
        n_idx = 1
    except:
        (n_observations_idx, n_idx) = np.shape(idx)

    if n_idx != 1:
        raise ValueError("Parameter `idx` has to have size `(n_observations,)` or `(n_observations,1)`.")

    if n_observations_idx != n_observed:
        raise ValueError('Vector of cluster classifications `idx` has different number of observations than the original data set `X`.')

    if not isinstance(verbose, bool):
        raise ValueError("Parameter `verbose` has to be a boolean.")

    __observed = observed.ravel()
    __predicted = predicted.ravel()

    rmse_in_bins = []

    for cl in np.unique(idx):

        (idx_bin,) = np.where(idx==cl)

        rmse = root_mean_squared_error(__observed[idx_bin], __predicted[idx_bin])

        constant_bin_metric_min = np.min(__observed[idx_bin])/np.mean(__observed[idx_bin])
        constant_bin_metric_max = np.max(__observed[idx_bin])/np.mean(__observed[idx_bin])

        if verbose:
            if (abs(constant_bin_metric_min - 1) < 0.01) and (abs(constant_bin_metric_max - 1) < 0.01):
                print('Bin\t' + str(cl+1) + '\t| size\t ' + str(len(idx_bin)) + '\t| RMSE\t' + str(round(rmse,6)) + '\t| ' + colored('This bin has almost constant values.', 'red'))
            else:
                print('Bin\t' + str(cl+1) + '\t| size\t ' + str(len(idx_bin)) + '\t| RMSE\t' + str(round(rmse,6)))

        rmse_in_bins.append(rmse)

    return rmse_in_bins

# ------------------------------------------------------------------------------

def normalized_root_mean_squared_error(observed, predicted, norm='std'):
    """
    Computes the normalized root mean squared error (NRMSE):

    .. math::

        \\mathrm{NRMSE} = \\frac{1}{d_{norm}} \\sqrt{\\frac{1}{N} \\sum_{i=1}^N (\\phi_{o,i} - \\phi_{p,i}) ^2}

    where :math:`d_{norm}` is the normalization factor, :math:`N` is the number of observations, :math:`\\phi_o` is the observed and
    :math:`\\phi_p` is the predicted dependent variable.

    Various normalizations are available:

    +----------------------------+--------------------------+------------------------------------------------------------------------------+
    | Normalization              | ``norm``                 | Normalization factor :math:`d_{norm}`                                        |
    +============================+==========================+==============================================================================+
    | Root square mean           | ``'root_square_mean'``   | :math:`d_{norm} = \sqrt{\mathrm{mean}(\phi_o^2)}`                            |
    +----------------------------+--------------------------+------------------------------------------------------------------------------+
    | Std                        | ``'std'``                | :math:`d_{norm} = \mathrm{std}(\phi_o)`                                      |
    +----------------------------+--------------------------+------------------------------------------------------------------------------+
    | Range                      | ``'range'``              | :math:`d_{norm} = \mathrm{max}(\phi_o) - \mathrm{min}(\phi_o)`               |
    +----------------------------+--------------------------+------------------------------------------------------------------------------+
    | Root square range          | ``'root_square_range'``  | :math:`d_{norm} = \sqrt{\mathrm{max}(\phi_o^2) - \mathrm{min}(\phi_o^2)}``   |
    +----------------------------+--------------------------+------------------------------------------------------------------------------+
    | Root square std            | ``'root_square_std'``    | :math:`d_{norm} = \sqrt{\mathrm{std}(\phi_o^2)}`                             |
    +----------------------------+--------------------------+------------------------------------------------------------------------------+
    | Absolute mean              | ``'abs_mean'``           | :math:`d_{norm} = | \mathrm{mean}(\phi_o) |`                                 |
    +----------------------------+--------------------------+------------------------------------------------------------------------------+

    **Example:**

    .. code:: python

        from PCAfold import PCA, normalized_root_mean_squared_error
        import numpy as np

        # Generate dummy data set:
        X = np.random.rand(100,3)

        # Instantiate PCA class object:
        pca_X = PCA(X, scaling='auto', n_components=2)

        # Approximate the data set:
        X_rec = pca_X.reconstruct(pca_X.transform(X))

        # Compute the root mean squared error for the first variable:
        nrmse = normalized_root_mean_squared_error(X[:,0], X_rec[:,0], norm='std')

    :param observed:
        ``numpy.ndarray`` specifying the observed values of a single dependent variable, :math:`\\phi_o`. It should be of size ``(n_observations,)`` or ``(n_observations, 1)``.
    :param predicted:
        ``numpy.ndarray`` specifying the predicted values of a single dependent variable, :math:`\\phi_p`. It should be of size ``(n_observations,)`` or ``(n_observations, 1)``.
    :param norm:
        ``str`` specifying the normalization, :math:`d_{norm}`. It can be one of the following: ``std``, ``range``, ``root_square_mean``, ``root_square_range``, ``root_square_std``, ``abs_mean``.

    :return:
        - **nrmse** - normalized root mean squared error (NRMSE).
    """

    __norms = ['root_square_mean', 'std', 'range', 'root_square_range', 'root_square_std', 'abs_mean']

    if not isinstance(observed, np.ndarray):
        raise ValueError("Parameter `observed` has to be of type `numpy.ndarray`.")

    try:
        (n_observed,) = np.shape(observed)
        n_var_observed = 1
    except:
        (n_observed, n_var_observed) = np.shape(observed)

    if n_var_observed != 1:
        raise ValueError("Parameter `observed` has to be a 0D or 1D vector.")

    if not isinstance(predicted, np.ndarray):
        raise ValueError("Parameter `predicted` has to be of type `numpy.ndarray`.")

    try:
        (n_predicted,) = np.shape(predicted)
        n_var_predicted = 1
    except:
        (n_predicted, n_var_predicted) = np.shape(predicted)

    if n_var_predicted != 1:
        raise ValueError("Parameter `predicted` has to be a 0D or 1D vector.")

    if n_observed != n_predicted:
        raise ValueError("Parameter `observed` has different number of elements than `predicted`.")

    if norm not in __norms:
        raise ValueError("Parameter `norm` can be one of the following: ``std``, ``range``, ``root_square_mean``, ``root_square_range``, ``root_square_std``, ``abs_mean``.")

    __observed = observed.ravel()
    __predicted = predicted.ravel()

    rmse = root_mean_squared_error(__observed, __predicted)

    if norm == 'root_square_mean':
        nrmse = rmse/np.sqrt(np.mean(__observed**2))
    elif norm == 'std':
        nrmse = rmse/(np.std(__observed))
    elif norm == 'range':
        nrmse = rmse/(np.max(__observed) - np.min(__observed))
    elif norm == 'root_square_range':
        nrmse = rmse/np.sqrt(np.max(__observed**2) - np.min(__observed**2))
    elif norm == 'root_square_std':
        nrmse = rmse/np.sqrt(np.std(__observed**2))
    elif norm == 'abs_mean':
        nrmse = rmse/abs(np.mean(__observed))

    return nrmse

# ------------------------------------------------------------------------------

def stratified_normalized_root_mean_squared_error(observed, predicted, idx, norm='std', use_global_norm=False, verbose=False):
    """
    Computes the stratified normalized root mean squared error (NRMSE) values. Stratified NRMSE is computed separately in each
    bin (cluster) of an observed dependent variable, :math:`\\phi_o`.

    NRMSE in the :math:`j^{th}` bin can be computed as:

    .. math::

        \\mathrm{NRMSE}_j = \\frac{1}{d_{norm}} \\sqrt{\\frac{1}{N_j} \\sum_{i=1}^{N_j} (\\phi_{o,i}^j - \\phi_{p,i}^j) ^2}

    where :math:`N_j` is the number of observations in the :math:`j^{th}` bin, :math:`\\phi_o` is the observed and
    :math:`\\phi_p` is the predicted dependent variable.

    **Example:**

    .. code:: python

        from PCAfold import PCA, variable_bins, stratified_normalized_root_mean_squared_error
        import numpy as np

        # Generate dummy data set:
        X = np.random.rand(100,10)

        # Instantiate PCA class object:
        pca_X = PCA(X, scaling='auto', n_components=2)

        # Approximate the data set:
        X_rec = pca_X.reconstruct(pca_X.transform(X))

        # Generate bins:
        (idx, bins_borders) = variable_bins(X[:,0], k=10, verbose=False)

        # Compute stratified NRMSE in 10 bins of the first variable in a data set:
        nrmse_in_bins = stratified_normalized_root_mean_squared_error(X[:,0],
                                                                      X_rec[:,0],
                                                                      idx=idx,
                                                                      norm='std',
                                                                      use_global_norm=True,
                                                                      verbose=True)

    :param observed:
        ``numpy.ndarray`` specifying the observed values of a single dependent variable, :math:`\\phi_o`. It should be of size ``(n_observations,)`` or ``(n_observations, 1)``.
    :param predicted:
        ``numpy.ndarray`` specifying the predicted values of a single dependent variable, :math:`\\phi_p`. It should be of size ``(n_observations,)`` or ``(n_observations, 1)``.
    :param idx:
        ``numpy.ndarray`` of cluster classifications. It should be of size ``(n_observations,)`` or ``(n_observations,1)``.
    :param norm:
        ``str`` specifying the normalization, :math:`d_{norm}`. It can be one of the following: ``std``, ``range``, ``root_square_mean``, ``root_square_range``, ``root_square_std``, ``abs_mean``.
    :param use_global_norm: (optional)
            ``bool`` specifying if global norm of the observed variable should be used in NRMSE calculation. If set to ``False``, norms are computed on samples from the the corresponding bin.
    :param verbose: (optional)
        ``bool`` for printing sizes (number of observations) and NRMSE values in each bin.

    :return:
        - **nrmse_in_bins** - ``list`` specifying the mean squared error (NRMSE) in each bin. It has length ``k``.
    """

    if not isinstance(observed, np.ndarray):
        raise ValueError("Parameter `observed` has to be of type `numpy.ndarray`.")

    try:
        (n_observed,) = np.shape(observed)
        n_var_observed = 1
    except:
        (n_observed, n_var_observed) = np.shape(observed)

    if n_var_observed != 1:
        raise ValueError("Parameter `observed` has to be a 0D or 1D vector.")

    if not isinstance(predicted, np.ndarray):
        raise ValueError("Parameter `predicted` has to be of type `numpy.ndarray`.")

    try:
        (n_predicted,) = np.shape(predicted)
        n_var_predicted = 1
    except:
        (n_predicted, n_var_predicted) = np.shape(predicted)

    if n_var_predicted != 1:
        raise ValueError("Parameter `predicted` has to be a 0D or 1D vector.")

    if n_observed != n_predicted:
        raise ValueError("Parameter `observed` has different number of elements than `predicted`.")

    if isinstance(idx, np.ndarray):
        if not all(isinstance(i, np.integer) for i in idx.ravel()):
            raise ValueError("Parameter `idx` can only contain integers.")
    else:
        raise ValueError("Parameter `idx` has to be of type `numpy.ndarray`.")

    try:
        (n_observations_idx, ) = np.shape(idx)
        n_idx = 1
    except:
        (n_observations_idx, n_idx) = np.shape(idx)

    if n_idx != 1:
        raise ValueError("Parameter `idx` has to have size `(n_observations,)` or `(n_observations,1)`.")

    if n_observations_idx != n_observed:
        raise ValueError('Vector of cluster classifications `idx` has different number of observations than the original data set `X`.')

    if not isinstance(verbose, bool):
        raise ValueError("Parameter `verbose` has to be a boolean.")

    __observed = observed.ravel()
    __predicted = predicted.ravel()

    nrmse_in_bins = []

    for cl in np.unique(idx):

        (idx_bin,) = np.where(idx==cl)

        if use_global_norm:

            rmse = root_mean_squared_error(__observed[idx_bin], __predicted[idx_bin])

            if norm == 'root_square_mean':
                nrmse = rmse/np.sqrt(np.mean(__observed**2))
            elif norm == 'std':
                nrmse = rmse/(np.std(__observed))
            elif norm == 'range':
                nrmse = rmse/(np.max(__observed) - np.min(__observed))
            elif norm == 'root_square_range':
                nrmse = rmse/np.sqrt(np.max(__observed**2) - np.min(__observed**2))
            elif norm == 'root_square_std':
                nrmse = rmse/np.sqrt(np.std(__observed**2))
            elif norm == 'abs_mean':
                nrmse = rmse/abs(np.mean(__observed))

        else:

            nrmse = normalized_root_mean_squared_error(__observed[idx_bin], __predicted[idx_bin], norm=norm)

        constant_bin_metric_min = np.min(__observed[idx_bin])/np.mean(__observed[idx_bin])
        constant_bin_metric_max = np.max(__observed[idx_bin])/np.mean(__observed[idx_bin])

        if verbose:
            if (abs(constant_bin_metric_min - 1) < 0.01) and (abs(constant_bin_metric_max - 1) < 0.01):
                print('Bin\t' + str(cl+1) + '\t| size\t ' + str(len(idx_bin)) + '\t| NRMSE\t' + str(round(nrmse,6)) + '\t| ' + colored('This bin has almost constant values.', 'red'))
            else:
                print('Bin\t' + str(cl+1) + '\t| size\t ' + str(len(idx_bin)) + '\t| NRMSE\t' + str(round(nrmse,6)))

        nrmse_in_bins.append(nrmse)

    return nrmse_in_bins

# ------------------------------------------------------------------------------

def turning_points(observed, predicted):
    """
    Computes the turning points percentage - the percentage of predicted outputs
    that have the opposite growth tendency to the corresponding observed growth tendency.

    .. warning::

        This function is under construction.

    :return:
        - **turning_points** - turning points percentage in %.
    """

    return turning_points

# ------------------------------------------------------------------------------

def good_estimate(observed, predicted, tolerance=0.05):
    """
    Computes the good estimate (GE) - the percentage of predicted values that
    are within the specified tolerance from the corresponding observed values.

    .. warning::

        This function is under construction.

    :param observed:
        ``numpy.ndarray`` specifying the observed values of a single dependent variable. It should be of size ``(n_observations,)`` or ``(n_observations, 1)``.
    :param predicted:
        ``numpy.ndarray`` specifying the predicted values of a single dependent variable. It should be of size ``(n_observations,)`` or ``(n_observations, 1)``.
    :parm tolerance:
        ``float`` specifying the tolerance.

    :return:
        - **good_estimate** - good estimate (GE) in %.
    """

    return good_estimate

# ------------------------------------------------------------------------------

def good_direction_estimate(observed, predicted, tolerance=0.05):
    """
    Computes the good direction (GD) and the good direction estimate (GDE).

    GD for observation :math:`i`, is computed as:

    .. math::

        GD_i = \\frac{\\vec{\\phi}_{o,i}}{|| \\vec{\\phi}_{o,i} ||} \\cdot \\frac{\\vec{\\phi}_{p,i}}{|| \\vec{\\phi}_{p,i} ||}

    where :math:`\\vec{\\phi}_o` is the observed vector quantity and :math:`\\vec{\\phi}_p` is the
    predicted vector quantity.

    GDE is computed as the percentage of predicted vector observations whose
    direction is within the specified tolerance from the direction of the
    corresponding observed vector.

    **Example:**

    .. code:: python

        from PCAfold import PCA, good_direction_estimate
        import numpy as np

        # Generate dummy data set:
        X = np.random.rand(100,3)

        # Instantiate PCA class object:
        pca_X = PCA(X, scaling='auto', n_components=2)

        # Approximate the data set:
        X_rec = pca_X.reconstruct(pca_X.transform(X))

        # Compute the vector of good direction and good direction estimate:
        (good_direction, good_direction_estimate) = good_direction_estimate(X, X_rec, tolerance=0.01)

    :param observed:
        ``numpy.ndarray`` specifying the observed vector quantity, :math:`\\vec{\\phi}_o`. It should be of size ``(n_observations,n_dimensions)``.
    :param predicted:
        ``numpy.ndarray`` specifying the predicted vector quantity, :math:`\\vec{\\phi}_p`. It should be of size ``(n_observations,n_dimensions)``.
    :param tolerance:
        ``float`` specifying the tolerance.

    :return:
        - **good_direction** - ``numpy.ndarray`` specifying the vector of good direction (GD). It has size ``(n_observations,)``.
        - **good_direction_estimate** - good direction estimate (GDE) in %.
    """

    if not isinstance(observed, np.ndarray):
        raise ValueError("Parameter `observed` has to be of type `numpy.ndarray`.")

    try:
        (n_observed, n_dimensions_1) = np.shape(observed)
    except:
        raise ValueError("Parameter `observed` should be a matrix.")

    if n_dimensions_1 < 2:
        raise ValueError("Parameter `observed` has to have at least two dimensions.")

    if not isinstance(predicted, np.ndarray):
        raise ValueError("Parameter `predicted` has to be of type `numpy.ndarray`.")

    try:
        (n_predicted, n_dimensions_2) = np.shape(predicted)
    except:
        raise ValueError("Parameter `predicted` should be a matrix.")

    if n_dimensions_2 < 2:
        raise ValueError("Parameter `predicted` has to have at least two dimensions.")

    if n_observed != n_predicted:
        raise ValueError("Parameter `observed` has different number of elements than `predicted`.")

    if n_dimensions_1 != n_dimensions_2:
        raise ValueError("Parameter `observed` has different number of dimensions than `predicted`.")

    if not isinstance(tolerance, float):
        raise ValueError("Parameter `tolerance` has to be of type `float`.")

    good_direction = np.zeros((n_observed,))

    for i in range(0,n_observed):
        good_direction[i] = np.dot(observed[i,:]/np.linalg.norm(observed[i,:]), predicted[i,:]/np.linalg.norm(predicted[i,:]))

    (idx_good_direction, ) = np.where(good_direction >= 1.0 - tolerance)

    good_direction_estimate = len(idx_good_direction)/n_observed * 100.0

    return (good_direction, good_direction_estimate)

# ------------------------------------------------------------------------------

def generate_tex_table(data_frame_table, float_format='.2f', caption='', label=''):
    """
    Generates ``tex`` code for a table stored in a ``pandas.DataFrame``. This function
    can be useful e.g. for printing regression results.

    **Example:**

    .. code:: python

        from PCAfold import PCA, generate_tex_table
        import numpy as np
        import pandas as pd

        # Generate dummy data set:
        X = np.random.rand(100,5)

        # Generate dummy variables names:
        variable_names = ['A1', 'A2', 'A3', 'A4', 'A5']

        # Instantiate PCA class object:
        pca_q2 = PCA(X, scaling='auto', n_components=2, use_eigendec=True, nocenter=False)
        pca_q3 = PCA(X, scaling='auto', n_components=3, use_eigendec=True, nocenter=False)

        # Calculate the R2 values:
        r2_q2 = pca_q2.calculate_r2(X)[None,:]
        r2_q3 = pca_q3.calculate_r2(X)[None,:]

        # Generate pandas.DataFrame from the R2 values:
        r2_table = pd.DataFrame(np.vstack((r2_q2, r2_q3)), columns=variable_names, index=['PCA, $q=2$', 'PCA, $q=3$'])

        # Generate tex code for the table:
        generate_tex_table(r2_table, float_format=".3f", caption='$R^2$ values.', label='r2-values')

    .. note::

        The code above will produce ``tex`` code:

        .. code-block:: text

            \\begin{table}[h!]
            \\begin{center}
            \\begin{tabular}{llllll} \\toprule
             & \\textit{A1} & \\textit{A2} & \\textit{A3} & \\textit{A4} & \\textit{A5} \\\\ \\midrule
            PCA, $q=2$ & 0.507 & 0.461 & 0.485 & 0.437 & 0.611 \\\\
            PCA, $q=3$ & 0.618 & 0.658 & 0.916 & 0.439 & 0.778 \\\\
            \\end{tabular}
            \\caption{$R^2$ values.}\\label{r2-values}
            \\end{center}
            \\end{table}

        Which, when compiled, will result in a table:

        .. image:: ../images/generate-tex-table.png
            :width: 450
            :align: center

    :param data_frame_table:
        ``pandas.DataFrame`` specifying the table to convert to ``tex`` code. It can include column names and
        index names.
    :param float_format:
        ``str`` specifying the display format for the numerical entries inside the
        table. By default it is set to ``'.2f'``.
    :param caption:
        ``str`` specifying caption for the table.
    :param label:
        ``str`` specifying label for the table.
    """

    (n_rows, n_columns) = np.shape(data_frame_table)
    rows_labels = data_frame_table.index.values
    columns_labels = data_frame_table.columns.values

    print('')
    print(r'\begin{table}[h!]')
    print(r'\begin{center}')
    print(r'\begin{tabular}{' + ''.join(['l' for i in range(0, n_columns+1)]) + r'} \toprule')
    print(' & ' + ' & '.join([r'\textit{' + name + '}' for name in columns_labels]) + r' \\ \midrule')

    for row_i, row_label in enumerate(rows_labels):

        row_values = list(data_frame_table.iloc[row_i,:])
        print(row_label + r' & '+  ' & '.join([str(('%' + float_format) % value) for value in row_values]) + r' \\')

    print(r'\end{tabular}')
    print(r'\caption{' + caption + r'}\label{' + label + '}')
    print(r'\end{center}')
    print(r'\end{table}')
    print('')

################################################################################
#
# Plotting functions
#
################################################################################

def plot_2d_regression(x, observed, predicted, x_label=None, y_label=None, color_observed=None, color_predicted=None, figure_size=(7,7), title=None, save_filename=None):
    """
    Plots the result of regression of a dependent variable on top
    of a one-dimensional manifold defined by a single independent variable ``x``.

    **Example:**

    .. code:: python

        from PCAfold import PCA, plot_2d_regression
        import numpy as np

        # Generate dummy data set:
        X = np.random.rand(100,10)

        # Obtain two-dimensional manifold from PCA:
        pca_X = PCA(X)
        PCs = pca_X.transform(X)
        X_rec = pca_X.reconstruct(PCs)

        # Plot the manifold:
        plt = plot_2d_regression(X[:,0],
                                 X[:,0],
                                 X_rec[:,0],
                                 x_label='$x$',
                                 y_label='$y$',
                                 color_observed='k',
                                 color_predicted='r',
                                 figure_size=(10,10),
                                 title='2D regression',
                                 save_filename='2d-regression.pdf')
        plt.close()

    :param x:
        ``numpy.ndarray`` specifying the variable on the :math:`x`-axis. It should be of size ``(n_observations,)`` or ``(n_observations,1)``.
    :param observed:
        ``numpy.ndarray`` specifying the observed values of a single dependent variable.
        It should be of size ``(n_observations,)`` or ``(n_observations, 1)``.
    :param predicted:
        ``numpy.ndarray`` specifying the predicted values of a single dependent variable.
        It should be of size ``(n_observations,)`` or ``(n_observations, 1)``.
    :param x_label: (optional)
        ``str`` specifying :math:`x`-axis label annotation. If set to ``None``
        label will not be plotted.
    :param y_label: (optional)
        ``str`` specifying :math:`y`-axis label annotation. If set to ``None``
        label will not be plotted.
    :param color_observed: (optional)
        ``str`` specifying the color of the plotted observed variable.
    :param color_predicted: (optional)
        ``str`` specifying the color of the plotted predicted variable.
    :param figure_size: (optional)
        ``tuple`` specifying figure size.
    :param title: (optional)
        ``str`` specifying plot title. If set to ``None`` title will not be
        plotted.
    :param save_filename: (optional)
        ``str`` specifying plot save location/filename. If set to ``None``
        plot will not be saved. You can also set a desired file extension,
        for instance ``.pdf``. If the file extension is not specified, the default
        is ``.png``.

    :return:
        - **plt** - ``matplotlib.pyplot`` plot handle.
    """

    if not isinstance(x, np.ndarray):
        raise ValueError("Parameter `x` has to be of type `numpy.ndarray`.")

    try:
        (n_x,) = np.shape(x)
        n_var_x = 1
    except:
        (n_x, n_var_x) = np.shape(x)

    if n_var_x != 1:
        raise ValueError("Parameter `x` has to be a 0D or 1D vector.")

    if not isinstance(observed, np.ndarray):
        raise ValueError("Parameter `observed` has to be of type `numpy.ndarray`.")

    try:
        (n_observed,) = np.shape(observed)
        n_var_observed = 1
    except:
        (n_observed, n_var_observed) = np.shape(observed)

    if n_var_observed != 1:
        raise ValueError("Parameter `observed` has to be a 0D or 1D vector.")

    if not isinstance(predicted, np.ndarray):
        raise ValueError("Parameter `predicted` has to be of type `numpy.ndarray`.")

    try:
        (n_predicted,) = np.shape(predicted)
        n_var_predicted = 1
    except:
        (n_predicted, n_var_predicted) = np.shape(predicted)

    if n_var_predicted != 1:
        raise ValueError("Parameter `predicted` has to be a 0D or 1D vector.")

    if n_observed != n_predicted:
        raise ValueError("Parameter `observed` has different number of elements than `predicted`.")

    if n_x != n_observed:
        raise ValueError("Parameter `observed` has different number of elements than `x`.")

    if n_x != n_predicted:
        raise ValueError("Parameter `predicted` has different number of elements than `x`.")

    if x_label is not None:
        if not isinstance(x_label, str):
            raise ValueError("Parameter `x_label` has to be of type `str`.")

    if y_label is not None:
        if not isinstance(y_label, str):
            raise ValueError("Parameter `y_label` has to be of type `str`.")

    if color_observed is not None:
        if not isinstance(color_observed, str):
            raise ValueError("Parameter `color_observed` has to be of type `str`.")
    else:
        color_observed = '#191b27'

    if color_predicted is not None:
        if not isinstance(color_predicted, str):
            raise ValueError("Parameter `color_predicted` has to be of type `str`.")
    else:
        color_predicted = '#C7254E'

    if not isinstance(figure_size, tuple):
        raise ValueError("Parameter `figure_size` has to be of type `tuple`.")

    if title is not None:
        if not isinstance(title, str):
            raise ValueError("Parameter `title` has to be of type `str`.")

    if save_filename is not None:
        if not isinstance(save_filename, str):
            raise ValueError("Parameter `save_filename` has to be of type `str`.")

    fig = plt.figure(figsize=figure_size)

    scat = plt.scatter(x.ravel(), observed.ravel(), c=color_observed, marker='o', s=scatter_point_size, alpha=0.1)
    scat = plt.scatter(x.ravel(), predicted.ravel(), c=color_predicted, marker='o', s=scatter_point_size, alpha=0.4)

    if x_label != None: plt.xlabel(x_label, **csfont, fontsize=font_labels)
    if y_label != None: plt.ylabel(y_label, **csfont, fontsize=font_labels)
    plt.xticks(fontsize=font_axes, **csfont)
    plt.yticks(fontsize=font_axes, **csfont)
    plt.grid(alpha=grid_opacity)
    lgnd = plt.legend(['Observed', 'Predicted'], fontsize=font_legend, loc="best")
    lgnd.legendHandles[0]._sizes = [marker_size*5]
    lgnd.legendHandles[1]._sizes = [marker_size*5]

    if title != None: plt.title(title, **csfont, fontsize=font_title)
    if save_filename != None: plt.savefig(save_filename, dpi=save_dpi, bbox_inches='tight')

    return plt

# ------------------------------------------------------------------------------

def plot_2d_regression_scalar_field(grid_bounds, regression_model, x=None, y=None, resolution=(10,10), extension=(0,0), x_label=None, y_label=None, s_field=None, s_manifold=None, manifold_color=None, colorbar_label=None, color_map='viridis', colorbar_range=None, manifold_alpha=1, grid_on=True, figure_size=(7,7), title=None, save_filename=None):
    """
    Plots a 2D field of a regressed scalar dependent variable.
    A two-dimensional manifold can be additionally plotted on top of the field.

    **Example:**

    .. code:: python

        from PCAfold import PCA, KReg, plot_2d_regression_scalar_field
        import numpy as np

        # Generate dummy data set:
        X = np.random.rand(100,2)
        Z = np.random.rand(100,1)

        # Train the kernel regression model:
        model = KReg(X, Z)

        # Define the regression model:
        def regression_model(query):

            predicted = model.predict(query, 'nearest_neighbors_isotropic', n_neighbors=1)[:,0]

            return predicted

        # Define the bounds for the scalar field:
        grid_bounds = ([np.min(X[:,0]),np.max(X[:,0])],[np.min(X[:,1]),np.max(X[:,1])])

        # Plot the regressed scalar field:
        plt = plot_2d_regression_scalar_field(grid_bounds,
                                            regression_model,
                                            x=X[:,0],
                                            y=X[:,1],
                                            resolution=(100,100),
                                            extension=(10,10),
                                            x_label='$X_1$',
                                            y_label='$X_2$',
                                            s_field=4,
                                            s_manifold=60,
                                            manifold_color=Z,
                                            colorbar_label='$Z_1$',
                                            color_map='inferno',
                                            colorbar_range=(0,1),
                                            manifold_alpha=1,
                                            grid_on=False,
                                            figure_size=(10,6),
                                            title='2D regressed scalar field',
                                            save_filename='2D-regressed-scalar-field.pdf')
        plt.close()

    :param grid_bounds:
        ``tuple`` of ``list`` specifying the bounds of the dependent variable on the :math:`x` and :math:`y` axis.
    :param regression_model:
        ``function`` that outputs the predicted vector using the regression model.
        It should take as input a ``numpy.ndarray`` of size ``(1,2)``, where the two
        elements specify the first and second independent variable values. It should output
        a ``float`` specifying the regressed scalar value at that input.
    :param x: (optional)
        ``numpy.ndarray`` specifying the variable on the :math:`x`-axis.
        It should be of size ``(n_observations,)`` or ``(n_observations,1)``.
        It can be used to plot a 2D manifold on top of the streamplot.
    :param y: (optional)
        ``numpy.ndarray`` specifying the variable on the :math:`y`-axis.
        It should be of size ``(n_observations,)`` or ``(n_observations,1)``.
        It can be used to plot a 2D manifold on top of the streamplot.
    :param resolution: (optional)
        ``tuple`` of ``int`` specifying the resolution of the streamplot grid on the :math:`x` and :math:`y` axis.
    :param extension: (optional)
        ``tuple`` of ``float`` or ``int`` specifying a percentage by which the
        dependent variable should be extended beyond on the :math:`x` and :math:`y` axis, beyond what has been specified by the ``grid_bounds`` parameter.
    :param x_label: (optional)
        ``str`` specifying :math:`x`-axis label annotation. If set to ``None``
        label will not be plotted.
    :param y_label: (optional)
        ``str`` specifying :math:`y`-axis label annotation. If set to ``None``
        label will not be plotted.
    :param s_field: (optional)
        ``int`` or ``float`` specifying the scatter point size for the scalar field.
    :param s_manifold: (optional)
        ``int`` or ``float`` specifying the scatter point size for the manifold.
    :param manifold_color: (optional)
        vector or string specifying color for the manifold. If it is a
        vector, it has to have length consistent with the number of observations
        in ``x`` and ``y`` vectors. It should be of type ``numpy.ndarray`` and size
        ``(n_observations,)`` or ``(n_observations,1)``.
        It can also be set to a string specifying the color directly, for
        instance ``'r'`` or ``'#006778'``.
        If not specified, manifold will be plotted in black.
    :param colorbar_label: (optional)
        ``str`` specifying colorbar label annotation.
    :param color_map: (optional)
        ``str`` or ``matplotlib.colors.ListedColormap`` specifying the colormap to use as per ``matplotlib.cm``. Default is ``'viridis'``.
    :param colorbar_range: (optional)
        ``tuple`` specifying the lower and the upper bound for the colorbar range.
    :param manifold_alpha: (optional)
        ``float`` or ``int`` specifying the opacity of the plotted manifold.
    :param grid_on:
        ``bool`` specifying whether grid should be plotted.
    :param figure_size: (optional)
        ``tuple`` specifying figure size.
    :param title: (optional)
        ``str`` specifying plot title. If set to ``None`` title will not be
        plotted.
    :param save_filename: (optional)
        ``str`` specifying plot save location/filename. If set to ``None``
        plot will not be saved. You can also set a desired file extension,
        for instance ``.pdf``. If the file extension is not specified, the default
        is ``.png``.

    :return:
        - **plt** - ``matplotlib.pyplot`` plot handle.
    """

    if not isinstance(grid_bounds, tuple):
        raise ValueError("Parameter `grid_bounds` has to be of type `tuple`.")

    if not callable(regression_model):
        raise ValueError("Parameter `regression_model` has to be of type `function`.")

    if x is not None:

        if not isinstance(x, np.ndarray):
            raise ValueError("Parameter `x` has to be of type `numpy.ndarray`.")

        try:
            (n_x,) = np.shape(x)
            n_var_x = 1
        except:
            (n_x, n_var_x) = np.shape(x)

        if n_var_x != 1:
            raise ValueError("Parameter `x` has to be a 0D or 1D vector.")

    if y is not None:

        if not isinstance(y, np.ndarray):
            raise ValueError("Parameter `y` has to be of type `numpy.ndarray`.")

        try:
            (n_y,) = np.shape(y)
            n_var_y = 1
        except:
            (n_y, n_var_y) = np.shape(y)

        if n_var_y != 1:
            raise ValueError("Parameter `y` has to be a 0D or 1D vector.")

        if n_x != n_y:
            raise ValueError("Parameter `x` has different number of elements than `y`.")

    if not isinstance(resolution, tuple):
        raise ValueError("Parameter `resolution` has to be of type `tuple`.")

    if not isinstance(extension, tuple):
        raise ValueError("Parameter `extension` has to be of type `tuple`.")

    if x_label is not None:
        if not isinstance(x_label, str):
            raise ValueError("Parameter `x_label` has to be of type `str`.")

    if y_label is not None:
        if not isinstance(y_label, str):
            raise ValueError("Parameter `y_label` has to be of type `str`.")

    if s_field is None:
        s_field = scatter_point_size
    else:
        if not isinstance(s_field, int) and not isinstance(s_field, float):
            raise ValueError("Parameter `s_field` has to be of type `int` or `float`.")

    if s_manifold is None:
        s_manifold = scatter_point_size
    else:
        if not isinstance(s_manifold, int) and not isinstance(s_manifold, float):
            raise ValueError("Parameter `s_manifold` has to be of type `int` or `float`.")

    if manifold_color is not None:
        if not isinstance(manifold_color, str):
            if not isinstance(manifold_color, np.ndarray):
                raise ValueError("Parameter `manifold_color` has to be `None`, or of type `str` or `numpy.ndarray`.")

    if isinstance(manifold_color, np.ndarray):

        try:
            (n_color,) = np.shape(manifold_color)
            n_var_color = 1
        except:
            (n_color, n_var_color) = np.shape(manifold_color)

        if n_var_color != 1:
            raise ValueError("Parameter `manifold_color` has to be a 0D or 1D vector.")

        if n_color != n_x:
            raise ValueError("Parameter `manifold_color` has different number of elements than `x` and `y`.")

    if colorbar_label is not None:
        if not isinstance(colorbar_label, str):
            raise ValueError("Parameter `colorbar_label` has to be of type `str`.")

    if not isinstance(color_map, str):
        if not isinstance(color_map, ListedColormap):
            raise ValueError("Parameter `color_map` has to be of type `str` or `matplotlib.colors.ListedColormap`.")

    if colorbar_range is not None:
        if not isinstance(colorbar_range, tuple):
            raise ValueError("Parameter `colorbar_range` has to be of type `tuple`.")
        else:
            (cbar_min, cbar_max) = colorbar_range

    if manifold_alpha is not None:
        if not isinstance(manifold_alpha, float) and not isinstance(manifold_alpha, int):
            raise ValueError("Parameter `manifold_alpha` has to be of type `float`.")

    if not isinstance(grid_on, bool):
        raise ValueError("Parameter `grid_on` has to be of type `bool`.")

    if not isinstance(figure_size, tuple):
        raise ValueError("Parameter `figure_size` has to be of type `tuple`.")

    if title is not None:
        if not isinstance(title, str):
            raise ValueError("Parameter `title` has to be of type `str`.")

    if save_filename is not None:
        if not isinstance(save_filename, str):
            raise ValueError("Parameter `save_filename` has to be of type `str`.")

    (x_extend, y_extend) = extension
    (x_resolution, y_resolution) = resolution
    ([x_minimum, x_maximum], [y_minimum, y_maximum]) = grid_bounds

    # Create extension in both dimensions:
    x_extension = x_extend/100.0 * abs(x_minimum - x_maximum)
    y_extension = y_extend/100.0 * abs(y_minimum - y_maximum)

    # Create grid points for the independent variables where regression will be applied:
    x_grid = np.linspace(x_minimum-x_extension, x_maximum+x_extension, x_resolution)
    y_grid = np.linspace(y_minimum-y_extension, y_maximum+y_extension, y_resolution)
    xy_mesh = np.meshgrid(x_grid, y_grid, indexing='xy')

    # Evaluate the predicted scalar using the regression model:
    regressed_scalar = np.zeros((y_grid.size*x_grid.size,))

    for i in range(0,y_grid.size*x_grid.size):

                regression_input = np.reshape(np.array([xy_mesh[0].ravel()[i], xy_mesh[1].ravel()[i]]), [1,2])
                regressed_scalar[i] = regression_model(regression_input)

    fig = plt.figure(figsize=figure_size)

    if colorbar_range is not None:
        scat_field = plt.scatter(xy_mesh[0].ravel(), xy_mesh[1].ravel(), c=regressed_scalar, cmap=color_map, s=s_field, vmin=cbar_min, vmax=cbar_max)
    else:
        scat_field = plt.scatter(xy_mesh[0].ravel(), xy_mesh[1].ravel(), c=regressed_scalar, cmap=color_map, s=s_field)

    if (x is not None) and (y is not None):

        if manifold_color is None:
            scat = plt.scatter(x.ravel(), y.ravel(), c='k', marker='o', s=s_manifold, edgecolor='none', alpha=manifold_alpha)
        elif isinstance(manifold_color, str):
            scat = plt.scatter(x.ravel(), y.ravel(), c=manifold_color, cmap=color_map, marker='o', s=s_manifold, edgecolor='none', alpha=manifold_alpha)
        elif isinstance(manifold_color, np.ndarray):
            if colorbar_range is not None:
                scat = plt.scatter(x.ravel(), y.ravel(), c=manifold_color.ravel(), cmap=color_map, marker='o', s=s_manifold, edgecolor='none', vmin=cbar_min, vmax=cbar_max, alpha=manifold_alpha)
            else:
                scat = plt.scatter(x.ravel(), y.ravel(), c=manifold_color.ravel(), cmap=color_map, marker='o', s=s_manifold, edgecolor='none', vmin=np.min(regressed_scalar), vmax=np.max(regressed_scalar), alpha=manifold_alpha)

    cb = fig.colorbar(scat_field)
    cb.ax.tick_params(labelsize=font_colorbar_axes)
    if colorbar_label is not None: cb.set_label(colorbar_label, fontsize=font_colorbar, rotation=0, horizontalalignment='left')
    if colorbar_range is not None: plt.clim(cbar_min, cbar_max)

    if x_label is not None: plt.xlabel(x_label, **csfont, fontsize=font_labels)
    if y_label is not None: plt.ylabel(y_label, **csfont, fontsize=font_labels)
    plt.xlim([np.min(x_grid),np.max(x_grid)])
    plt.ylim([np.min(y_grid),np.max(y_grid)])
    plt.xticks(fontsize=font_axes, **csfont)
    plt.yticks(fontsize=font_axes, **csfont)
    if grid_on: plt.grid(alpha=grid_opacity)

    if title is not None: plt.title(title, **csfont, fontsize=font_title)
    if save_filename is not None: plt.savefig(save_filename, dpi=save_dpi, bbox_inches='tight')

    return plt

# ------------------------------------------------------------------------------

def plot_2d_regression_streamplot(grid_bounds, regression_model, x=None, y=None, resolution=(10,10), extension=(0,0), color='k', x_label=None, y_label=None, manifold_color=None, colorbar_label=None, color_map='viridis', colorbar_range=None, manifold_alpha=1, grid_on=True, figure_size=(7,7), title=None, save_filename=None):
    """
    Plots a streamplot of a regressed vector field of a dependent variable.
    A two-dimensional manifold can be additionally plotted on top of the streamplot.

    **Example:**

    .. code:: python

        from PCAfold import PCA, KReg, plot_2d_regression_streamplot
        import numpy as np

        # Generate dummy data set:
        X = np.random.rand(100,5)
        S_X = np.random.rand(100,5)

        # Obtain two-dimensional manifold from PCA:
        pca_X = PCA(X, n_components=2)
        PCs = pca_X.transform(X)
        S_Z = pca_X.transform(S_X, nocenter=True)

        # Train the kernel regression model:
        model = KReg(PCs, S_Z)

        # Define the regression model:
        def regression_model(query):

            predicted = model.predict(query, 'nearest_neighbors_isotropic', n_neighbors=1)

            return predicted

        # Define the bounds for the streamplot:
        grid_bounds = ([np.min(PCs[:,0]),np.max(PCs[:,0])],[np.min(PCs[:,1]),np.max(PCs[:,1])])

        # Plot the regression streamplot:
        plt = plot_2d_regression_streamplot(grid_bounds,
                                            regression_model,
                                            x=PCs[:,0],
                                            y=PCs[:,1],
                                            resolution=(15,15),
                                            extension=(20,20),
                                            color='r',
                                            x_label='$Z_1$',
                                            y_label='$Z_2$',
                                            manifold_color=X[:,0],
                                            colorbar_label='$X_1$',
                                            color_map='plasma',
                                            colorbar_range=(0,1),
                                            manifold_alpha=1,
                                            grid_on=False,
                                            figure_size=(10,6),
                                            title='Streamplot',
                                            save_filename='streamplot.pdf')
        plt.close()

    :param grid_bounds:
        ``tuple`` of ``list`` specifying the bounds of the dependent variable on the :math:`x` and :math:`y` axis.
    :param regression_model:
        ``function`` that outputs the predicted vector using the regression model.
        It should take as input a ``numpy.ndarray`` of size ``(1,2)``, where the two
        elements specify the first and second independent variable values. It should output
        a ``numpy.ndarray`` of size ``(1,2)``, where the two elements specify
        the first and second regressed vector elements.
    :param x: (optional)
        ``numpy.ndarray`` specifying the variable on the :math:`x`-axis.
        It should be of size ``(n_observations,)`` or ``(n_observations,1)``.
        It can be used to plot a 2D manifold on top of the streamplot.
    :param y: (optional)
        ``numpy.ndarray`` specifying the variable on the :math:`y`-axis.
        It should be of size ``(n_observations,)`` or ``(n_observations,1)``.
        It can be used to plot a 2D manifold on top of the streamplot.
    :param resolution: (optional)
        ``tuple`` of ``int`` specifying the resolution of the streamplot grid on the :math:`x` and :math:`y` axis.
    :param extension: (optional)
        ``tuple`` of ``float`` or ``int`` specifying a percentage by which the
        dependent variable should be extended beyond on the :math:`x` and :math:`y` axis, beyond what has been specified by the ``grid_bounds`` parameter.
    :param color: (optional)
        ``str`` specifying the streamlines color.
    :param x_label: (optional)
        ``str`` specifying :math:`x`-axis label annotation. If set to ``None``
        label will not be plotted.
    :param y_label: (optional)
        ``str`` specifying :math:`y`-axis label annotation. If set to ``None``
        label will not be plotted.
    :param manifold_color: (optional)
        vector or string specifying color for the manifold. If it is a
        vector, it has to have length consistent with the number of observations
        in ``x`` and ``y`` vectors. It should be of type ``numpy.ndarray`` and size
        ``(n_observations,)`` or ``(n_observations,1)``.
        It can also be set to a string specifying the color directly, for
        instance ``'r'`` or ``'#006778'``.
        If not specified, manifold will be plotted in black.
    :param colorbar_label: (optional)
        ``str`` specifying colorbar label annotation.
    :param color_map: (optional)
        ``str`` or ``matplotlib.colors.ListedColormap`` specifying the colormap to use as per ``matplotlib.cm``. Default is ``'viridis'``.
    :param colorbar_range: (optional)
        ``tuple`` specifying the lower and the upper bound for the colorbar range.
    :param manifold_alpha: (optional)
        ``float`` or ``int`` specifying the opacity of the plotted manifold.
    :param grid_on:
        ``bool`` specifying whether grid should be plotted.
    :param figure_size: (optional)
        ``tuple`` specifying figure size.
    :param title: (optional)
        ``str`` specifying plot title. If set to ``None`` title will not be
        plotted.
    :param save_filename: (optional)
        ``str`` specifying plot save location/filename. If set to ``None``
        plot will not be saved. You can also set a desired file extension,
        for instance ``.pdf``. If the file extension is not specified, the default
        is ``.png``.

    :return:
        - **plt** - ``matplotlib.pyplot`` plot handle.
    """

    if not isinstance(grid_bounds, tuple):
        raise ValueError("Parameter `grid_bounds` has to be of type `tuple`.")

    if not callable(regression_model):
        raise ValueError("Parameter `regression_model` has to be of type `function`.")

    if x is not None:

        if not isinstance(x, np.ndarray):
            raise ValueError("Parameter `x` has to be of type `numpy.ndarray`.")

        try:
            (n_x,) = np.shape(x)
            n_var_x = 1
        except:
            (n_x, n_var_x) = np.shape(x)

        if n_var_x != 1:
            raise ValueError("Parameter `x` has to be a 0D or 1D vector.")

    if y is not None:

        if not isinstance(y, np.ndarray):
            raise ValueError("Parameter `y` has to be of type `numpy.ndarray`.")

        try:
            (n_y,) = np.shape(y)
            n_var_y = 1
        except:
            (n_y, n_var_y) = np.shape(y)

        if n_var_y != 1:
            raise ValueError("Parameter `y` has to be a 0D or 1D vector.")

        if n_x != n_y:
            raise ValueError("Parameter `x` has different number of elements than `y`.")

    if not isinstance(resolution, tuple):
        raise ValueError("Parameter `resolution` has to be of type `tuple`.")

    if not isinstance(extension, tuple):
        raise ValueError("Parameter `extension` has to be of type `tuple`.")

    if color is not None:
        if not isinstance(color, str):
            raise ValueError("Parameter `color` has to be of type `str`.")

    if x_label is not None:
        if not isinstance(x_label, str):
            raise ValueError("Parameter `x_label` has to be of type `str`.")

    if y_label is not None:
        if not isinstance(y_label, str):
            raise ValueError("Parameter `y_label` has to be of type `str`.")

    if manifold_color is not None:
        if not isinstance(manifold_color, str):
            if not isinstance(manifold_color, np.ndarray):
                raise ValueError("Parameter `manifold_color` has to be `None`, or of type `str` or `numpy.ndarray`.")

    if isinstance(manifold_color, np.ndarray):

        try:
            (n_color,) = np.shape(manifold_color)
            n_var_color = 1
        except:
            (n_color, n_var_color) = np.shape(manifold_color)

        if n_var_color != 1:
            raise ValueError("Parameter `manifold_color` has to be a 0D or 1D vector.")

        if n_color != n_x:
            raise ValueError("Parameter `manifold_color` has different number of elements than `x` and `y`.")

    if colorbar_label is not None:
        if not isinstance(colorbar_label, str):
            raise ValueError("Parameter `colorbar_label` has to be of type `str`.")

    if not isinstance(color_map, str):
        if not isinstance(color_map, ListedColormap):
            raise ValueError("Parameter `color_map` has to be of type `str` or `matplotlib.colors.ListedColormap`.")

    if colorbar_range is not None:
        if not isinstance(colorbar_range, tuple):
            raise ValueError("Parameter `colorbar_range` has to be of type `tuple`.")
        else:
            (cbar_min, cbar_max) = colorbar_range

    if manifold_alpha is not None:
        if not isinstance(manifold_alpha, float) and not isinstance(manifold_alpha, int):
            raise ValueError("Parameter `manifold_alpha` has to be of type `float`.")

    if not isinstance(grid_on, bool):
        raise ValueError("Parameter `grid_on` has to be of type `bool`.")

    if not isinstance(figure_size, tuple):
        raise ValueError("Parameter `figure_size` has to be of type `tuple`.")

    if title is not None:
        if not isinstance(title, str):
            raise ValueError("Parameter `title` has to be of type `str`.")

    if save_filename is not None:
        if not isinstance(save_filename, str):
            raise ValueError("Parameter `save_filename` has to be of type `str`.")

    (x_extend, y_extend) = extension
    (x_resolution, y_resolution) = resolution
    ([x_minimum, x_maximum], [y_minimum, y_maximum]) = grid_bounds

    # Create extension in both dimensions:
    x_extension = x_extend/100.0 * abs(x_minimum - x_maximum)
    y_extension = y_extend/100.0 * abs(y_minimum - y_maximum)

    # Create grid points for the independent variables where regression will be applied:
    x_grid = np.linspace(x_minimum-x_extension, x_maximum+x_extension, x_resolution)
    y_grid = np.linspace(y_minimum-y_extension, y_maximum+y_extension, y_resolution)
    xy_mesh = np.meshgrid(x_grid, y_grid, indexing='xy')

    # Evaluate the predicted vectors using the regression model:
    x_vector = np.zeros((y_grid.size, x_grid.size))
    y_vector = np.zeros((y_grid.size, x_grid.size))

    for j, x_variable in enumerate(x_grid):
        for i, y_variable in enumerate(y_grid):

                regression_input = np.reshape(np.array([x_variable, y_variable]), [1,2])

                regressed_vector = regression_model(regression_input)

                x_vector[i,j] = regressed_vector[0,0]
                y_vector[i,j] = regressed_vector[0,1]

    fig = plt.figure(figsize=figure_size)

    if (x is not None) and (y is not None):

        if manifold_color is None:
            scat = plt.scatter(x.ravel(), y.ravel(), c='k', marker='o', s=scatter_point_size, edgecolor='none', alpha=manifold_alpha)
        elif isinstance(manifold_color, str):
            scat = plt.scatter(x.ravel(), y.ravel(), c=manifold_color, cmap=color_map, marker='o', s=scatter_point_size, edgecolor='none', alpha=manifold_alpha)
        elif isinstance(manifold_color, np.ndarray):
            scat = plt.scatter(x.ravel(), y.ravel(), c=manifold_color.ravel(), cmap=color_map, marker='o', s=scatter_point_size, edgecolor='none', alpha=manifold_alpha)

    if isinstance(manifold_color, np.ndarray):
        if manifold_color is not None:
            cb = fig.colorbar(scat)
            cb.ax.tick_params(labelsize=font_colorbar_axes)
            if colorbar_label is not None: cb.set_label(colorbar_label, fontsize=font_colorbar, rotation=0, horizontalalignment='left')
            if colorbar_range is not None: plt.clim(cbar_min, cbar_max)

    plt.streamplot(xy_mesh[0], xy_mesh[1], x_vector, y_vector, color=color, density=3, linewidth=1, arrowsize=1)

    if x_label is not None: plt.xlabel(x_label, **csfont, fontsize=font_labels)
    if y_label is not None: plt.ylabel(y_label, **csfont, fontsize=font_labels)
    plt.xlim([np.min(x_grid),np.max(x_grid)])
    plt.ylim([np.min(y_grid),np.max(y_grid)])
    plt.xticks(fontsize=font_axes, **csfont)
    plt.yticks(fontsize=font_axes, **csfont)
    if grid_on: plt.grid(alpha=grid_opacity)

    if title is not None: plt.title(title, **csfont, fontsize=font_title)
    if save_filename is not None: plt.savefig(save_filename, dpi=save_dpi, bbox_inches='tight')

    return plt

# ------------------------------------------------------------------------------

def plot_3d_regression(x, y, observed, predicted, elev=45, azim=-45, x_label=None, y_label=None, z_label=None, color_observed=None, color_predicted=None, figure_size=(7,7), title=None, save_filename=None):
    """
    Plots the result of regression of a dependent variable on top
    of a two-dimensional manifold defined by two independent variables ``x`` and ``y``.

    **Example:**

    .. code:: python

        from PCAfold import PCA, plot_3d_regression
        import numpy as np

        # Generate dummy data set:
        X = np.random.rand(100,10)

        # Obtain three-dimensional manifold from PCA:
        pca_X = PCA(X)
        PCs = pca_X.transform(X)
        X_rec = pca_X.reconstruct(PCs)

        # Plot the manifold:
        plt = plot_3d_regression(X[:,0],
                                 X[:,1],
                                 X[:,0],
                                 X_rec[:,0],
                                 elev=45,
                                 azim=-45,
                                 x_label='$x$',
                                 y_label='$y$',
                                 z_label='$z$',
                                 color_observed='k',
                                 color_predicted='r',
                                 figure_size=(10,10),
                                 title='3D regression',
                                 save_filename='3d-regression.pdf')
        plt.close()

    :param x:
        ``numpy.ndarray`` specifying the variable on the :math:`x`-axis. It should be of size ``(n_observations,)`` or ``(n_observations,1)``.
    :param y:
        ``numpy.ndarray`` specifying the variable on the :math:`y`-axis. It should be of size ``(n_observations,)`` or ``(n_observations,1)``.
    :param observed:
        ``numpy.ndarray`` specifying the observed values of a single dependent variable.
        It should be of size ``(n_observations,)`` or ``(n_observations, 1)``.
    :param predicted:
        ``numpy.ndarray`` specifying the predicted values of a single dependent variable.
        It should be of size ``(n_observations,)`` or ``(n_observations, 1)``.
    :param elev: (optional)
        ``float`` or ``int`` specifying the elevation angle.
    :param azim: (optional)
        ``float`` or ``int`` specifying the azimuth angle.
    :param x_label: (optional)
        ``str`` specifying :math:`x`-axis label annotation. If set to ``None``
        label will not be plotted.
    :param y_label: (optional)
        ``str`` specifying :math:`y`-axis label annotation. If set to ``None``
        label will not be plotted.
    :param z_label: (optional)
        ``str`` specifying :math:`z`-axis label annotation. If set to ``None``
        label will not be plotted.
    :param color_observed: (optional)
        ``str`` specifying the color of the plotted observed variable.
    :param color_predicted: (optional)
        ``str`` specifying the color of the plotted predicted variable.
    :param figure_size: (optional)
        ``tuple`` specifying figure size.
    :param title: (optional)
        ``str`` specifying plot title. If set to ``None`` title will not be
        plotted.
    :param save_filename: (optional)
        ``str`` specifying plot save location/filename. If set to ``None``
        plot will not be saved. You can also set a desired file extension,
        for instance ``.pdf``. If the file extension is not specified, the default
        is ``.png``.

    :return:
        - **plt** - ``matplotlib.pyplot`` plot handle.
    """

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

    if not isinstance(observed, np.ndarray):
        raise ValueError("Parameter `observed` has to be of type `numpy.ndarray`.")

    try:
        (n_observed,) = np.shape(observed)
        n_var_observed = 1
    except:
        (n_observed, n_var_observed) = np.shape(observed)

    if n_var_observed != 1:
        raise ValueError("Parameter `observed` has to be a 0D or 1D vector.")

    if not isinstance(predicted, np.ndarray):
        raise ValueError("Parameter `predicted` has to be of type `numpy.ndarray`.")

    try:
        (n_predicted,) = np.shape(predicted)
        n_var_predicted = 1
    except:
        (n_predicted, n_var_predicted) = np.shape(predicted)

    if n_var_predicted != 1:
        raise ValueError("Parameter `predicted` has to be a 0D or 1D vector.")

    if n_observed != n_predicted:
        raise ValueError("Parameter `observed` has different number of elements than `predicted`.")

    if n_x != n_observed:
        raise ValueError("Parameter `observed` has different number of elements than `x`, `y` and `z`.")

    if n_x != n_predicted:
        raise ValueError("Parameter `predicted` has different number of elements than `x`, `y` and `z`.")

    if not isinstance(elev, float) and not isinstance(elev, int):
        raise ValueError("Parameter `elev` has to be of type `int` or `float`.")

    if not isinstance(azim, float) and not isinstance(azim, int):
        raise ValueError("Parameter `azim` has to be of type `int` or `float`.")

    if x_label is not None:
        if not isinstance(x_label, str):
            raise ValueError("Parameter `x_label` has to be of type `str`.")

    if y_label is not None:
        if not isinstance(y_label, str):
            raise ValueError("Parameter `y_label` has to be of type `str`.")

    if z_label is not None:
        if not isinstance(z_label, str):
            raise ValueError("Parameter `z_label` has to be of type `str`.")

    if color_observed is not None:
        if not isinstance(color_observed, str):
            raise ValueError("Parameter `color_observed` has to be of type `str`.")
    else:
        color_observed = '#191b27'

    if color_predicted is not None:
        if not isinstance(color_predicted, str):
            raise ValueError("Parameter `color_predicted` has to be of type `str`.")
    else:
        color_predicted = '#C7254E'

    if not isinstance(figure_size, tuple):
        raise ValueError("Parameter `figure_size` has to be of type `tuple`.")

    if title is not None:
        if not isinstance(title, str):
            raise ValueError("Parameter `title` has to be of type `str`.")

    if save_filename is not None:
        if not isinstance(save_filename, str):
            raise ValueError("Parameter `save_filename` has to be of type `str`.")

    fig = plt.figure(figsize=figure_size)
    ax = fig.add_subplot(111, projection='3d')

    scat = ax.scatter(x.ravel(), y.ravel(), observed.ravel(), c=color_observed, marker='o', s=scatter_point_size, alpha=0.1)
    scat = ax.scatter(x.ravel(), y.ravel(), predicted.ravel(), c=color_predicted, marker='o', s=scatter_point_size, alpha=0.4)

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

    lgnd = plt.legend(['Observed', 'Predicted'], fontsize=font_legend, bbox_to_anchor=(0.9,0.9), loc="upper left")
    lgnd.legendHandles[0]._sizes = [marker_size*5]
    lgnd.legendHandles[1]._sizes = [marker_size*5]

    if title != None: ax.set_title(title, **csfont, fontsize=font_title)
    if save_filename != None: plt.savefig(save_filename, dpi=save_dpi, bbox_inches='tight')

    return plt

# ------------------------------------------------------------------------------

def plot_normalized_variance(variance_data, plot_variables=[], color_map='Blues', figure_size=(10,5), title=None, save_filename=None):
    """
    This function plots normalized variance :math:`\mathcal{N}(\sigma)` over
    bandwith values :math:`\sigma` from an object of a ``VarianceData`` class.

    *Note:* this function can accomodate plotting up to 18 variables at once.
    You can specify which variables should be plotted using ``plot_variables`` list.

    **Example:**

    .. code:: python

        from PCAfold import PCA, compute_normalized_variance, plot_normalized_variance
        import numpy as np

        # Generate dummy data set:
        X = np.random.rand(100,5)

        # Perform PCA to obtain the low-dimensional manifold:
        pca_X = PCA(X, n_components=2)
        principal_components = pca_X.transform(X)

        # Compute normalized variance quantities:
        variance_data = compute_normalized_variance(principal_components, X, depvar_names=['A', 'B', 'C', 'D', 'E'], bandwidth_values=np.logspace(-3, 1, 20), scale_unit_box=True)

        # Plot normalized variance quantities:
        plt = plot_normalized_variance(variance_data,
                                       plot_variables=[0,1,2],
                                       color_map='Blues',
                                       figure_size=(10,5),
                                       title='Normalized variance',
                                       save_filename='N.pdf')
        plt.close()

    :param variance_data:
        an object of ``VarianceData`` class objects whose normalized variance quantities
        should be plotted.
    :param plot_variables: (optional)
        ``list`` of ``int`` specifying indices of variables to be plotted.
        By default, all variables are plotted.
    :param color_map: (optional)
        ``str`` or ``matplotlib.colors.ListedColormap`` specifying the colormap to use as per ``matplotlib.cm``. Default is ``'Blues'``.
    :param figure_size: (optional)
        ``tuple`` specifying figure size.
    :param title: (optional)
        ``str`` specifying plot title. If set to ``None`` title will not be
        plotted.
    :param save_filename: (optional)
        ``str`` specifying plot save location/filename. If set to ``None``
        plot will not be saved. You can also set a desired file extension,
        for instance ``.pdf``. If the file extension is not specified, the default
        is ``.png``.

    :return:
        - **plt** - ``matplotlib.pyplot`` plot handle.
    """

    if not isinstance(figure_size, tuple):
        raise ValueError("Parameter `figure_size` has to be of type `tuple`.")

    if title is not None:
        if not isinstance(title, str):
            raise ValueError("Parameter `title` has to be of type `str`.")

    if save_filename is not None:
        if not isinstance(save_filename, str):
            raise ValueError("Parameter `save_filename` has to be of type `str`.")

    from matplotlib import cm
    color_map_colors = cm.get_cmap(color_map)

    markers_list = ["o-","v-","^-","<-",">-","s-","p-","P-","*-","h-","H-","+-","x-","X-","D-","d-","|-","_-"]

    # Extract quantities from the VarianceData class object:
    variable_names = variance_data.variable_names
    bandwidth_values = variance_data.bandwidth_values
    normalized_variance = variance_data.normalized_variance

    if len(plot_variables) != 0:
        variables_to_plot = []
        for i in plot_variables:
            variables_to_plot.append(variable_names[i])
    else:
        variables_to_plot = variable_names

    n_variables = len(variables_to_plot)

    if n_variables > 18:
        raise ValueError("Only 18 variables can be plotted at once. Consider pre-selecting the variables to plot using `plot_variables` parameter.")

    if n_variables == 1:
        variable_colors = np.flipud(color_map_colors([0.8]))
    else:
        variable_colors = np.flipud(color_map_colors(np.linspace(0.2, 0.8, n_variables)))

    figure = plt.figure(figsize=figure_size)

    # Plot the normalized variance:
    for i, variable_name in enumerate(variables_to_plot):
        plt.semilogx(bandwidth_values, normalized_variance[variable_name], markers_list[i], label=variable_name, color=variable_colors[i])

    plt.xlabel('$\sigma$', fontsize=font_labels, **csfont)
    plt.ylabel('$N(\sigma)$', fontsize=font_labels, **csfont)
    plt.grid(alpha=grid_opacity)

    if n_variables <=5:
        plt.legend(loc='best', fancybox=True, shadow=True, ncol=1, fontsize=font_legend, markerscale=marker_scale_legend)
    else:
        plt.legend(bbox_to_anchor=(1.05,1), fancybox=True, shadow=True, ncol=2, fontsize=font_legend, markerscale=marker_scale_legend)

    if title != None: plt.title(title, fontsize=font_title, **csfont)
    if save_filename != None: plt.savefig(save_filename, dpi=save_dpi, bbox_inches='tight')

    return plt

# ------------------------------------------------------------------------------

def plot_normalized_variance_comparison(variance_data_tuple, plot_variables_tuple, color_map_tuple, figure_size=(10,5), title=None, save_filename=None):
    """
    This function plots a comparison of normalized variance :math:`\mathcal{N}(\sigma)` over
    bandwith values :math:`\sigma` from several objects of a ``VarianceData`` class.

    *Note:* this function can accomodate plotting up to 18 variables at once.
    You can specify which variables should be plotted using ``plot_variables`` list.

    **Example:**

    .. code:: python

        from PCAfold import PCA, compute_normalized_variance, plot_normalized_variance_comparison
        import numpy as np

        # Generate dummy data sets:
        X = np.random.rand(100,5)
        Y = np.random.rand(100,5)

        # Perform PCA to obtain low-dimensional manifolds:
        pca_X = PCA(X, n_components=2)
        pca_Y = PCA(Y, n_components=2)
        principal_components_X = pca_X.transform(X)
        principal_components_Y = pca_Y.transform(Y)

        # Compute normalized variance quantities:
        variance_data_X = compute_normalized_variance(principal_components_X, X, depvar_names=['A', 'B', 'C', 'D', 'E'], bandwidth_values=np.logspace(-3, 2, 20), scale_unit_box=True)
        variance_data_Y = compute_normalized_variance(principal_components_Y, Y, depvar_names=['F', 'G', 'H', 'I', 'J'], bandwidth_values=np.logspace(-3, 2, 20), scale_unit_box=True)

        # Plot a comparison of normalized variance quantities:
        plt = plot_normalized_variance_comparison((variance_data_X, variance_data_Y),
                                                  ([0,1,2], [0,1,2]),
                                                  ('Blues', 'Reds'),
                                                  figure_size=(10,5),
                                                  title='Normalized variance comparison',
                                                  save_filename='N.pdf')
        plt.close()

    :param variance_data_tuple:
        ``tuple`` of ``VarianceData`` class objects whose normalized variance quantities
        should be compared on one plot. For instance: ``(variance_data_1, variance_data_2)``.
    :param plot_variables_tuple:
        ``list`` of ``int`` specifying indices of variables to be plotted.
        It should have as many elements as there are ``VarianceData`` class objects supplied.
        For instance: ``([], [])`` will plot all variables.
    :param color_map: (optional)
        ``tuple`` of ``str`` or ``matplotlib.colors.ListedColormap`` specifying the colormap to use as per ``matplotlib.cm``.
        It should have as many elements as there are ``VarianceData`` class objects supplied.
        For instance: ``('Blues', 'Reds')``.
    :param figure_size: (optional)
        ``tuple`` specifying figure size.
    :param title: (optional)
        ``str`` specifying plot title. If set to ``None`` title will not be
        plotted.
    :param save_filename: (optional)
        ``str`` specifying plot save location/filename. If set to ``None``
        plot will not be saved. You can also set a desired file extension,
        for instance ``.pdf``. If the file extension is not specified, the default
        is ``.png``.

    :return:
        - **plt** - ``matplotlib.pyplot`` plot handle.
    """

    if not isinstance(figure_size, tuple):
        raise ValueError("Parameter `figure_size` has to be of type `tuple`.")

    if title is not None:
        if not isinstance(title, str):
            raise ValueError("Parameter `title` has to be of type `str`.")

    if save_filename is not None:
        if not isinstance(save_filename, str):
            raise ValueError("Parameter `save_filename` has to be of type `str`.")

    from matplotlib import cm

    markers_list = ["o-","v-","^-","<-",">-","s-","p-","P-","*-","h-","H-","+-","x-","X-","D-","d-","|-","_-"]

    figure = plt.figure(figsize=figure_size)

    variable_count = 0

    for variance_data, plot_variables, color_map in zip(variance_data_tuple, plot_variables_tuple, color_map_tuple):

        color_map_colors = cm.get_cmap(color_map)

        # Extract quantities from the VarianceData class object:
        variable_names = variance_data.variable_names
        bandwidth_values = variance_data.bandwidth_values
        normalized_variance = variance_data.normalized_variance

        if len(plot_variables) != 0:
            variables_to_plot = []
            for i in plot_variables:
                variables_to_plot.append(variable_names[i])
        else:
            variables_to_plot = variable_names

        n_variables = len(variables_to_plot)

        if n_variables == 1:
            variable_colors = np.flipud(color_map_colors([0.8]))
        else:
            variable_colors = np.flipud(color_map_colors(np.linspace(0.2, 0.8, n_variables)))

        # Plot the normalized variance:
        for i, variable_name in enumerate(variables_to_plot):
            plt.semilogx(bandwidth_values, normalized_variance[variable_name], markers_list[variable_count], label=variable_name, color=variable_colors[i])

            variable_count = variable_count + 1

    plt.xlabel(r'$\sigma$', fontsize=font_labels, **csfont)
    plt.ylabel(r'$N(\sigma)$', fontsize=font_labels, **csfont)
    plt.grid(alpha=grid_opacity)

    if variable_count <=5:
        plt.legend(loc='best', fancybox=True, shadow=True, ncol=1, fontsize=font_legend, markerscale=marker_scale_legend)
    else:
        plt.legend(bbox_to_anchor=(1.05,1), fancybox=True, shadow=True, ncol=2, fontsize=font_legend, markerscale=marker_scale_legend)

    if title != None: plt.title(title, fontsize=font_title, **csfont)
    if save_filename != None: plt.savefig(save_filename, dpi=save_dpi, bbox_inches='tight')

    return plt

# ------------------------------------------------------------------------------

def plot_normalized_variance_derivative(variance_data, plot_variables=[], color_map='Blues', figure_size=(10,5), title=None, save_filename=None):
    """
    This function plots a scaled normalized variance derivative (computed over logarithmically scaled bandwidths), :math:`\hat{\mathcal{D}(\sigma)}`,
    over bandwith values :math:`\sigma` from an object of a ``VarianceData`` class.

    *Note:* this function can accomodate plotting up to 18 variables at once.
    You can specify which variables should be plotted using ``plot_variables`` list.

    **Example:**

    .. code:: python

        from PCAfold import PCA, compute_normalized_variance, plot_normalized_variance_derivative
        import numpy as np

        # Generate dummy data set:
        X = np.random.rand(100,5)

        # Perform PCA to obtain the low-dimensional manifold:
        pca_X = PCA(X, n_components=2)
        principal_components = pca_X.transform(X)

        # Compute normalized variance quantities:
        variance_data = compute_normalized_variance(principal_components, X, depvar_names=['A', 'B', 'C', 'D', 'E'], bandwidth_values=np.logspace(-3, 1, 20), scale_unit_box=True)

        # Plot normalized variance derivative:
        plt = plot_normalized_variance_derivative(variance_data,
                                                  plot_variables=[0,1,2],
                                                  color_map='Blues',
                                                  figure_size=(10,5),
                                                  title='Normalized variance derivative',
                                                  save_filename='D-hat.pdf')
        plt.close()

    :param variance_data:
        an object of ``VarianceData`` class objects whose normalized variance derivative quantities
        should be plotted.
    :param plot_variables: (optional)
        ``list`` of ``int`` specifying indices of variables to be plotted.
        By default, all variables are plotted.
    :param color_map: (optional)
        ``str`` or ``matplotlib.colors.ListedColormap`` specifying the colormap to use as per ``matplotlib.cm``. Default is ``'Blues'``.
    :param figure_size: (optional)
        ``tuple`` specifying figure size.
    :param title: (optional)
        ``str`` specifying plot title. If set to ``None`` title will not be
        plotted.
    :param save_filename: (optional)
        ``str`` specifying plot save location/filename. If set to ``None``
        plot will not be saved. You can also set a desired file extension,
        for instance ``.pdf``. If the file extension is not specified, the default
        is ``.png``.

    :return:
        - **plt** - ``matplotlib.pyplot`` plot handle.
    """

    if not isinstance(figure_size, tuple):
        raise ValueError("Parameter `figure_size` has to be of type `tuple`.")

    if title is not None:
        if not isinstance(title, str):
            raise ValueError("Parameter `title` has to be of type `str`.")

    if save_filename is not None:
        if not isinstance(save_filename, str):
            raise ValueError("Parameter `save_filename` has to be of type `str`.")

    from matplotlib import cm
    color_map_colors = cm.get_cmap(color_map)

    markers_list = ["o-","v-","^-","<-",">-","s-","p-","P-","*-","h-","H-","+-","x-","X-","D-","d-","|-","_-"]

    # Extract quantities from the VarianceData class object:
    variable_names = variance_data.variable_names
    derivatives, bandwidth_values, _ = normalized_variance_derivative(variance_data)

    if len(plot_variables) != 0:
        variables_to_plot = []
        for i in plot_variables:
            variables_to_plot.append(variable_names[i])
    else:
        variables_to_plot = variable_names

    n_variables = len(variables_to_plot)

    if n_variables > 18:
        raise ValueError("Only 18 variables can be plotted at once. Consider pre-selecting the variables to plot using `plot_variables` parameter.")

    if n_variables == 1:
        variable_colors = np.flipud(color_map_colors([0.8]))
    else:
        variable_colors = np.flipud(color_map_colors(np.linspace(0.2, 0.8, n_variables)))

    figure = plt.figure(figsize=figure_size)

    # Plot the normalized variance derivative:
    for i, variable_name in enumerate(variables_to_plot):
        plt.semilogx(bandwidth_values, derivatives[variable_name], markers_list[i], label=variable_name, color=variable_colors[i])

    plt.xlabel('$\sigma$', fontsize=font_labels, **csfont)
    plt.ylabel('$\hat{\mathcal{D}}(\sigma)$', fontsize=font_labels, **csfont)
    plt.grid(alpha=grid_opacity)

    if n_variables <=5:
        plt.legend(loc='best', fancybox=True, shadow=True, ncol=1, fontsize=font_legend, markerscale=marker_scale_legend)
    else:
        plt.legend(bbox_to_anchor=(1.05,1), fancybox=True, shadow=True, ncol=2, fontsize=font_legend, markerscale=marker_scale_legend)

    if title != None: plt.title(title, fontsize=font_title, **csfont)
    if save_filename != None: plt.savefig(save_filename, dpi=save_dpi, bbox_inches='tight')

    return plt

# ------------------------------------------------------------------------------

def plot_normalized_variance_derivative_comparison(variance_data_tuple, plot_variables_tuple, color_map_tuple, figure_size=(10,5), title=None, save_filename=None):
    """
    This function plots a comparison of scaled normalized variance derivative (computed over logarithmically scaled bandwidths), :math:`\hat{\mathcal{D}(\sigma)}`,
    over bandwith values :math:`\sigma` from an object of a ``VarianceData`` class.

    *Note:* this function can accomodate plotting up to 18 variables at once.
    You can specify which variables should be plotted using ``plot_variables`` list.

    **Example:**

    .. code:: python

        from PCAfold import PCA, compute_normalized_variance, plot_normalized_variance_derivative_comparison
        import numpy as np

        # Generate dummy data sets:
        X = np.random.rand(100,5)
        Y = np.random.rand(100,5)

        # Perform PCA to obtain low-dimensional manifolds:
        pca_X = PCA(X, n_components=2)
        pca_Y = PCA(Y, n_components=2)
        principal_components_X = pca_X.transform(X)
        principal_components_Y = pca_Y.transform(Y)

        # Compute normalized variance quantities:
        variance_data_X = compute_normalized_variance(principal_components_X, X, depvar_names=['A', 'B', 'C', 'D', 'E'], bandwidth_values=np.logspace(-3, 2, 20), scale_unit_box=True)
        variance_data_Y = compute_normalized_variance(principal_components_Y, Y, depvar_names=['F', 'G', 'H', 'I', 'J'], bandwidth_values=np.logspace(-3, 2, 20), scale_unit_box=True)

        # Plot a comparison of normalized variance derivatives:
        plt = plot_normalized_variance_derivative_comparison((variance_data_X, variance_data_Y),
                                                             ([0,1,2], [0,1,2]),
                                                             ('Blues', 'Reds'),
                                                             figure_size=(10,5),
                                                             title='Normalized variance derivative comparison',
                                                             save_filename='D-hat.pdf')
        plt.close()

    :param variance_data_tuple:
        ``tuple`` of ``VarianceData`` class objects whose normalized variance derivative quantities
        should be compared on one plot. For instance: ``(variance_data_1, variance_data_2)``.
    :param plot_variables_tuple:
        ``list`` of ``int`` specifying indices of variables to be plotted.
        It should have as many elements as there are ``VarianceData`` class objects supplied.
        For instance: ``([], [])`` will plot all variables.
    :param color_map: (optional)
        ``tuple`` of ``str`` or ``matplotlib.colors.ListedColormap`` specifying the colormap to use as per ``matplotlib.cm``.
        It should have as many elements as there are ``VarianceData`` class objects supplied.
        For instance: ``('Blues', 'Reds')``.
    :param figure_size: (optional)
        ``tuple`` specifying figure size.
    :param title: (optional)
        ``str`` specifying plot title. If set to ``None`` title will not be
        plotted.
    :param save_filename: (optional)
        ``str`` specifying plot save location/filename. If set to ``None``
        plot will not be saved. You can also set a desired file extension,
        for instance ``.pdf``. If the file extension is not specified, the default
        is ``.png``.

    :return:
        - **plt** - ``matplotlib.pyplot`` plot handle.
    """

    if not isinstance(figure_size, tuple):
        raise ValueError("Parameter `figure_size` has to be of type `tuple`.")

    if title is not None:
        if not isinstance(title, str):
            raise ValueError("Parameter `title` has to be of type `str`.")

    if save_filename is not None:
        if not isinstance(save_filename, str):
            raise ValueError("Parameter `save_filename` has to be of type `str`.")

    from matplotlib import cm

    markers_list = ["o-","v-","^-","<-",">-","s-","p-","P-","*-","h-","H-","+-","x-","X-","D-","d-","|-","_-"]

    figure = plt.figure(figsize=figure_size)

    variable_count = 0

    for variance_data, plot_variables, color_map in zip(variance_data_tuple, plot_variables_tuple, color_map_tuple):

        color_map_colors = cm.get_cmap(color_map)

        # Extract quantities from the VarianceData class object:
        variable_names = variance_data.variable_names
        derivatives, bandwidth_values, _ = normalized_variance_derivative(variance_data)

        if len(plot_variables) != 0:
            variables_to_plot = []
            for i in plot_variables:
                variables_to_plot.append(variable_names[i])
        else:
            variables_to_plot = variable_names

        n_variables = len(variables_to_plot)

        if n_variables == 1:
            variable_colors = np.flipud(color_map_colors([0.8]))
        else:
            variable_colors = np.flipud(color_map_colors(np.linspace(0.2, 0.8, n_variables)))

        # Plot the normalized variance:
        for i, variable_name in enumerate(variables_to_plot):
            plt.semilogx(bandwidth_values, derivatives[variable_name], markers_list[variable_count], label=variable_name, color=variable_colors[i])

            variable_count = variable_count + 1

    plt.xlabel('$\sigma$', fontsize=font_labels, **csfont)
    plt.ylabel('$\hat{\mathcal{D}}(\sigma)$', fontsize=font_labels, **csfont)
    plt.grid(alpha=grid_opacity)

    if variable_count <=5:
        plt.legend(loc='best', fancybox=True, shadow=True, ncol=1, fontsize=font_legend, markerscale=marker_scale_legend)
    else:
        plt.legend(bbox_to_anchor=(1.05,1), fancybox=True, shadow=True, ncol=2, fontsize=font_legend, markerscale=marker_scale_legend)

    if title != None: plt.title(title, fontsize=font_title, **csfont)
    if save_filename != None: plt.savefig(save_filename, dpi=save_dpi, bbox_inches='tight')

    return plt

# ------------------------------------------------------------------------------

def plot_stratified_metric(metric_in_bins, bins_borders, variable_name=None, metric_name=None, yscale='linear', figure_size=(10,5), title=None, save_filename=None):
    """
    This function plots a stratified metric across bins of a dependent variable.

    **Example:**

    .. code:: python

        from PCAfold import PCA, variable_bins, stratified_coefficient_of_determination, plot_stratified_metric
        import numpy as np

        # Generate dummy data set:
        X = np.random.rand(100,10)

        # Instantiate PCA class object:
        pca_X = PCA(X, scaling='auto', n_components=2)

        # Approximate the data set:
        X_rec = pca_X.reconstruct(pca_X.transform(X))

        # Generate bins:
        (idx, bins_borders) = variable_bins(X[:,0], k=10, verbose=False)

        # Compute stratified R2 in 10 bins of the first variable in a data set:
        r2_in_bins = stratified_coefficient_of_determination(X[:,0], X_rec[:,0], idx=idx, use_global_mean=True, verbose=True)

        # Visualize how R2 changes across bins:
        plt = plot_stratified_metric(r2_in_bins,
                                      bins_borders,
                                      variable_name='$X_1$',
                                      metric_name='$R^2$',
                                      yscale='log',
                                      figure_size=(10,5),
                                      title='Stratified $R^2$',
                                      save_filename='r2.pdf')
        plt.close()

    :param metric_in_bins:
        ``list`` of metric values in each bin.
    :param bins_borders:
        ``list`` of bins borders that were created to stratify the dependent variable.
    :param variable_name: (optional)
        ``str`` specifying the name of the variable for which the metric was computed. If set to ``None``
        label on the x-axis will not be plotted.
    :param metric_name: (optional)
        ``str`` specifying the name of the metric to be plotted on the y-axis. If set to ``None``
        label on the x-axis will not be plotted.
    :param yscale: (optional)
        ``str`` specifying the scale for the y-axis.
    :param figure_size: (optional)
        ``tuple`` specifying figure size.
    :param title: (optional)
        ``str`` specifying plot title. If set to ``None`` title will not be
        plotted.
    :param save_filename: (optional)
        ``str`` specifying plot save location/filename. If set to ``None``
        plot will not be saved. You can also set a desired file extension,
        for instance ``.pdf``. If the file extension is not specified, the default
        is ``.png``.

    :return:
        - **plt** - ``matplotlib.pyplot`` plot handle.
    """

    if not isinstance(figure_size, tuple):
        raise ValueError("Parameter `figure_size` has to be of type `tuple`.")

    if title is not None:
        if not isinstance(title, str):
            raise ValueError("Parameter `title` has to be of type `str`.")

    if save_filename is not None:
        if not isinstance(save_filename, str):
            raise ValueError("Parameter `save_filename` has to be of type `str`.")

    bin_length = bins_borders[1] - bins_borders[0]
    bin_centers = bins_borders[0:-1] + bin_length/2

    figure = plt.figure(figsize=figure_size)
    plt.scatter(bin_centers, metric_in_bins, c='#191b27')
    plt.grid(alpha=grid_opacity)
    plt.yscale(yscale)
    if variable_name is not None: plt.xlabel(variable_name, **csfont, fontsize=font_labels)
    if metric_name is not None: plt.ylabel(metric_name, **csfont, fontsize=font_labels)

    if title is not None: plt.title(title, fontsize=font_title, **csfont)
    if save_filename is not None: plt.savefig(save_filename, dpi=save_dpi, bbox_inches='tight')

    return plt
