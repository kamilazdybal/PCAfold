"""analysis.py: module for manifolds analysis."""

__author__ = "Kamila Zdybal, Elizabeth Armstrong, Alessandro Parente and James C. Sutherland"
__copyright__ = "Copyright (c) 2020-2023, Kamila Zdybal, Elizabeth Armstrong, Alessandro Parente and James C. Sutherland"
__credits__ = ["Department of Chemical Engineering, University of Utah, Salt Lake City, Utah, USA", "Universite Libre de Bruxelles, Aero-Thermo-Mechanics Laboratory, Brussels, Belgium"]
__license__ = "MIT"
__version__ = "2.0.0"
__maintainer__ = ["Kamila Zdybal", "Elizabeth Armstrong"]
__email__ = ["kamilazdybal@gmail.com", "Elizabeth.Armstrong@chemeng.utah.edu", "James.Sutherland@chemeng.utah.edu"]
__status__ = "Production"

import numpy as np
import copy as cp
import multiprocessing as multiproc
from PCAfold import KReg
from scipy.spatial import KDTree
from scipy.spatial import cKDTree
from scipy.optimize import minimize
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import random as rnd
from scipy.interpolate import CubicSpline
from PCAfold.styles import *
from PCAfold import preprocess
from PCAfold import reduction
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

    def __init__(self, bandwidth_values, norm_var, global_var, bandwidth_10pct_rise, keys, norm_var_limit, sample_norm_var, sample_norm_range):
        self._bandwidth_values = bandwidth_values.copy()
        self._normalized_variance = norm_var.copy()
        self._global_variance = global_var.copy()
        self._bandwidth_10pct_rise = bandwidth_10pct_rise.copy()
        self._variable_names = keys.copy()
        self._normalized_variance_limit = norm_var_limit.copy()
        self._sample_normalized_variance = sample_norm_var.copy()
        self._sample_normalized_range = sample_norm_range.copy()

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

    @property
    def sample_normalized_range(self):
        """return a dictionary of the sample normalized ranges for each bandwidth and for each variable"""
        return self._sample_normalized_range.copy()

# ------------------------------------------------------------------------------

def compute_normalized_variance(indepvars, depvars, depvar_names, npts_bandwidth=25, min_bandwidth=None,
                                max_bandwidth=None, bandwidth_values=None, scale_unit_box=True, n_threads=None, compute_sample_norm_var=False, compute_sample_norm_range=False):
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
    :param compute_sample_norm_var:
        (optional, default False) ``bool`` specifying if sample normalized variance should be computed.
    :param compute_sample_norm_range:
        (optional, default False) ``bool`` specifying if sample normalized range should be computed.

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

    if compute_sample_norm_var:

        # Computing normalized variance for each individual observation:
        sample_norm_var = {}
        for idx, key in enumerate(depvar_names):
            sample_local_variance = np.zeros((yi.shape[0], bandwidth_values.size))
            for si in range(bandwidth_values.size):
                sample_local_variance[:,si] = (yi[:, idx] - kregmodResults[si][:,idx])**2
            sample_norm_var[key] = sample_local_variance / global_var[key]

    else:

        sample_norm_var = {}

    if compute_sample_norm_range:

        # Computing normalized range for each individual observation:
        sample_norm_range = {}
        point_tree = cKDTree(xi)
        for idx, key in enumerate(depvar_names):
            neighborhood_range = np.zeros((yi.shape[0], bandwidth_values.size))
            for si in range(bandwidth_values.size):
                for i in range(0,yi.shape[0]):
                    neighbors = point_tree.query_ball_point(xi[i], bandwidth_values[si])
                    yi_neighbors = yi[neighbors,idx]
                    neighborhood_range[i,si] = ((np.max(yi_neighbors) - np.min(yi_neighbors)))**2
            sample_norm_range[key] = neighborhood_range / global_var[key]

    else:

        sample_norm_range = {}

    # computing normalized variance as bandwidth approaches zero to check for non-uniqueness
    lvar_limit = kregmod.predict(xi, 1.e-16)
    nlvar_limit = np.linalg.norm(yi - lvar_limit, axis=0) ** 2
    normvar_limit = dict({key: nlvar_limit[idx] for idx, key in enumerate(depvar_names)})

    solution_data = VarianceData(bandwidth_values, norm_local_var, global_var, bandwidth_10pct_rise, depvar_names, normvar_limit, sample_norm_var, sample_norm_range)

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
                    avg_der[key] = der[key] / np.float64(n_sample_iterations)
                else:
                    avg_der[key] += der[key] / np.float64(n_sample_iterations)

        avg_der_data[p] = avg_der
        normvar_data[p] = nv_data
    return avg_der_data, xder, normvar_data

# ------------------------------------------------------------------------------

def feature_size_map(variance_data, variable_name, cutoff=1, starting_bandwidth_idx='peak', use_variance=False, verbose=False):
    """
    Computes a map of local feature sizes on a manifold.

    **Example:**

    .. code:: python

        from PCAfold import PCA, compute_normalized_variance, feature_size_map
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

        # Compute the feature size map:
        feature_size_map = feature_size_map(variance_data,
                                            variable_name='X_1',
                                            cutoff=1,
                                            starting_bandwidth_idx='peak',
                                            verbose=True)

    :param variance_data:
        an object of ``VarianceData`` class.
    :param variable_name:
        ``str`` specifying the name of the dependent variable for which the feature size map should be computed. It should be as per name specified when computing ``variance_data``.
    :param cutoff: (optional)
        ``float`` or ``int`` specifying the cutoff percentage, :math:`p`. It should be a number between 0 and 100.
    :param starting_bandwidth_idx: (optional)
        ``int`` or ``str`` specifying the index of the starting bandwidth to compute the local feature sizes from. Local feature sizes computed will never be smaller than the starting bandwidth. If set to ``'peak'``, the starting bandwidth will be automatically calculated as the rightmost peak, :math:`\sigma_{peak}`.
    :param verbose: (optional)
        ``bool`` for printing verbose details.

    :return:
        - **feature_size_map** - ``numpy.ndarray`` specifying the local feature sizes on a manifold, :math:`\\mathbf{B}`. It has size ``(n_observations,)``.
    """

    if not isinstance(variance_data, VarianceData):
        raise ValueError("Parameter `variance_data` has to be an instance of class `PCAfold.analysis.VarianceData`.")

    if not isinstance(variable_name, str):
        raise ValueError("Parameter `variable_name` has to be of type `str`.")

    if not isinstance(cutoff, int) and not isinstance(cutoff, float):
        raise ValueError("Parameter `cutoff` has to be of type `int` or `float`.")

    if cutoff < 0 or cutoff > 100:
        raise ValueError("Parameter `cutoff` has to be of a number between 0 and 100.")

    if starting_bandwidth_idx != 'peak':
        if not isinstance(starting_bandwidth_idx, int):
            raise ValueError("Parameter `starting_bandwidth_idx` has to be of type `int` or be set to 'peak'.")

    if not isinstance(verbose, bool):
        raise ValueError("Parameter `verbose` has to be of type `bool`.")

    NV = variance_data.normalized_variance[variable_name]
    if use_variance:
        SNV = variance_data.sample_normalized_variance[variable_name]
    else:
        SNV = variance_data.sample_normalized_range[variable_name]

    derivative, sigmas, _ = normalized_variance_derivative(variance_data)
    derivatives = derivative[variable_name]

    if starting_bandwidth_idx == 'peak':
        idx_peaks, _ = find_peaks(derivatives, height=0)
        starting_bandwidth_idx = idx_peaks[-1]

        if verbose: print('Rightmost peak at index ' + str(starting_bandwidth_idx) + ' is used as the starting bandwidth.\n')

    starting_bandwidth = sigmas[starting_bandwidth_idx]
    starting_derivative = derivatives[starting_bandwidth_idx]
    bandwidth_idx_list = [i for i in range(0, starting_bandwidth_idx)]

    if verbose:
        print('Feature sizes will be computed starting from size:\n' + str(starting_bandwidth) + '\nwhere the normalized variance derivative is equal to:\n' + str(starting_derivative))

    n_observations, _ = np.shape(SNV)

    # Populate the initial bandwidth vector with the starting bandwidth:
    feature_size_map = np.ones((n_observations,)) * starting_bandwidth

    # Run iterative bandwidth update:
    for i, bandwidth_idx in enumerate(bandwidth_idx_list[::-1]):

        # The reason for the (+1) is that the smallest bandwidth has been used up to compute the derivative
        # so the SNV is shifted by one index with respect to the D-hat:
        current_SNV = SNV[:,bandwidth_idx+1]

        # Compute the current maximum normalized variance:
        current_max_SNV = np.max(current_SNV)

        # Look for locations where the local variance is larger than a cutoff:
        (change_bandwidth_idx, ) = np.where(current_SNV > (cutoff/100)*current_max_SNV)

        # Update the feature sizes at locations where the condition is met:
        feature_size_map[change_bandwidth_idx] = sigmas[bandwidth_idx]

    return feature_size_map

# ------------------------------------------------------------------------------

def feature_size_map_smooth(indepvars, feature_size_map, method='median', n_neighbors=10):
    """
    Smooths out a map of local feature sizes on a manifold.

    .. note::

        This function requires the ``scikit-learn`` module. You can install it through:

        ``pip install scikit-learn``

    **Example:**

    .. code:: python

        from PCAfold import PCA, compute_normalized_variance, feature_size_map, smooth_feature_size_map
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

        # Compute the feature size map:
        feature_size_map = feature_size_map(variance_data,
                                            variable_name='X_1',
                                            cutoff=1,
                                            starting_bandwidth_idx='peak',
                                            verbose=True)

        # Smooth out the feature size map:
        updated_feature_size_map = feature_size_map_smooth(principal_components,
                                                           feature_size_map,
                                                           method='median',
                                                           n_neighbors=4)

    :param indepvars:
        ``numpy.ndarray`` specifying the independent variable values. It should be of size ``(n_observations,n_independent_variables)``.
    :param feature_size_map:
        ``numpy.ndarray`` specifying the local feature sizes on a manifold, :math:`\\mathbf{B}`. It should be of size ``(n_observations,)`` or ``(n_observations,1)``.
    :param method: (optional)
        ``str`` specifying the smoothing method. It can be ``'median'``, ``'mean'``, ``'max'`` or ``'min'``.
    :param n_neighbors: (optional)
        ``int`` specifying the number of nearest neighbors to smooth over.

    :return:
        - **updated_feature_size_map** - ``numpy.ndarray`` specifying the smoothed local feature sizes on a manifold, :math:`\\mathbf{B}`. It has size ``(n_observations,)``.
    """

    __methods = ['median', 'mean', 'max', 'min']

    if not isinstance(indepvars, np.ndarray):
        raise ValueError("Parameter `indepvars` has to be of type `numpy.ndarray`.")

    if not isinstance(feature_size_map, np.ndarray):
        raise ValueError("Parameter `feature_size_map` has to be of type `numpy.ndarray`.")

    try:
        (n_observations, n_independent_variables) = np.shape(indepvars)
    except:
        raise ValueError("Parameter `indepvars` has to have size `(n_observations,n_independent_variables)`.")

    try:
        (n_observations_feature_size_map,) = np.shape(feature_size_map)
    except:
        (n_observations_feature_size_map,n_variables_feature_size_map) = np.shape(feature_size_map)
        if n_variables_feature_size_map != 1:
            raise ValueError("Parameter `feature_size_map` has to have size `(n_observations,)` or `(n_observations,1)`.")

    if n_observations != n_observations_feature_size_map:
        raise ValueError("Parameter `indepvars` has different number of observations than `feature_size_map`.")

    if method not in __methods:
        raise ValueError("Parameter `method` can only be 'median', 'mean', 'max', or 'min'.")

    if not isinstance(n_neighbors, int):
        raise ValueError("Parameter `n_neighbors` has to be of type `int`.")

    if n_neighbors < 0 or n_neighbors > n_observations:
        raise ValueError("Parameter `n_neighbors` has to be positive and smaller than the total number of observations, `n_observations`.")

    try:
        from sklearn.neighbors import NearestNeighbors
    except:
        raise ValueError("Nearest neighbors search requires the `sklearn` module: `pip install scikit-learn`.")

    indepvars_normalized, _, _ = preprocess.center_scale(indepvars, scaling='-1to1')

    knn_model = NearestNeighbors(n_neighbors=n_neighbors)
    knn_model.fit(indepvars_normalized)

    average_distances = np.zeros((n_observations,))

    updated_feature_size_map = np.zeros_like(feature_size_map)

    for query_point in range(0,n_observations):

        (idx_neigh) = knn_model.kneighbors(indepvars_normalized[query_point,:][None,:], n_neighbors=n_neighbors, return_distance=False)

        if method == 'median':
            updated_value = np.median(feature_size_map[idx_neigh])
        if method == 'mean':
            updated_value = np.mean(feature_size_map[idx_neigh])
        if method == 'max':
            updated_value = np.max(feature_size_map[idx_neigh])
        if method == 'min':
            updated_value = np.min(feature_size_map[idx_neigh])

        updated_feature_size_map[query_point] = updated_value

    return updated_feature_size_map

# ------------------------------------------------------------------------------

def cost_function_normalized_variance_derivative(variance_data, penalty_function=None, power=1, vertical_shift=1, norm=None, integrate_to_peak=False, rightmost_peak_shift=None):
    """
    Defines a cost function for manifold topology assessment based on the areas, or weighted (penalized) areas, under
    the normalized variance derivatives curves, :math:`\\hat{\\mathcal{D}}(\\sigma)`, for the selected :math:`n_{dep}` dependent variables.

    More information on the theory and application of the cost function can be found in :cite:`zdybal2022cost`.

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
        Set ``penalty_function='log-sigma'`` to weight each area continuously by the :math:`\\log_{10}` -transformed bandwidth.
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
    :param rightmost_peak_shift: (optional)
        ``float`` or ``int`` specifying the percentage, :math:`p`, of shift in the rightmost peak location. If set to a number between 0 and 100, a quantity :math:`p/100 (\\sigma_{max} - \\sigma_{peak, i})` is added to the rightmost peak location. It can be used to move the rightmost peak location further right, for instance if there is a blending of scales in the :math:`\\hat{\\mathcal{D}}(\\sigma)` profile.

    :return:
        - **cost** - ``float`` specifying the normalized cost, :math:`\\mathcal{L}`, or, if ``norm=None``, a list of costs, :math:`A_i`, for each dependent variable.
    """

    __penalty_functions = ['peak', 'sigma', 'log-sigma', 'log-sigma-over-peak']
    __norms = ['average', 'cumulative', 'max', 'median', 'min']

    if not isinstance(variance_data, VarianceData):
        raise ValueError("Parameter `variance_data` has to be an instance of class `PCAfold.analysis.VarianceData`.")

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

    if rightmost_peak_shift is not None:
        if not isinstance(rightmost_peak_shift, int) and not isinstance(rightmost_peak_shift, float):
            raise ValueError("Parameter `rightmost_peak_shift` has to be of type `int` or `float`.")
        if rightmost_peak_shift > 100 or rightmost_peak_shift < 0:
            raise ValueError("Parameter `rightmost_peak_shift` should represent percentage between 0 and 100.")

    derivative, sigma, _ = normalized_variance_derivative(variance_data)

    costs = []

    for variable in variance_data.variable_names:

        idx_peaks, _ = find_peaks(derivative[variable], height=0)
        idx_rightmost_peak = idx_peaks[-1]
        rightmost_peak_location = sigma[idx_rightmost_peak]

        if rightmost_peak_shift is not None:
            rightmost_peak_location = rightmost_peak_location + rightmost_peak_shift/100 * (sigma[-1] - rightmost_peak_location)

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

            elif penalty_function == 'log-sigma':
                normalized_sigma, _, _ = preprocess.center_scale(np.log10(sigma[:,None]), scaling='0to1')
                addition = normalized_sigma[idx_rightmost_peak][0]
                penalty_log_sigma = (abs(np.log10(sigma[indices_to_the_left_of_peak])))**power + vertical_shift * 1. / addition
                cost = np.trapz(derivative[variable][indices_to_the_left_of_peak]*penalty_log_sigma, np.log10(sigma[indices_to_the_left_of_peak]))
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

            elif penalty_function == 'log-sigma':
                normalized_sigma, _, _ = preprocess.center_scale(np.log10(sigma[:,None]), scaling='0to1')
                addition = normalized_sigma[idx_rightmost_peak][0]
                penalty_log_sigma = (abs(np.log10(sigma)))**power + vertical_shift * 1. / addition
                cost = np.trapz(derivative[variable]*penalty_log_sigma, np.log10(sigma))
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

def plot_2d_regression_streamplot(grid_bounds, regression_model, x=None, y=None, resolution=(10,10), extension=(0,0), color='k', x_label=None, y_label=None, s_manifold=None, manifold_color=None, colorbar_label=None, color_map='viridis', colorbar_range=None, manifold_alpha=1, grid_on=True, figure_size=(7,7), title=None, save_filename=None):
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

    if color is not None:
        if not isinstance(color, str):
            raise ValueError("Parameter `color` has to be of type `str`.")

    if x_label is not None:
        if not isinstance(x_label, str):
            raise ValueError("Parameter `x_label` has to be of type `str`.")

    if y_label is not None:
        if not isinstance(y_label, str):
            raise ValueError("Parameter `y_label` has to be of type `str`.")

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
            scat = plt.scatter(x.ravel(), y.ravel(), c='k', marker='o', s=s_manifold, edgecolor='none', alpha=manifold_alpha)
        elif isinstance(manifold_color, str):
            scat = plt.scatter(x.ravel(), y.ravel(), c=manifold_color, cmap=color_map, marker='o', s=s_manifold, edgecolor='none', alpha=manifold_alpha)
        elif isinstance(manifold_color, np.ndarray):
            scat = plt.scatter(x.ravel(), y.ravel(), c=manifold_color.ravel(), cmap=color_map, marker='o', s=s_manifold, edgecolor='none', alpha=manifold_alpha)

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

def plot_3d_regression(x, y, observed, predicted, elev=45, azim=-45, clean=False, x_label=None, y_label=None, z_label=None, color_observed=None, color_predicted=None, s_observed=None, s_predicted=None, alpha_observed=None, alpha_predicted=None, figure_size=(7,7), title=None, save_filename=None):
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
    :param clean: (optional)
        ``bool`` specifying if a clean plot should be made. If set to ``True``, nothing else but the data points and the 3D axes is plotted.
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
    :param s_observed: (optional)
        ``int`` or ``float`` specifying the scatter point size for the observed variable.
    :param s_predicted: (optional)
        ``int`` or ``float`` specifying the scatter point size for the predicted variable.
    :param alpha_observed: (optional)
        ``int`` or ``float`` specifying the point opacity for the observed variable.
    :param alpha_predicted: (optional)
        ``int`` or ``float`` specifying the point opacity for the predicted variable.
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

    if not isinstance(clean, bool):
        raise ValueError("Parameter `clean` has to be of type `bool`.")

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

    if s_observed is None:
        s_observed = scatter_point_size
    else:
        if not isinstance(s_observed, int) and not isinstance(s_observed, float):
            raise ValueError("Parameter `s_observed` has to be of type `int` or `float`.")

    if s_predicted is None:
        s_predicted = scatter_point_size
    else:
        if not isinstance(s_predicted, int) and not isinstance(s_predicted, float):
            raise ValueError("Parameter `s_predicted` has to be of type `int` or `float`.")

    if alpha_observed is None:
        alpha_observed = 0.1
    else:
        if not isinstance(alpha_observed, int) and not isinstance(alpha_observed, float):
            raise ValueError("Parameter `alpha_observed` has to be of type `int` or `float`.")

    if alpha_predicted is None:
        alpha_predicted = 0.4
    else:
        if not isinstance(alpha_predicted, int) and not isinstance(alpha_predicted, float):
            raise ValueError("Parameter `alpha_predicted` has to be of type `int` or `float`.")

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

    scat = ax.scatter(x.ravel(), y.ravel(), observed.ravel(), c=color_observed, marker='o', s=s_observed, alpha=alpha_observed)
    scat = ax.scatter(x.ravel(), y.ravel(), predicted.ravel(), c=color_predicted, marker='o', s=s_predicted, alpha=alpha_predicted)

    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('w')
    ax.yaxis.pane.set_edgecolor('w')
    ax.zaxis.pane.set_edgecolor('w')
    ax.view_init(elev=elev, azim=azim)

    if clean:

        plt.xticks([])
        plt.yticks([])
        ax.set_zticks([])
        ax.spines["top"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)

    else:

        if x_label != None: ax.set_xlabel(x_label, **csfont, fontsize=font_labels, rotation=0, labelpad=20)
        if y_label != None: ax.set_ylabel(y_label, **csfont, fontsize=font_labels, rotation=0, labelpad=20)
        if z_label != None: ax.set_zlabel(z_label, **csfont, fontsize=font_labels, rotation=0, labelpad=20)

        ax.tick_params(pad=5)
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
