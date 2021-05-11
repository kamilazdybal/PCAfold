"""analysis.py: module for manifolds analysis."""

__author__ = "Kamila Zdybal, Elizabeth Armstrong, Alessandro Parente and James C. Sutherland"
__copyright__ = "Copyright (c) 2020, Kamila Zdybal and Elizabeth Armstrong"
__credits__ = ["Department of Chemical Engineering, University of Utah, Salt Lake City, Utah, USA", "Universite Libre de Bruxelles, Aero-Thermo-Mechanics Laboratory, Brussels, Belgium"]
__license__ = "MIT"
__version__ = "1.0.0"
__maintainer__ = ["Kamila Zdybal", "Elizabeth Armstrong"]
__email__ = ["kamilazdybal@gmail.com", "Elizabeth.Armstrong@chemeng.utah.edu", "James.Sutherland@chemeng.utah.edu"]
__status__ = "Production"

import numpy as np
import multiprocessing as multiproc
from PCAfold import KReg
from scipy.spatial import KDTree
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import random as rnd
from scipy.interpolate import CubicSpline
from PCAfold.styles import *
from PCAfold import preprocess
from termcolor import colored


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
    """

    def __init__(self, bandwidth_values, norm_var, global_var, bandwidth_10pct_rise, keys, norm_var_limit):
        self._bandwidth_values = bandwidth_values.copy()
        self._normalized_variance = norm_var.copy()
        self._global_variance = global_var.copy()
        self._bandwidth_10pct_rise = bandwidth_10pct_rise.copy()
        self._variable_names = keys.copy()
        self._normalized_variance_limit = norm_var_limit.copy()

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
        independent variable values (size: n_observations x n_independent variables)
    :param depvars:
        dependent variable values (size: n_observations x n_dependent variables)
    :param depvar_names:
        list of strings corresponding to the names of the dependent variables (for saving values in a dictionary)
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
        a ``VarianceData`` class
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

    # computing normalized variance as bandwidth approaches zero to check for non-uniqueness
    lvar_limit = kregmod.predict(xi, 1.e-16)
    nlvar_limit = np.linalg.norm(yi - lvar_limit, axis=0) ** 2
    normvar_limit = dict({key: nlvar_limit[idx] for idx, key in enumerate(depvar_names)})

    solution_data = VarianceData(bandwidth_values, norm_local_var, global_var, bandwidth_10pct_rise, depvar_names, normvar_limit)
    return solution_data


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

    :param variance_data:
        a ``VarianceData`` class returned from ``compute_normalized_variance``

    :return:
        - a dictionary of :math:`\\hat{\\mathcal{D}}(\\sigma)` for each variable in the provided ``VarianceData`` object
        - the :math:`\\sigma` values where :math:`\\hat{\\mathcal{D}}(\\sigma)` was computed
    """
    x_plus = variance_data.bandwidth_values[2:]
    x_minus = variance_data.bandwidth_values[:-2]
    x = variance_data.bandwidth_values[1:-1]
    derivative_dict = {}
    for key in variance_data.variable_names:
        y_plus = variance_data.normalized_variance[key][2:]
        y_minus = variance_data.normalized_variance[key][:-2]
        derivative = (y_plus-y_minus)/(np.log10(x_plus)-np.log10(x_minus)) + variance_data.normalized_variance_limit[key]
        scaled_derivative = derivative/np.max(derivative)
        derivative_dict[key] = scaled_derivative
    return derivative_dict, x


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


def r2value(observed, predicted):
    """
    Calculate the coefficient of determination :math:`R^2` value

    :param observed:
        observed values
    :param predicted:
        predicted values

    :return:
        coefficient of determination
    """
    r2 = 1. - np.sum((observed - predicted) * (observed - predicted)) / np.sum(
        (observed - np.mean(observed)) * (observed - np.mean(observed)))
    return r2


def stratified_r2(observed, predicted, n_bins, use_global_mean=True, verbose=False):
    """
    Computes the stratified coefficient of determination,
    :math:`R^2`, values. Stratified :math:`R^2` is computed separately in each
    of the ``n_bins`` of an observed dependent variable, :math:`\\phi`.

    :math:`R_j^2` in the :math:`j^{th}` bin can be computed in two ways:

    - If ``use_global_mean=True``, the mean of the entire observed variable is used as a reference:

    .. math::

        R_j^2 = 1 - \\frac{\\sum_{i=1}^{N_j} (\\phi_i^{j} - \\hat{\\phi}_{i}^{j})^2}{\\sum_{i=1}^{N_j} (\\phi_i^{j} - mean(\\phi))^2}

    - If ``use_global_mean=False``, the mean of the considered :math:`j^{th}` bin is used as a reference:

    .. math::

        R_j^2 = 1 - \\frac{\\sum_{i=1}^{N_j} (\\phi_i^{j} - \\hat{\\phi}_{i}^{j})^2}{\\sum_{i=1}^{N_j} (\\phi_i^{j} - mean(\\phi^{j}))^2}

    where :math:`N_j` is the number of observations in the :math:`j^{th}` bin and
    :math:`\\hat{\\phi}` is the predicted dependent variable.

    .. note::

        After running this function you can call
        ``analysis.plot_stratified_r2(r2_in_bins, bins_borders)`` on the
        function outputs and it will visualize how stratified :math:`R^2` changes across bins.

    .. warning::

        The stratified :math:`R^2` metric can be misleading if there are large
        variations in point density in an observed variable. For instance, below is a data set
        composed of lines of points that have uniform spacing on the :math:`x` axis
        but become more and more sparse in the direction of increasing :math:`\\phi`
        due to an increasing gradient of :math:`\\phi`.
        If bins are narrow enough (``n_bins`` is high enough), a single bin
        (like the bin bounded by the red dashed lines) can contain only one of
        those lines of points for high value of :math:`\\phi`. :math:`R^2` will then be computed
        for constant, or almost constant observations, even though globally those
        observations lie in a location of a large gradient of the observed variable!

        .. image:: ../images/stratified-r2.png
            :width: 500
            :align: center

    **Example:**

    .. code:: python

        from PCAfold import PCA, stratified_r2, plot_stratified_r2
        import numpy as np

        # Generate dummy data set:
        X = np.random.rand(100,10)

        # Instantiate PCA class object:
        pca_X = PCA(X, scaling='auto', n_components=2)

        # Approximate the data set:
        X_rec = pca_X.reconstruct(pca_X.transform(X))

        # Compute stratified R2 in 10 bins of the first variable in a data set:
        (r2_in_bins, bins_borders) = stratified_r2(X[:,0], X_rec[:,0], n_bins=10, use_global_mean=True, verbose=True)

        # Plot the stratified R2 values:
        plot_stratified_r2(r2_in_bins, bins_borders)

    :param observed:
        ``numpy.ndarray`` specifying the observed values of a single dependent variable. It should be of size ``(n_observations,)`` or ``(n_observations, 1)``.
    :param predicted:
        ``numpy.ndarray`` specifying the predicted values of a single dependent variable. It should be of size ``(n_observations,)`` or ``(n_observations, 1)``.
    :param n_bins:
        ``int`` specifying the number of bins to consider in a dependent variable (uses the ``preprocess.variable_bins`` function to generate bins).
    :param use_global_mean: (optional)
        ``bool`` specifying if global mean of the observed variable should be used as a reference in :math:`R^2` calculation.
    :param verbose: (optional)
        ``bool`` for printing sizes (number of observations) and :math:`R^2` values in each bin.

    :return:
        - **r2_in_bins** - ``list`` specifying the coefficients of determination :math:`R^2` in each bin. It has length ``n_bins``.
        - **bins_borders** - ``list`` specifying the bins borders that were created to stratify the dependent variable. It has length ``n_bins+1``.
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

    if not isinstance(n_bins, int):
        raise ValueError("Parameter `n_bins` has to be an integer.")

    if n_bins < 1:
        raise ValueError("Parameter `n_bins` has to be an integer larger than 0.")

    if not isinstance(use_global_mean, bool):
        raise ValueError("Parameter `use_global_mean` has to be a boolean.")

    if not isinstance(verbose, bool):
        raise ValueError("Parameter `verbose` has to be a boolean.")

    __observed = observed.ravel()
    __predicted = predicted.ravel()

    (idx, bins_borders) = preprocess.variable_bins(__observed, n_bins, verbose=False)

    r2_in_bins = []

    if use_global_mean:
        global_mean = np.mean(__observed)

    for cl in np.unique(idx):

        (idx_bin,) = np.where(idx==cl)

        if use_global_mean:
            r2 = 1. - np.sum((__observed[idx_bin] - __predicted[idx_bin]) * (__observed[idx_bin] - __predicted[idx_bin])) / np.sum(
                (__observed[idx_bin] - global_mean) * (__observed[idx_bin] - global_mean))
        else:
            r2 = r2value(__observed[idx_bin], __predicted[idx_bin])

        constant_bin_metric_min = np.min(__observed[idx_bin])/np.mean(__observed[idx_bin])
        constant_bin_metric_max = np.max(__observed[idx_bin])/np.mean(__observed[idx_bin])

        if verbose:
            if (abs(constant_bin_metric_min - 1) < 0.01) and (abs(constant_bin_metric_max - 1) < 0.01):
                print('Bin\t' + str(cl+1) + '\t| size\t ' + str(len(idx_bin)) + '\t| R2\t' + str(round(r2,6)) + '\t| ' + colored('This bin has almost constant values.', 'red'))
            else:
                print('Bin\t' + str(cl+1) + '\t| size\t ' + str(len(idx_bin)) + '\t| R2\t' + str(round(r2,6)))

        r2_in_bins.append(r2)

    return (r2_in_bins, bins_borders)


def random_sampling_normalized_variance(sampling_percentages, indepvars, depvars, depvar_names,
                                        n_sample_iterations=1, verbose=True, npts_bandwidth=25, min_bandwidth=None,
                                        max_bandwidth=None, bandwidth_values=None, scale_unit_box=True, n_threads=None):
    """
    Compute the normalized variance derivatives :math:`\\hat{\\mathcal{D}}(\\sigma)` for random samples of the provided
    data specified using ``sampling_percentages``. These will be averaged over ``n_sample_iterations`` iterations. Analyzing
    the shift in peaks of :math:`\\hat{\\mathcal{D}}(\\sigma)` due to sampling can distinguish between characteristic
    features and non-uniqueness due to a transformation/reduction of manifold coordinates. True features should not show
    significant sensitivity to sampling while non-uniqueness/folds in the manifold will.

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

            der, xder = normalized_variance_derivative(nv_data[it])
            for key in der.keys():
                if it == 0:
                    avg_der[key] = der[key] / np.float(n_sample_iterations)
                else:
                    avg_der[key] += der[key] / np.float(n_sample_iterations)

        avg_der_data[p] = avg_der
        normvar_data[p] = nv_data
    return avg_der_data, xder, normvar_data


################################################################################
#
# Plotting functions
#
################################################################################

def plot_3d_regression(x, y, observed, predicted, elev=45, azim=-45, x_label=None, y_label=None, z_label=None, figure_size=(7,7), title=None, save_filename=None):
    """
    This function plots the result of regression of a dependent variable on top
    of a two-dimensional manifold defined by ``x`` and ``y``.

    **Example:**

    .. code:: python

        from PCAfold import PCA, plot_3d_regression
        import numpy as np

        # Generate dummy data set:
        X = np.random.rand(100,10)

        # Obtain 3-dimensional manifold from PCA:
        pca_X = PCA(X)
        PCs = pca_X.transform(X)
        X_rec = pca_X.reconstruct(PCs)

        # Plot the manifold:
        plt = plot_3d_regression(X[:,0], X[:,1], X[:,0], X_rec[:,0], elev=45, azim=-45, x_label='$x$', y_label='$y$', z_label='$z$', figure_size=(10,10), title='3D regression', save_filename='3d-regression.pdf')
        plt.close()

    :param x:
        variable on the :math:`x`-axis. It should be of type ``numpy.ndarray`` and size
        ``(n_observations,)`` or ``(n_observations,1)``.
    :param y:
        variable on the :math:`y`-axis. It should be of type ``numpy.ndarray`` and size
        ``(n_observations,)`` or ``(n_observations,1)``.
    :param observed:
        vector of the observed values of a single dependent variable.
        It should be of type ``numpy.ndarray`` and size ``(n_observations,)`` or ``(n_observations, 1)``.
    :param predicted:
        vector of the predicted values of a single dependent variable.
        It should be of type ``numpy.ndarray`` and size ``(n_observations,)`` or ``(n_observations, 1)``.
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
    :param figure_size: (optional)
        tuple specifying figure size.
    :param title: (optional)
        string specifying plot title. If set to ``None`` title will not be
        plotted.
    :param save_filename: (optional)
        string specifying plot save location/filename. If set to ``None``
        plot will not be saved.
        You can also set a desired file extension,
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

    color_observed = '#191b27'
    color_predicted = '#ff928b'

    fig = plt.figure(figsize=figure_size)
    ax = fig.add_subplot(111, projection='3d')

    scat = ax.scatter(x.ravel(), y.ravel(), observed.ravel(), c=color_observed, marker='o', s=scatter_point_size, alpha=1)
    scat = ax.scatter(x.ravel(), y.ravel(), predicted.ravel(), c=color_predicted, marker='o', s=scatter_point_size, alpha=1)

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

    lgnd = plt.legend(['Observed', 'Predicted'], fontsize=font_legend, bbox_to_anchor=(1,1), loc="upper left")
    lgnd.legendHandles[0]._sizes = [marker_size*5]
    lgnd.legendHandles[1]._sizes = [marker_size*5]

    if title != None: ax.set_title(title, **csfont, fontsize=font_title)
    if save_filename != None: plt.savefig(save_filename, dpi=save_dpi, bbox_inches='tight')

    return plt

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
        plt = plot_normalized_variance(variance_data, plot_variables=[0,1,2], color_map='Blues', figure_size=(10,5), title='Normalized variance', save_filename='N.pdf')
        plt.close()

    :param variance_data:
        an object of ``VarianceData`` class objects whose normalized variance quantities
        should be plotted.
    :param plot_variables: (optional)
        list of integers specifying indices of variables to be plotted.
        By default, all variables are plotted.
    :param color_map: (optional)
        colormap to use as per ``matplotlib.cm``. Default is *Blues*.
    :param figure_size: (optional)
        tuple specifying figure size.
    :param title: (optional)
        string specifying plot title. If set to ``None``
        title will not be plotted.
    :param save_filename: (optional)
        plot save location/filename. If set to ``None`` plot will not be saved.
        You can also set a desired file extension,
        for instance ``.pdf``. If the file extension is not specified, the default
        is ``.png``.

    :return:
        - **plt** - ``matplotlib.pyplot`` plot handle.
    """

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
        plt = plot_normalized_variance_comparison((variance_data_X, variance_data_Y), ([0,1,2], [0,1,2]), ('Blues', 'Reds'), title='Normalized variance comparison', save_filename='N.pdf')
        plt.close()

    :param variance_data_tuple:
        a tuple of ``VarianceData`` class objects whose normalized variance quantities
        should be compared on one plot. For instance: ``(variance_data_1, variance_data_2)``.
    :param plot_variables_tuple:
        list of integers specifying indices of variables to be plotted.
        It should have as many elements as there are ``VarianceData`` class objects supplied.
        For instance: ``([], [])`` will plot all variables.
    :param color_map_tuple:
        colormap to use as per ``matplotlib.cm``.
        It should have as many elements as there are ``VarianceData`` class objects supplied.
        For instance: ``('Blues', 'Reds')``.
    :param figure_size: (optional)
        tuple specifying figure size.
    :param title: (optional)
        string specifying plot title. If set to ``None``
        title will not be plotted.
    :param save_filename: (optional)
        plot save location/filename. If set to ``None`` plot will not be saved.
        You can also set a desired file extension,
        for instance ``.pdf``. If the file extension is not specified, the default
        is ``.png``.

    :return:
        - **plt** - ``matplotlib.pyplot`` plot handle.
    """

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

    plt.xlabel('$\sigma$', fontsize=font_labels, **csfont)
    plt.ylabel('$N(\sigma)$', fontsize=font_labels, **csfont)
    plt.grid(alpha=grid_opacity)

    if variable_count <=5:
        plt.legend(loc='best', fancybox=True, shadow=True, ncol=1, fontsize=font_legend, markerscale=marker_scale_legend)
    else:
        plt.legend(bbox_to_anchor=(1.05,1), fancybox=True, shadow=True, ncol=2, fontsize=font_legend, markerscale=marker_scale_legend)

    if title != None: plt.title(title, fontsize=font_title, **csfont)
    if save_filename != None: plt.savefig(save_filename, dpi=save_dpi, bbox_inches='tight')

    return plt

def plot_normalized_variance_derivative(variance_data, plot_variables=[], color_map='Blues', figure_size=(10,5), title=None, save_filename=None):
    """
    This function plots a scaled normalized variance derivative (computed over logarithmically scaled bandwidths), :math:`\hat{\mathcal{D}(\sigma)}`,
    over bandwith values :math:`\sigma` from an object of a ``VarianceData`` class.

    *Note:* this function can accomodate plotting up to 18 variables at once.
    You can specify which variables should be plotted using ``plot_variables`` list.

    Example is similar to that found for ``plot_normalized_variance``.

    :param variance_data:
        an object of ``VarianceData`` class objects whose normalized variance derivative quantities
        should be plotted.
    :param plot_variables: (optional)
        list of integers specifying indices of variables to be plotted.
        By default, all variables are plotted.
    :param color_map: (optional)
        colormap to use as per ``matplotlib.cm``. Default is *Blues*.
    :param figure_size: (optional)
        tuple specifying figure size.
    :param title: (optional)
        string specifying plot title. If set to ``None``
        title will not be plotted.
    :param save_filename: (optional)
        plot save location/filename. If set to ``None`` plot will not be saved.
        You can also set a desired file extension,
        for instance ``.pdf``. If the file extension is not specified, the default
        is ``.png``.

    :return:
        - **plt** - ``matplotlib.pyplot`` plot handle.
    """

    from matplotlib import cm
    color_map_colors = cm.get_cmap(color_map)

    markers_list = ["o-","v-","^-","<-",">-","s-","p-","P-","*-","h-","H-","+-","x-","X-","D-","d-","|-","_-"]

    # Extract quantities from the VarianceData class object:
    variable_names = variance_data.variable_names
    derivatives, bandwidth_values = normalized_variance_derivative(variance_data)

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


def plot_normalized_variance_derivative_comparison(variance_data_tuple, plot_variables_tuple, color_map_tuple, figure_size=(10,5), title=None, save_filename=None):
    """
    This function plots a comparison of scaled normalized variance derivative (computed over logarithmically scaled bandwidths), :math:`\hat{\mathcal{D}(\sigma)}`,
    over bandwith values :math:`\sigma` from an object of a ``VarianceData`` class.

    *Note:* this function can accomodate plotting up to 18 variables at once.
    You can specify which variables should be plotted using ``plot_variables`` list.

    Example is similar to that found for ``plot_normalized_variance_comparison``.

    :param variance_data_tuple:
        a tuple of ``VarianceData`` class objects whose normalized variance derivative quantities
        should be compared on one plot. For instance: ``(variance_data_1, variance_data_2)``.
    :param plot_variables_tuple:
        list of integers specifying indices of variables to be plotted.
        It should have as many elements as there are ``VarianceData`` class objects supplied.
        For instance: ``([], [])`` will plot all variables.
    :param color_map_tuple:
        colormap to use as per ``matplotlib.cm``.
        It should have as many elements as there are ``VarianceData`` class objects supplied.
        For instance: ``('Blues', 'Reds')``.
    :param figure_size: (optional)
        tuple specifying figure size.
    :param title: (optional)
        string specifying plot title. If set to ``None``
        title will not be plotted.
    :param save_filename: (optional)
        plot save location/filename. If set to ``None`` plot will not be saved.
        You can also set a desired file extension,
        for instance ``.pdf``. If the file extension is not specified, the default
        is ``.png``.

    :return:
        - **plt** - ``matplotlib.pyplot`` plot handle.
    """

    from matplotlib import cm

    markers_list = ["o-","v-","^-","<-",">-","s-","p-","P-","*-","h-","H-","+-","x-","X-","D-","d-","|-","_-"]

    figure = plt.figure(figsize=figure_size)

    variable_count = 0

    for variance_data, plot_variables, color_map in zip(variance_data_tuple, plot_variables_tuple, color_map_tuple):

        color_map_colors = cm.get_cmap(color_map)

        # Extract quantities from the VarianceData class object:
        variable_names = variance_data.variable_names
        derivatives, bandwidth_values = normalized_variance_derivative(variance_data)

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

def plot_stratified_r2(r2_in_bins, bins_borders, variable_name=None, figure_size=(10,5), title=None, save_filename=None):
    """
    This function plots the stratified coefficient of determination :math:`R^2`
    across bins of a dependent variable.

    **Example:**

    .. code:: python

        from PCAfold import PCA, stratified_r2, plot_stratified_r2
        import numpy as np

        # Generate dummy data set:
        X = np.random.rand(100,10)

        # Instantiate PCA class object:
        pca_X = PCA(X, scaling='auto', n_components=2)

        # Approximate the data set:
        X_rec = pca_X.reconstruct(pca_X.transform(X))

        # Compute stratified R2 in 10 bins of the first variable in a data set:
        (r2_in_bins, bins_borders) = stratified_r2(X[:,0], X_rec[:,0], n_bins=10, use_global_mean=True, verbose=True)

        # Visualize how R2 changes across bins:
        plt = plot_stratified_r2(r2_in_bins, bins_borders, variable_name='$X_1$', figure_size=(10,5), title='Stratified R2', save_filename='r2.pdf')
        plt.close()

    :param r2_in_bins:
        list of coefficients of determination :math:`R^2` in each bin as per ``analysis.stratified_r2`` function.
    :param bins_borders:
        list of bins borders that were created to stratify the dependent variable as per ``analysis.stratified_r2`` function.
    :param variable_name: (optional)
        string specifying the name of the variable for which :math:`R^2` were computed. If set to ``None``
        label on the x-axis will not be plotted.
    :param figure_size: (optional)
        tuple specifying figure size.
    :param title: (optional)
        string specifying plot title. If set to ``None``
        title will not be plotted.
    :param save_filename: (optional)
        plot save location/filename. If set to ``None`` plot will not be saved.
        You can also set a desired file extension,
        for instance ``.pdf``. If the file extension is not specified, the default
        is ``.png``.

    :return:
        - **plt** - ``matplotlib.pyplot`` plot handle.
    """

    bin_length = bins_borders[1] - bins_borders[0]
    bin_centers = bins_borders[0:-1] + bin_length/2

    figure = plt.figure(figsize=figure_size)
    plt.scatter(bin_centers, r2_in_bins, c='#191b27')
    plt.grid(alpha=0.3)
    if variable_name != None: plt.xlabel(variable_name)
    plt.ylabel('$R^2$ [-]')

    if title != None: plt.title(title, fontsize=font_title, **csfont)
    if save_filename != None: plt.savefig(save_filename, dpi=save_dpi, bbox_inches='tight')

    return plt
