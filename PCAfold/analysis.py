import numpy as np
from PCAfold import KReg
from scipy.spatial import KDTree
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable


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
    """

    def __init__(self, bandwidth_values, norm_var, global_var, bandwidth_10pct_rise, keys):
        self._bandwidth_values = bandwidth_values.copy()
        self._normalized_variance = norm_var.copy()
        self._global_variance = global_var.copy()
        self._bandwidth_10pct_rise = bandwidth_10pct_rise.copy()
        self._variable_names = keys.copy()

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


def compute_normalized_variance(indepvars, depvars, depvar_names, npts_bandwidth=25, min_bandwidth=None,
                                max_bandwidth=None, bandwidth_values=None, scale_unit_box=True):
    """
    Compute a normalized variance (and related quantities) for analyzing manifold dimensionality.
    The normalized variance is computed as

    .. math::

        \\mathcal{N}(\\sigma) = \\frac{\\sum_{i=1}^n (y_i - \\mathcal{K}_{i}(\\sigma))^2}{\\sum_{i=1}^n (y_i - \\bar{y} )^2}

    where :math:`\\bar{y}` is the average quantity over the whole manifold and :math:`\\mathcal{K}_{i}(\\sigma)` is the
    weighted average quantity calculated using kernel regression with a Gaussian kernel of bandwidth :math:`\\sigma` centered
    around the :math:`i^{th}` observation. :math:`n` is the number of observations.
    :math:`\\mathcal{N}(\\sigma)` is computed for each bandwidth in an array of bandwidth values.
    By default, the ``indepvars`` are centered and scaled to reside inside a unit box so that the bandwidths have the
    same meaning in each dimensions. Therefore, the bandwidth and its involved calculations are applied in the normalized
    independent variable space. This may be turned off by setting ``scale_unit_box`` to False.
    The bandwidth values may be specified directly through ``bandwidth_values`` or default values will be calculated as a
    logspace from ``min_bandwidth`` to ``max_bandwidth`` with ``npts_bandwidth`` number of values. If left unspecified,
    ``min_bandwidth`` and ``max_bandwidth`` will be calculated as the minimum and maximum nonzero distance between points, respectively.

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
    for si in range(bandwidth_values.size):
        lvar[si, :] = np.linalg.norm(yi - kregmod.predict(xi, bandwidth_values[si]), axis=0) ** 2

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
    solution_data = VarianceData(bandwidth_values, norm_local_var, global_var, bandwidth_10pct_rise, depvar_names)
    return solution_data


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


def logistic_fit(normalized_variance_values, bandwidth_values, show_plot=False):
    """
    Calculates parameters :math:`k` and :math:`\\sigma_0` for a logistic fit,

    .. math::

        \\mathcal{N}(\\sigma) \\approx \\frac{1}{ 1+e^{ -k[ \\log(\\sigma)-\\log(\\sigma_0) ] } },

    of the normalized variance for a single variable over the log scale of the bandwidth values used in the variance calculation.
    This is used in ``assess_manifolds`` to provide metrics for a comparison of manifolds.

    :param normalized_variance_values:
        the array of normalized variance values for a single dependent variable
    :param bandwidth_values:
        the array of bandwidth values corresponding to the ``normalized_variance_values``
    :param show_plot:
        (optional, default False) if True, a plot of the logistic fit, ``normalized_variance_values``, and difference between the two will be shown

    :return:
        the :math:`\\sigma_0` parameter of the logistic, and the :math:`R^2` value for the logistic fit
    """
    log_bandwidth_values = np.log10(bandwidth_values)
    L = 1.  # parameter for the logistic function

    def findlogfit(args):
        """function to minimize in finding logistic fit parameters"""
        k = args[0]
        sigma0 = args[1]

        logistic = L / (1 + np.exp(-k * (log_bandwidth_values - sigma0)))
        return np.sum((normalized_variance_values - logistic) ** 2) / np.sum(normalized_variance_values ** 2)

    iguess = [1., log_bandwidth_values[
        np.argmin(np.abs(normalized_variance_values - 0.5))]]  # initial guess for logistic parameters

    ans = minimize(findlogfit, iguess, tol=1.e-8)

    logistic = L / (1 + np.exp(-ans.x[0] * (log_bandwidth_values - ans.x[1])))
    R2 = r2value(normalized_variance_values, logistic)

    if show_plot:
        diff = normalized_variance_values - logistic
        plt.plot(log_bandwidth_values, normalized_variance_values, 'k*-', label='original')
        plt.plot(log_bandwidth_values, logistic, 'r*', label='fit')
        plt.plot(log_bandwidth_values, diff, 'b*', label='difference')
        plt.xlabel('log(normalized bandwidth)')
        plt.ylabel('normalized variance')
        plt.grid()
        plt.legend()
        plt.show()

    return 10 ** ans.x[1], R2


def assess_manifolds(variancedata_dict, assess_method='min', show_plot=True):
    """
    Provides data for assessing the manifolds represented in the dictionary of ``VarianceData`` classes by the smoothness
    and scales present in the normalized variance, which represent the uniqueness of the dependent variables on the
    manifold and the scales of variation respectively. This is done by measuring how well a logistic function
    fits the normalized variance (as a single continuous rise in the normalized variance indicates a unique manifold)
    which is done with an R-squared value (R2). The shift of the logistic fit (sigma0) can also be taken into account as a
    higher value indicates variation at larger scales.

    :param variancedata_dict:
        dictionary of ``VarianceData`` classes that have been created using ``compute_normalized_variance``
    :param assess_method:
        (optional, default 'min') method of selecting a single R2 and sigma0 value (from all the dependent variables)
        to represent each manifold for the bar plot (min, max, or avg)
    :param show_plot:
        (optional, default True) bar plot for visualizing and comparing manifolds according to the R2 and sigma0 values in the function description

    :return:
        a dictionary of the R2, sigma0 data for each variable in each ``VarianceData`` class of ``variancedata_dict``
    """
    datadict = {}
    for key in variancedata_dict.keys():
        datadict[key] = {}
        R2name = 'R2'  # dictionary key for r-squared values
        x0name = 'sigma0'  # dictionary key for sigma0 logistic shift

        datadict[key][R2name] = []
        datadict[key][x0name] = []
        for varname in variancedata_dict[key].variable_names:
            x0, R2 = logistic_fit(variancedata_dict[key].normalized_variance[varname],
                                  variancedata_dict[key].bandwidth_values)
            datadict[key][R2name].append(R2)
            datadict[key][x0name].append(x0)

        datadict[key][R2name] = np.array(datadict[key][R2name])
        datadict[key][x0name] = np.array(datadict[key][x0name])

    if show_plot:
        x0s = []
        R2s = []

        for key in datadict.keys():
            if assess_method == 'min':
                x0s.append(np.min(datadict[key][x0name]))
                R2s.append(np.min(datadict[key][R2name]))
            elif assess_method == 'max':
                x0s.append(np.max(datadict[key][x0name]))
                R2s.append(np.max(datadict[key][R2name]))
            elif assess_method == 'avg':
                x0s.append(np.mean(datadict[key][x0name]))
                R2s.append(np.mean(datadict[key][R2name]))
            else:
                print('unsupported method')

        x0s = np.array(x0s)
        R2s = np.array(R2s)

        R2sortidx = np.argsort(R2s)

        sorted_labels = np.array(list(datadict.keys()))[R2sortidx]
        data_height = R2s[R2sortidx]
        data_color = x0s[R2sortidx]

        data_color_norm = [(x - min(data_color)) / (max(data_color) - min(data_color)) for x in data_color]
        fig, ax = plt.subplots(figsize=(6, 4))

        my_cmap = plt.cm.get_cmap('Blues')
        colors = my_cmap(data_color_norm)
        rects = ax.barh(list(range(0, len(sorted_labels))), data_height, color=colors, edgecolor='k')

        sm = ScalarMappable(cmap=my_cmap, norm=plt.Normalize(min(data_color), max(data_color)))
        sm.set_array([])

        cbar = plt.colorbar(sm)
        cbar.set_label('manifold spread parameter', rotation=270, labelpad=25)

        plt.gca().set_yticks(list(range(0, len(sorted_labels))))
        plt.gca().set_yticklabels([*sorted_labels])

        xmin = 0.8
        xmax = 1.0
        plt.gca().set_xticks(np.linspace(xmin,xmax,5))

        plt.xlabel("manifold uniqueness parameter")
        plt.xlim([xmin,xmax])
        ax.set_axisbelow(True)
        plt.grid()
        plt.show()

    return datadict
