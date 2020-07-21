import numpy as np
from PCAfold import KReg
from scipy.spatial import KDTree


class VarianceData:
    """
    A class for storing helpful quantities in analyzing dimensionality of manifolds through normalized local variance measures.
    This class will be returned by compute_normalized_local_variance_quantities.
    Attributes
    ----------
          bandwidth_values: the array of bandwidth values used
          normalized_local_variance: dictionary of the normalized local variance for each dependent variable over bandwidth_values
          local_var_dict: dictionary of the local variance for each dependent variable over bandwidth_values
          global_var_dict: dictionary of the global variance for each dependent variable
          idxmax_local_var_dict: dictionary of the index for the observation contributing most to the local variance for each dependent variable
          bandwidth_rise_dict: dictionary of the bandwidth where normalized_local_variance increases by 10% for each dependent variable
          integral_dict: dictionary of the integral of normalized_local_variance over bandwidth_values for each dependent variable
    """

    def __init__(self, bandwidth_values, lvar_dict, gvar_dict, ilvar_dict, bandwidth_rise_dict, integral_dict):
        self.bandwidth_values = bandwidth_values
        self.local_var_dict = lvar_dict
        self.global_var_dict = gvar_dict
        self.idxmax_local_var_dict = ilvar_dict
        self.bandwidth_rise_dict = bandwidth_rise_dict
        self.integral_dict = integral_dict
        self.normalized_local_variance = dict({key: lvar_dict[key] / gvar_dict[key] for key in lvar_dict.keys()})


def compute_normalized_local_variance_quantities(xi, yi, keys, npts_bandwidth=25, min_bandwidth=None, max_bandwidth=None, bandwidth_values=None):
    """
    Compute a normalized local variance (and related quantities) for analyzing manifold dimensionality.
    The local variance is computed for each bandwidth or filter width in an array of bandwidth values.
    These bandwidth values may be specified directly through bandwidth_values or default values will be calculated
    as a logspace from min_bandwidth to max_bandwidth with npts_bandwidth number of values. If left unspecified,
    min_bandwidth and max_bandwidth will be calculated as the minimum/maximum nonzero distance between points.

    :param xi: independent variable values (size: n_observations x n_independent variables)
    :param yi: dependent variable values (size: n_observations x n_dependent variables)
    :param keys: list of strings corresponding to the names of the dependent variables (for saving values in a dictionary)
    :param npts_bandwidth: (optional, default 25) number of points to build a logspace of bandwidth values
    :param min_bandwidth: (optional, default to minimum nonzero interpoint distance) minimum bandwidth
    :param max_bandwidth: (optional, default to estimated maximum interpoint distance) maximum bandwidth
    :param bandwidth_values: (optional) array of bandwidth values, i.e. filter widths for a Gaussian filter, to loop over

    :return: a VarianceData class
    """
    assert xi.ndim == 2, "independent variable array must be 2D: n_observations x n_variables."
    assert yi.ndim == 2, "dependent variable array must be 2D: n_observations x n_variables."
    assert (xi.shape[0] == yi.shape[0]), "The number of observations for dependent and independent variables must match."
    assert (len(keys) == yi.shape[1]), "The provided keys do not match the shape of the dependent variables yi."

    if bandwidth_values is None:
        if min_bandwidth is None:
            tree = KDTree(xi)
            min_bandwidth = np.min(tree.query(xi, k=2)[0][tree.query(xi, k=2)[0][:, 1] > 1.e-16, 1])
        if max_bandwidth is None:
            diff = np.max(xi, axis=0) - np.min(xi, axis=0)
            max_bandwidth = np.sqrt(np.sum(diff * diff))*10
        bandwidth_values = np.logspace(np.log10(min_bandwidth), np.log10(max_bandwidth), npts_bandwidth)
    else:
        if not isinstance(bandwidth_values, np.ndarray):
            raise ValueError("bandwidth_values must be an array.")

    def get_local_variance(sigma):
        """
        function for computing local variance for a single bandwidth
        :param sigma: a single bandwidth value
        :return: the local variance and the index of the largest contributing observation
        """
        # using kernel regression to compute an average using the bandwidth sigma
        kregmod = KReg(xi, yi)
        kc = kregmod.predict(xi, sigma)

        yvars = np.abs(yi - kc)
        imaxvar = np.argmax(yvars, axis=0)
        lvar = np.linalg.norm(yvars, axis=0) ** 2
        return lvar, imaxvar

    lvar = np.zeros((bandwidth_values.size, yi.shape[1]))
    ilvar = np.zeros((bandwidth_values.size, yi.shape[1]), dtype=int)
    for si in range(bandwidth_values.size):
        lvar[si, :], ilvar[si, :] = get_local_variance(bandwidth_values[si])

    # saving the local variance for each yi...
    lvar_dict = dict({key: lvar[:, idx] for idx, key in enumerate(keys)})
    # saving the index of the observation that contributes the most to the local variance for each yi...
    ilvar_dict = dict({key: ilvar[:, idx] for idx, key in enumerate(keys)})
    # saving the global variance for each yi...
    gvar_dict = dict({key: np.linalg.norm(yi[:, idx] - np.mean(yi[:, idx])) ** 2 for idx, key in enumerate(keys)})
    # saving the values of the bandwidth where the normalized variance increases by 10%...
    bandwidth_rise_dict = dict()
    for key in keys:
        bandwidth_idx = np.argwhere(lvar_dict[key] / gvar_dict[key] >= 0.1)
        if len(bandwidth_idx) == 0.:
            bandwidth_rise_dict[key] = None
        else:
            bandwidth_rise_dict[key] = bandwidth_values[bandwidth_idx[0]][0]
    # saving the integral of the normalized variance over the bandwidth...
    integral_dict = dict({key: np.trapz(lvar_dict[key] / gvar_dict[key], np.log10(bandwidth_values)) /
                               (np.max(np.log10(bandwidth_values)) - np.min(np.log10(bandwidth_values))) for idx, key in enumerate(keys)})

    solution_data = VarianceData(bandwidth_values, lvar_dict, gvar_dict, ilvar_dict, bandwidth_rise_dict, integral_dict)
    return solution_data
