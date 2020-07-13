import numpy as np
from PCAfold import KReg


def compute_normalized_local_variance_quantities(xi, yi, keys, bandwidth_values=np.logspace(-6, 1, 160)):
    """
    Compute a normalized local variance (and related quantities) for analyzing manifold dimensionality.
    The local variance is computed for each bandwidth or filter width in bandwidth_values.

    :param xi: independent variable values (size: n_observations x n_independent variables)
    :param yi: dependent variable values (size: n_observations x n_dependent variables)
    :param keys: list of strings corresponding to the names of the dependent variables (for saving values in a dictionary)
    :param bandwidth_values: array of bandwidth values, i.e. filter widths for a Gaussian filter, to loop over

    :return: a class with the following members:
          bandwidth_values: the array of bandwidth values used
          normalized_local_variance: dictionary of the normalized local variance for each dependent variable over bandwidth_values
          local_var_dict: dictionary of the local variance for each dependent variable over bandwidth_values
          global_var_dict: dictionary of the global variance for each dependent variable
          idxmax_local_var_dict: dictionary of the index for the observation contributing most to the local variance for each dependent variable
          bandwidth_rise_dict: dictionary of the bandwidth where normalized_local_variance increases by 10% for each dependent variable
          integral_dict: dictionary of the integral of normalized_local_variance over bandwidth_values for each dependent variable
    """

    assert (len(keys) == yi.shape[1]), "The provided keys do not match the shape of the dependent variables yi."
    assert (xi.shape[0] == yi.shape[
        0]), "The number of observations for dependent and independent variables must match."

    class Variance_Data:
        """
        A class for storing helpful quantities in analyzing dimensionality of manifolds through normalized local variance measures.
        This class will be returned by the function. Parameter descriptions are provided in the function return description.
        """

        def __init__(self, bandwidth_values, lvar_dict, gvar_dict, ilvar_dict, bandwidth_rise_dict, integral_dict):
            self.bandwidth_values = bandwidth_values
            self.local_var_dict = lvar_dict
            self.global_var_dict = gvar_dict
            self.idxmax_local_var_dict = ilvar_dict
            self.bandwidth_rise_dict = bandwidth_rise_dict
            self.integral_dict = integral_dict
            self.normalized_local_variance = dict({key: lvar_dict[key] / gvar_dict[key] for key in lvar_dict.keys()})

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
    bandwidth_rise_dict = dict(
        {key: bandwidth_values[np.argwhere(lvar_dict[key] / gvar_dict[key] >= 0.1)[0]] for idx, key in
         enumerate(keys)})
    # saving the integral of the normalized variance over the bandwidth...
    integral_dict = dict({key: np.trapz(lvar_dict[key] / gvar_dict[key], np.log10(bandwidth_values)) /
                               (np.max(np.log10(bandwidth_values)) - np.min(np.log10(bandwidth_values))) for idx, key in enumerate(keys)})

    solution_data = Variance_Data(bandwidth_values, lvar_dict, gvar_dict, ilvar_dict, bandwidth_rise_dict, integral_dict)
    return solution_data
