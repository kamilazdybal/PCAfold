import numpy as np
from cykernel import k_pure_python_cython as cython_kernel_evaluate


def norm_local_var(xi, yi, keys, sigma_values=np.logspace(-6, 1, 160)):
    """
    Compute a normalized local variance using kernel regression with varying bandwidth sigma
    :param xi: independent variable values (size: # observations x # independent variables)
    :param yi: dependent variable values (size: # observations x # dependent variables)
    :param keys: list of strings corresponding to the names of the dependent variables (for saving values in a dictionary)
    :param sigma_values: array of bandwidth values to loop over
    :return:
      sigma_values: the array of bandwidth values
      error_dict: dictionary of the local variance for each dependent variable over the bandwidth
      ierror_dict: dictionary of the index for the observation contributing most to the local variance for each dependent variable
      error_inf_dict: dictionary of the global variance for each dependent variable
      sigma_rise_dict: dictionary of the bandwidth where the normalized variance increases by 10% for each dependent variable
      integral_dict: dictionary of the integral of the normalized variance over the bandwidth for each dependent variable
    """

    assert (len(keys) == yi.shape[1]), "The provided keys do not match the shape of the dependent variables yi."
    assert (xi.shape[0] == yi.shape[0]), "The number of observations for dependent and independent variables must match."

    # function for computing local variance for a single bandwidth
    # returns the local variance and the index of the largest contributing observation
    def get_smoother_error(sig):
        kc = np.zeros_like(yi)
        s = np.zeros_like(xi[:, 0].ravel()) + sig
        cython_kernel_evaluate(xi, kc, xi, yi, s)

        yvars = np.abs(yi - kc)
        imaxvar = np.argmax(yvars, axis=0)
        maxvar = np.linalg.norm(yvars, axis=0) ** 2
        return maxvar, imaxvar

    errs = np.zeros((sigma_values.size, yi.shape[1]))
    ierrs = np.zeros((sigma_values.size, yi.shape[1]), dtype=int)
    for si in range(sigma_values.size):
        errs[si, :], ierrs[si, :] = get_smoother_error(sigma_values[si])

    # saving the local variance for each yi...
    error_dict = dict({key: errs[:, idx] for idx, key in enumerate(keys)})
    # saving the index of the observation that contributes the most to the local variance for each yi...
    ierror_dict = dict({key: ierrs[:, idx] for idx, key in enumerate(keys)})
    # saving the global variance for each yi...
    error_inf_dict = dict({key: np.linalg.norm(yi[:, idx] - np.mean(yi[:, idx])) ** 2 for idx, key in enumerate(keys)})
    # saving the values of sigma where the normalized variance increases by 10%...
    sigma_rise_dict = dict({key: sigma_values[np.argwhere(error_dict[key] / error_inf_dict[key] >= 0.1)[0]] for idx, key in enumerate(keys)})
    # saving the integral of the normalized variance over the bandwidth...
    integral_dict = dict({key: np.trapz(error_dict[key] / error_inf_dict[key], np.log10(sigma_values)) /
                               (np.max(np.log10(sigma_values)) - np.min(np.log10(sigma_values))) for idx, key in enumerate(keys)})

    return sigma_values, error_dict, ierror_dict, error_inf_dict, sigma_rise_dict, integral_dict
