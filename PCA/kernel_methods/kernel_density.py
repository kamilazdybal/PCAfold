import numpy as np
from math import pi

# Implementation of the kernel density method from:
#
# Coussement, A., Gicquel, O., & Parente, A. (2012). Kernel density weighted
# principal component analysis of combustion processes. Combustion and flame,
# 159(9), 2844-2855.

# Computes eq.(26):
def bandwidth(n, mean_standard_deviation):
    """
    This function computes kernel bandwidth.

    Input:
    ----------
    `n`           - number of observations in a data set or a variable vector.
    `mean_standard_deviation`
                  - mean standard deviation in the entire data set or a variable
                    vector.

    Output:
    ----------
    `h`           - kernel bandwidth, scalar.
    """

    h = (4*mean_standard_deviation/(3*n))**(1/5)

    return(h)

# Computes eq.(21):
def distance(x1, x2):
    """
    This function computes distance between two observations.

    Input:
    ----------
    `x1`          - first observation.
    `x2`          - second observation.

    Output:
    ----------
    `d`           - distance between `x1` and `x2`.
    """

    d = abs(x1 - x2)

    return(d)

# Computes eq.(22):
def gaussian_kernel(x1, x2, n, mean_standard_deviation):
    """
    This function computes a Gaussian kernel.

    Input:
    ----------
    `x1`          - first observation.
    `x2`          - second observation.
    `n`           - number of observations in a data set or a variable vector.
    `mean_standard_deviation`
                  - mean standard deviation in the entire data set or a variable
                    vector.

    Output:
    ----------
    `K`           - Gaussian kernel, scalar.
    """

    d = distance(x1, x2)
    h = bandwidth(n, mean_standard_deviation)

    K = (1/(2*pi*h**2))**0.5 * np.exp(- d/(2*h**2))

    return(K)

# Computes eq.(23):
def variable_density(x, mean_standard_deviation):
    """
    This function computes a vector of variable densities for all observations.

    Input:
    ----------
    `x`           - single variable vector.
    `mean_standard_deviation`
                  - mean standard deviation in the entire data set or a variable
                    vector.

    Output:
    ----------
    `Kck`         - a vector of variable densities for all observations, it has
                    the same size as the variable vector `x`.
    """

    n = len(x)

    Kck = np.zeros((n,1))

    for i in range(0,n):

        gaussian_kernel_sum = 0

        for j in range(0,n):

            gaussian_kernel_sum = gaussian_kernel_sum + gaussian_kernel(x[i], x[j], n, mean_standard_deviation)

        Kck[i] = 1/n * gaussian_kernel_sum

    return(Kck)

# Computes eq.(24):
def multi_variable_global_density(X):
    """
    This function computes a vector of variable global densities for a
    multi-variable case, for all observations.

    Input:
    ----------
    `X`           - multi-variable data set matrix.

    Output:
    ----------
    `Kc`          - a vector of global densities for all observations.
    """

    (n, n_vars) = np.shape(X)

    mean_standard_deviation = np.mean(np.std(X, axis=0))

    Kck_matrix = np.zeros((n, n_vars))

    for variable in range(0, n_vars):

        Kck_matrix[:,variable] = np.reshape(variable_density(X[:,variable], mean_standard_deviation), (n,))

    # Compute the global densities vector:
    Kc = np.zeros((n,1))

    K = 1

    for i in range(0,n):

        Kc[i] = K * np.prod(Kck_matrix[i,:])

    return(Kc)

# Computes eq.(25):
def multi_variable_observation_weights(X):
    """
    This function computes a vector of observation weights for a
    multi-variable case.

    Input:
    ----------
    `X`           - multi-variable data set matrix.

    Output:
    ----------
    `W_c`         - a vector of observation weights.
    """

    (n, n_vars) = np.shape(X)

    W_c = np.zeros((n,1))

    Kc = multi_variable_global_density(X)

    Kc_inv = 1/Kc

    for i in range(0,n):

        W_c[i] = Kc_inv[i] / np.max(Kc_inv)

    return(W_c)

# Computes eq.(20):
def single_variable_observation_weights(x):
    """
    This function computes a vector of observation weights for a
    single-variable case.

    Input:
    ----------
    `X`           - multi-variable data set matrix.

    Output:
    ----------
    `W_c`         - a vector of observation weights.
    """

    n = len(x)

    mean_standard_deviation = np.std(x)

    W_c = np.zeros((n,1))

    Kc = variable_density(x, mean_standard_deviation)

    Kc_inv = 1/Kc

    for i in range(0,n):

        W_c[i] = Kc_inv[i] / np.max(Kc_inv)

    return(W_c)
