import numpy as np

# Single-variable kernel density method:





# Multi-variable kernel density method:

# Computes eq.(26):
def bandwidth(n):

    h = (4/(3*n))**(1/5)

    return(h)

# Computes eq.(21):
def distance(x1, x2):
    """
    x1 and x2 are scalars.

    Returns a scalar.
    """

    d = abs(x1 - x2)

    return(d)

# Computes eq.(22):
def gaussian_kernel(x1, x2, n):
    """
    x1 and x2 are scalars.

    Returns a scalar.
    """

    d = distance(x1, x2)
    h = bandwidth(n)

    K = (1/(2*math.pi*h**2))**0.5 * np.exp(- d/(2*h**2))

    return(K)

# Computes eq.(23):
def variable_density(x):
    """
    x is a single variable vector.

    Returns a vector of variable densities for all observations, size(n,1).
    """

    (n,) = np.shape(x)

    Kck = np.zeros((n,1))

    for i in range(0,n):

        gaussian_kernel_sum = 0

        for j in range(0,n):

            gaussian_kernel_sum = gaussian_kernel_sum + gaussian_kernel(x[i], x[j], n)

        Kck[i] = 1/n * gaussian_kernel_sum

    return(Kck)

# Computes eq.(24):
def global_density(X):
    """
    X is a matrix (entire data set).

    Returns a vector of global densities for all observations, size(n,1).
    """

    (n, n_vars) = np.shape(X)

    Kck_matrix = np.zeros((n, n_vars))

    for variable in range(0, n_vars):

        Kck_matrix[:,variable] = np.reshape(variable_density(X[:,variable]), (n,))

    # Compute the global densities vector:
    Kc = np.zeros((n,1))

    K = 1

    for i in range(0,n):

        Kc[i] = K * np.prod(Kck_matrix[i,:])

    return(Kc)

# Computes eq.(25):
def observation_weights(X):
    """
    X is a matrix (entire data set).

    Returns a vector of global weightings for all observations, size(n,1).
    """

    (n, n_vars) = np.shape(X)

    W_c = np.zeros((n,1))

    Kc = global_density(X)

    Kc_inv = 1/Kc

    for i in range(0,n):

        W_c[i] = Kc_inv[i] / np.max(Kc_inv)

    return(W_c)
