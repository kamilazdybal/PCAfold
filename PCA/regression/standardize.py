import numpy as np

def uncenter_unscale(scaled_data, centers, scales):
    """
    This function uncenters and unscales data.

    Input:

    `scaled_data` is the data scaled by none
    `data` is the original data

    Output:

    `uu_data` is the unscaled original data
    """

    uu_data = scaled_data * scales + centers

    return uu_data

def none(data):
    """
    This function does not center or scale data. Just a placeholder.
    """

    centers = np.zeros((np.size(data,axis=1),))
    scales = np.ones((np.size(data,axis=1),))

    return (data, centers, scales)

def mean_center(data):
    """
    This function centers data by mean.
    """

    centers = np.mean(data, axis=0)
    scales = np.ones((np.size(data,axis=1),))
    scaled_data = (data - centers)/scales

    return (scaled_data, centers, scales)

def zero_one(data):
    """
    This function scales data to be in range 0 to 1.
    """

    centers = np.min(data, axis=0)
    scales = (np.max(data, axis=0) - np.min(data, axis=0))
    scaled_data = (data - centers)/scales

    return (scaled_data, centers, scales)

def minus_one_one(data):
    """
    This function scales data to be in range -1 to 1.
    """

    centers = 0.5*(np.max(data, axis=0) + np.min(data, axis=0))
    scales = 0.5*(np.max(data, axis=0) - np.min(data, axis=0))
    scaled_data = (data - centers)/scales

    return (scaled_data, centers, scales)

def z_score(data):
    """
    This function standardizes data with z-score (centering by mean, scaling by std).

    The same can be achieved using:

    ```
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    ```
    """

    centers = np.mean(data, axis=0)
    scales = np.std(data, axis=0)
    scaled_data = (data - centers)/scales

    return (scaled_data, centers, scales)

def std(data):
    """
    This function scales data by std.
    """

    centers = np.zeros((np.size(data,axis=1),))
    scales = np.std(data, axis=0)
    scaled_data = (data - centers)/scales

    return (scaled_data, centers, scales)

def mean(data):
    """
    This function scales data by mean.
    """

    centers = np.zeros((np.size(data,axis=1),))
    scales = np.mean(data, axis=0)
    scaled_data = (data - centers)/scales

    return (scaled_data, centers, scales)

def test():
    """
    Performs regression testing of the standardizing functions.

    Tests are performed on the Iris flower data set.
    """

    import pandas as pd
    from sklearn.datasets import load_iris
    import PCA.post_processing as pp

    # Import data:
    Xdf = pd.DataFrame(load_iris().data)
    X = Xdf.to_numpy()
    (n_obs, n_vars) = np.shape(X)

    bound_tolerance = 10**-12
    error_tolerance = 10**-12

    # Test minus_one_one:
    (Xs, centers, scales) = minus_one_one(X)
    min_Xs = np.min(Xs, axis=0)
    max_Xs = np.max(Xs, axis=0)
    X_uu = uncenter_unscale(Xs, centers, scales)
    nrmse = pp.nrmse(X, X_uu)
    r2 = pp.r2(X, X_uu)

    if (r2 < 1 - error_tolerance) or (nrmse > error_tolerance):
        print("Test of minus_one_one function failed.")
        return 0

    for i in range(0, n_vars):
        if (min_Xs[i] < -1-bound_tolerance or min_Xs[i] > -1+bound_tolerance) or (max_Xs[i] < 1-bound_tolerance or max_Xs[i] > 1+bound_tolerance):
            print("Test of minus_one_one function failed.")
            return 0

    # Test zero_one:
    (Xs, centers, scales) = zero_one(X)
    min_Xs = np.min(Xs, axis=0)
    max_Xs = np.max(Xs, axis=0)
    X_uu = uncenter_unscale(Xs, centers, scales)
    nrmse = pp.nrmse(X, X_uu)
    r2 = pp.r2(X, X_uu)

    if (r2 < 1 - error_tolerance) or (nrmse > error_tolerance):
        print("Test of zero_one function failed.")
        return 0

    for i in range(0, n_vars):

        if (min_Xs[i] < -bound_tolerance or min_Xs[i] > bound_tolerance) or (max_Xs[i] < 1-bound_tolerance or max_Xs[i] > 1+bound_tolerance):
            print("Test of zero_one function failed.")
            return 0

    # Test std:
    (Xs, centers, scales) = std(X)
    min_Xs = np.min(Xs, axis=0)
    max_Xs = np.max(Xs, axis=0)
    X_uu = uncenter_unscale(Xs, centers, scales)
    nrmse = pp.nrmse(X, X_uu)
    r2 = pp.r2(X, X_uu)

    if (r2 < 1 - error_tolerance) or (nrmse > error_tolerance):
        print("Test of std function failed.")
        return 0

    # Test mean:
    (Xs, centers, scales) = mean(X)
    min_Xs = np.min(Xs, axis=0)
    max_Xs = np.max(Xs, axis=0)
    X_uu = uncenter_unscale(Xs, centers, scales)
    nrmse = pp.nrmse(X, X_uu)
    r2 = pp.r2(X, X_uu)

    if (r2 < 1 - error_tolerance) or (nrmse > error_tolerance):
        print("Test of mean function failed.")
        return 0

    # Test z_score:
    (Xs, centers, scales) = z_score(X)
    min_Xs = np.min(Xs, axis=0)
    max_Xs = np.max(Xs, axis=0)
    X_uu = uncenter_unscale(Xs, centers, scales)
    nrmse = pp.nrmse(X, X_uu)
    r2 = pp.r2(X, X_uu)

    if (r2 < 1 - error_tolerance) or (nrmse > error_tolerance):
        print("Test of z_score function failed.")
        return 0

    # Test mean_center:
    (Xs, centers, scales) = mean_center(X)
    min_Xs = np.min(Xs, axis=0)
    max_Xs = np.max(Xs, axis=0)
    X_uu = uncenter_unscale(Xs, centers, scales)
    nrmse = pp.nrmse(X, X_uu)
    r2 = pp.r2(X, X_uu)

    if (r2 < 1 - error_tolerance) or (nrmse > error_tolerance):
        print("Test of mean_center function failed.")
        return 0

    print("Test passed.")

    return 1
