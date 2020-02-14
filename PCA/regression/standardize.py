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

    centers = np.zeros(np.size(data))
    scales = np.ones(np.size(data))

    return (data, centers, scales)

def mean_center(data):
    """
    This function centers data by mean.
    """

    centers = np.mean(data, axis=0)
    scales = np.ones(np.size(data))
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

    centers = np.zeros(np.size(data))
    scales = np.std(data, axis=0)
    scaled_data = (data - centers)/scales

    return (scaled_data, centers, scales)

def mean(data):
    """
    This function scales data by mean.
    """

    centers = np.zeros(np.size(data))
    scales = np.mean(data, axis=0)
    scaled_data = (data - centers)/scales

    return (scaled_data, centers, scales)
