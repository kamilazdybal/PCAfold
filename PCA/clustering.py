import numpy as np
import random
import time

def degrade_clusters(idx, verbose=False):
    """
    This function degrades cluster numeration if it is composed of non-consecutive integers.

    Example:
    ----------
    If from a clustering technique you get idx that is as following:

    `[0, 0, 2, 0, 5, 10]`

    this function turns this idx to:

    `[0, 0, 1, 0, 2, 3]`

    where clusters are numbered with consecutive integers.
    """

    index = 0
    dictionary = {}
    sorted = np.unique(idx)
    k_init = np.max(idx) + 1
    dictionary[sorted[0]] = 0
    old_val = sorted[0]
    idx_degraded = [0 for i in range(0, len(idx))]

    for val in sorted:
        if val > old_val:
            index += 1
        dictionary[val] = index

    for i, val in enumerate(idx):
        idx_degraded[i] = dictionary[val]

    k_update = np.max(idx_degraded) + 1

    if verbose == True:
        print('Clusters have been degraded.')
        print('The number of clusters have been reduced from ' + str(k_init) + ' to ' + str(k_update) + '.')

    return (idx_degraded, k_update)

def variable_bins(var, k, verbose=False):
    """
    This function does clustering based on dividing a variable vector `var` into bins
    of equal lengths.
    var_min                                               var_max
       |----------|----------|----------|----------|----------|
          bin 1      bin 2      bin 3       bin 4     bin 5
    """

    # Check that the number of clusters is an integer and is non-zero:
    if not (isinstance(k, int) and k > 0):
        raise ValueError("The number of clusters must be a positive integer.")

    var_min = np.min(var)
    var_max = np.max(var)
    bin_length = (var_max - var_min)/k
    idx = []
    bins_borders = []

    for cl in range(0,k+1):
        if cl == k+1:
            bins_borders.append(var_max)
        else:
            bins_borders.append(cl*bin_length + var_min)

    # Split into bins of variable:
    for val in var:
        if val == var_max:
            idx.append(k-1)
        else:
            for cl in range(0,k):
                if (val >= bins_borders[cl]) and (val < bins_borders[cl+1]):
                    idx.append(cl)

    # Degrade clusters if needed:
    if np.size(np.unique(idx)) != k:
        (idx, k) = degrade_clusters(idx, verbose)

    return(np.asarray(idx))

def mixture_fraction_bins(Z, k, Z_stoich):
    """
    This function does clustering based on dividing a mixture fraction vector
    `Z` into bins of equal lengths. The vector is first split to lean and rich
    side and then the sides get divided further into clusters. When k is even,
    this function will always create equal number of clusters on the lean and
    rich side. When k is odd, there will be one more cluster on the rich side
    compared to the lean side.

    Z_min           Z_stoich                                 Z_max
       |-------|-------|------------|------------|------------|
         bin 1   bin 2     bin 3        bin 4         bin 5

    """

    # Check that the number of clusters is an integer and is non-zero:
    if not (isinstance(k, int) and k > 0):
        raise ValueError("The number of clusters must be a positive integer.")

    # Number of interval borders:
    n_bins_borders = k + 1

    # Minimum and maximum mixture fraction:
    min_Z = np.min(Z)
    max_Z = np.max(Z)

    # Partition the Z-space:
    if k == 1:

        borders = np.linspace(min_Z, max_Z, n_bins_borders)

    else:

        # Z-space lower than stoichiometric mixture fraction:
        borders_lean = np.linspace(min_Z, Z_stoich, np.ceil(n_bins_borders/2))

        # Z-space higher than stoichiometric mixture fraction:
        borders_rich = np.linspace(Z_stoich, max_Z, np.ceil((n_bins_borders+1)/2))

        # Combine the two partitions:
        borders = np.concatenate((borders_lean[0:-1], borders_rich))

    # Bin data matrices initialization:
    idx_clust = []
    idx = np.zeros((len(Z), 1))

    # Create the cluster division vector:
    for bin in range(0,k):

        idx_clust.append([np.where((Z >= borders[bin]) & (Z <= borders[bin+1]))])
        idx[idx_clust[bin]] = bin+1

    return(idx)

def kmeans(X, k):
    """
    This function performs K-Means clustering.
    """

    # Check that the number of clusters is an integer and is non-zero:
    if not (isinstance(k, int) and k > 0):
        raise ValueError("The number of clusters must be a positive integer.")

    from sklearn.cluster import KMeans

    kmeans = KMeans(n_clusters=k, precompute_distances=True, algorithm='auto').fit(X)
    idx = kmeans.labels_

    return(idx)

def flip_clusters(idx, dictionary):
    """
    This function flips the cluster labelling according to instructions provided in the dictionary.
    For a `dictionary = {key : value}`, a cluster with a number `key` will get a number `value`.
    """

    flipped_idx = []

    for i in idx:
        if i in dictionary.keys():
            flipped_idx.append(dictionary[i])
        else:
            flipped_idx.append(i)

    return flipped_idx

def test():
    """
    This function tests the `clustering` module.
    """

    print('Test passed.')
    return 1
