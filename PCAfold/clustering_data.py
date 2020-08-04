import numpy as np
import copy

def _print_verbose_information(var, idx, bins_borders):

    k = len(np.unique(idx))

    print('Border values for bins:')
    print(bins_borders)
    print('')

    for cl in range(0,k):
        print("Bounds for cluster " + str(cl+1) + ":")
        print("\t" + str(round(np.min(var[np.argwhere(idx==cl)]), 4)) + ", " + str(round(np.max(var[np.argwhere(idx==cl)]), 4)))

def variable_bins(var, k, verbose=False):
    """
    This function does clustering by dividing a variable vector ``var`` into
    bins of equal lengths.

    :param var:
        vector of variable values.
    :param k:
        number of clusters to partition the data.
    :param verbose: (optional)
        boolean for printing clustering details.

    :raises ValueError:
        if number of clusters ``k`` is not a positive integer.

    :return:
        - **idx** - vector of cluster classifications.
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

    idx = np.asarray(idx)

    if verbose==True:
        _print_verbose_information(var, idx, bins_borders)

    return(idx)

def predefined_variable_bins(var, split_values, verbose=False):
    """
    This function does clustering by dividing a variable vector ``var`` into
    bins such that the split is done at values specified in the ``split_values``
    list. In general: ``split_values = [value_1, value_2, ..., value_n]``.

    *Note:*
    When a split is performed at a given ``value_i``, the observation in ``var``
    that takes exactly that value is assigned to the newly created bin.

    :param var:
        vector of variable values.
    :param split_values:
        list containing values at which the split to bins should be performed.
    :param verbose: (optional)
        boolean for printing clustering details.

    :raises ValueError:
        if any value within ``split_values`` is not within the range of
        vector ``var`` values.

    :return:
        - **idx** - vector of cluster classifications.
    """

    var_min = np.min(var)
    var_max = np.max(var)

    # Check that all values specified in `split_values` fall within the range of
    # the variable `var`:
    for value in split_values:
        if value < var_min or value > var_max:
            raise ValueError("Value " + str(value) + " is not within the range of the variable values.")

    idx = []
    bins_borders = copy.deepcopy(split_values)
    bins_borders.insert(0, var_min)
    bins_borders.append(var_max)

    k = len(bins_borders) - 1

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

    idx = np.asarray(idx)

    if verbose==True:
        _print_verbose_information(var, idx, bins_borders)

    return(idx)

def mixture_fraction_bins(Z, k, Z_stoich, verbose=False):
    """
    This function does clustering by dividing a mixture fraction vector
    ``Z`` into bins of equal lengths. The vector is first split to lean and rich
    side and then the sides get divided further into clusters. When ``k`` is even,
    this function will always create equal number of clusters on the lean and
    rich side. When ``k`` is odd, there will be one more cluster on the rich side
    compared to the lean side.

    :param Z:
        vector of mixture fraction values.
    :param k:
        number of clusters to partition the data.
    :param Z_stoich:
        stoichiometric mixture fraction.
    :param verbose: (optional)
        boolean for printing clustering details.

    :raises ValueError:
        if number of clusters ``k`` is not a positive integer.

    :return:
        - **idx** - vector of cluster classifications.
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
        borders_lean = np.linspace(min_Z, Z_stoich, int(np.ceil(n_bins_borders/2)))

        # Z-space higher than stoichiometric mixture fraction:
        borders_rich = np.linspace(Z_stoich, max_Z, int(np.ceil((n_bins_borders+1)/2)))

        # Combine the two partitions:
        borders = np.concatenate((borders_lean[0:-1], borders_rich))

    # Bin data matrices initialization:
    idx_clust = []
    idx = np.zeros((len(Z),))

    # Create the cluster division vector:
    for bin in range(0,k):

        idx_clust.append(np.where((Z >= borders[bin]) & (Z <= borders[bin+1])))
        idx[idx_clust[bin]] = bin+1

    idx = [int(i) for i in idx]

    # Degrade clusters if needed:
    if len(np.unique(idx)) != (np.max(idx)+1):
        (idx, k_new) = degrade_clusters(idx, verbose=False)

    idx = np.asarray(idx)

    if verbose==True:
        _print_verbose_information(Z, idx, borders)

    return(idx)

def pc_source_bins(pc_source, k, zero_offset_percentage=0.1, split_at_zero=False, verbose=False):
    """
    This function does clustering by dividing a PC-source vector
    ``pc_source`` into bins. By default, it finds one cluster between a negative and
    a positive offset from PC-source=0. The offset is computed from the input
    parameter ``zero_offset_percentage`` which specifies a percentage of the range
    ``pc_source_max - pc_source_min``. Further clusters are found by clustering
    positive and negative PC-sources alternatingly into bins of equal lengths.

    If ``split_at_zero`` is set to ``True``, the partitioning will always find one
    cluster that is between ``-offset`` and 0 and another cluster that is between
    0 and ``+offset``.

    Due to the nature of this clustering technique, the smallest allowed number
    of clusters is 3 if ``split_at_zero=False``. This is to assure that there are
    at least there three clusters: with high negative values, with close to zero
    values, with high positive values.

    If ``split_at_zero=True``, the smallest allowed number of clusters is 4. This
    is to assure that there are at least four clusters: with high negative
    values, with negative values close to zero, with positive values close to
    zero and with high positive values.

    :param pc_source:
        vector of variable values.
    :param k:
        number of clusters to partition the data.
        Cannot be smaller than 3 if ``split_at_zero=False`` or smaller
        than 4 if ``split_at_zero=True``.
    :param zero_offset_percentage: (optional)
        percentage of ``|pc_source_max - pc_source_min|`` to take as the
        ``offset`` value.
    :param split_at_zero: (optional)
        boolean specifying whether partitioning should be done at PC-source=0.
    :param verbose: (optional)
        boolean for printing clustering details.

    :raises ValueError:
        if number of clusters ``k`` is not an integer or smaller than 3 when
        ``split_at_zero=False`` or smaller than 4 when ``split_at_zero=True``.

    :raises ValueError:
        if PC-source vector ``pc_source`` has only non-negative or only
        non-positive values. For such vectors it is recommended to use
        ``predefined_variable_bins`` function instead.

    :raises ValueError:
        if the requested offset from zero crosses the minimum or maximum value
        of the PC-source vector ``pc_source``. If that is the case, it is
        recommended to lower the ``zero_offset_percentage`` value.

    :return:
        - **idx** - vector of cluster classifications.
    """

    # Check that the number of clusters is an integer and is larger than 2:
    if (not split_at_zero) and (not (isinstance(k, int) and k > 2)):
        raise ValueError("The number of clusters must be an integer not smaller than 3 when not splitting at zero.")

    # Check that the number of clusters is an integer and is larger than 2:
    if split_at_zero and (not (isinstance(k, int) and k > 3)):
        raise ValueError("The number of clusters must be an integer not smaller than 4 when splitting at zero.")

    pc_source_min = np.min(pc_source)
    pc_source_max = np.max(pc_source)
    pc_source_range = abs(pc_source_max - pc_source_min)
    offset = zero_offset_percentage * pc_source_range / 100

    # Basic checks on the PC-source vector:
    if not (pc_source_min <= 0):
        raise ValueError("PC-source vector does not have negative values. Use `predefined_variable_bins` as a clustering technique instead.")

    if not (pc_source_max >= 0):
        raise ValueError("PC-source vector does not have positive values. Use `predefined_variable_bins` as a clustering technique instead.")

    if (pc_source_min > -offset) or (pc_source_max < offset):
        raise ValueError("Offset from zero crosses the minimum or maximum value of the PC-source vector. Consider lowering `zero_offset_percentage`.")

    # Number of interval borders:
    if split_at_zero:
        n_bins_borders = k-1
    else:
        n_bins_borders = k

    # Generate cluster borders on the negative side:
    borders_negative = np.linspace(pc_source_min, -offset, int(np.ceil(n_bins_borders/2)))

    # Generate cluster borders on the positive side:
    borders_positive = np.linspace(offset, pc_source_max, int(np.ceil((n_bins_borders+1)/2)))

    # Combine the two partitions:
    if split_at_zero:
        borders = np.concatenate((borders_negative, np.array([0]), borders_positive))
    else:
        borders = np.concatenate((borders_negative, borders_positive))

    # Bin data matrices initialization:
    idx_clust = []
    idx = np.zeros((len(pc_source),))

    # Create the cluster division vector:
    for bin in range(0,k):

        idx_clust.append(np.where((pc_source >= borders[bin]) & (pc_source <= borders[bin+1])))
        idx[idx_clust[bin]] = bin+1

    idx = np.asarray([int(i) for i in idx])

    # Degrade clusters if needed:
    if len(np.unique(idx)) != (np.max(idx)+1):
        (idx, k_new) = degrade_clusters(idx, verbose=False)

    if verbose==True:
        _print_verbose_information(pc_source, idx, borders)

    return(idx)

def vqpca(X, k, n_pcs, scaling_criteria, idx_0=[], maximum_number_of_iterations=1000, verbose=False):
    """
    This function performs Vector Quantization clustering using
    Principal Component Analysis. VQPCA assigns observations to a particular
    cluster based on the minimum reconstruction error from PCA approximation
    with ``n_pcs`` number of Principal Components. This is an iterative
    procedure in which the reconstruction errors are evaluated for every
    observation as if that observation belonged to cluster *j* and next,
    the observation is assigned to that cluster for which the error was smallest.

    *Note:*
    VQPCA algorithm will center the global data set ``X`` by mean and scale by
    the scaling specified in the ``scaling_criteria`` parameter. Data in local
    clusters will be centered by the mean but will not be scaled.

    :param X:
        raw global data set, uncentered and unscaled.
    :param k:
        number of clusters to partition the data.
    :param n_pcs:
        number of Principal Components (PCs) that will be used to reconstruct the data
        at each iteration.
    :param scaling_criteria:
        scaling critertion for the global data set.
    :param idx_0: (optional)
        user-supplied initial ``idx`` for initializing the centroids. By default
        random intialization is performed.
    :param maximum_number_of_iterations: (optional)
        the maximum number of iterations that the algorithm will loop through.
    :param verbose: (optional)
        boolean for printing clustering details.

    :raises ValueError:
        if the number of observations in the data set ``X`` does not match the
        number of elements in the ``idx_0`` vector.

    :return:
        - **idx** - vector of cluster classifications.
    """

    import PCAfold.pca_impl as PCA
    import numpy.linalg
    from sklearn.decomposition import PCA as sklPCA

    (n_observations, n_variables) = np.shape(X)

    # Check that the provided idx_0 has the same number of entries as there are observations in X:
    if len(idx_0) > 0:
        if len(idx_0) != n_observations:
            raise ValueError("The number of observations in the data set `X` must match the number of elements in `idx_0` vector.")

    # Initialize the iteration counter:
    iteration = 0

    # Initialize the eigenvectors (Principal Components) matrix:
    eigenvectors = []

    # Initialize the scalings matrix:
    scalings = []

    # Global convergence parameter that represents convergence of reconstruction errors and centroids:
    convergence = 0

    # Initialize the reconstruction error:
    eps_rec = 1.0

    # Tolerance for division operations (to avoid division by zero):
    a_tol = 1.0e-16

    # Tolerance for the reconstruction error and for cluster centroids:
    r_tol = 1.0e-08

    # Populate the initial eigenvectors and scalings matrices (scalings will not
    # be updated later in the algorithm since we do not scale the data locally):
    for i in range(0,k):
        eigenvectors.append(np.eye(n_variables, n_pcs))
        scalings.append(np.ones((n_variables,)))

    # Center and scale the data:
    (X_pre_processed, _, _) = PCA.center_scale(X, scaling_criteria)

    # Initialization of cluster centroids:
    if len(idx_0) > 0:

        # If there is a user provided initial idx_0, find the initial centroids:
        centroids = get_centroids(X_pre_processed, idx_0)

    else:

        # Initialize centroids automatically as observations uniformly selected from X:
        centroids_indices = [int(i) for i in np.linspace(0, n_observations-1, k+2)]
        centroids_indices.pop()
        centroids_indices.pop(0)
        centroids = X_pre_processed[centroids_indices, :]

    # VQPCA algorithm:
    while ((convergence == 0) and (iteration <= maximum_number_of_iterations)):

        if verbose==True:
            print('\nIteration: ' + str(iteration) + '\n----------')

        # Initialize the reconstruction error matrix:
        sq_rec_err = np.zeros((n_observations, k))

        # Initialize the convergence of the cluster centroids:
        centroids_convergence = 0

        # Initialize the convergence of the reconstruction error:s
        eps_rec_convergence = 0

        # Reconstruct the data from the low-dimensional representation, evaluate the mean squared reconstruction error:
        for j in range(0,k):

            D = np.diag(scalings[j])
            C_mat = np.tile(centroids[j, :], (n_observations, 1))

            result = np.dot(numpy.linalg.inv(D), np.dot(eigenvectors[j], np.dot(np.transpose(eigenvectors[j]), D)))
            rec_err_os = (X_pre_processed - C_mat - np.dot((X_pre_processed - C_mat), result))
            sq_rec_err[:, j] = np.sum(rec_err_os**2, axis=1)

        # Assign the observations to clusters based on the minimum reconstruction error:
        idx = np.argmin(sq_rec_err, axis = 1)
        rec_err_min = np.min(sq_rec_err, axis = 1)
        rec_err_min_rel = rec_err_min.copy()

        # Evaluate the global mean reconstruction error (single value):
        eps_rec_new = np.mean(rec_err_min_rel)

        # Partition the data observations into clusters:
        (nz_X_k, nz_idx_clust, k) = get_partition(X_pre_processed, idx, verbose)

        # Evaluate the relative recontruction errors in each cluster:
        rec_err_min_rel_k = []

        for j in range(0,k):
            rec_err_min_rel_k.append(rec_err_min_rel[nz_idx_clust[j]])

        # Evaluate the mean reconstruction error in each cluster:
        eps_rec_new_clust = np.zeros(k)
        size_clust = np.zeros(k)

        for j in range(0,k):
            eps_rec_new_clust[j] = np.mean(rec_err_min_rel_k[j])
            size_clust[j] = len(nz_X_k[j])

        if verbose==True:
            print('Global mean recontruction error at iteration ' + str(iteration) + ' is ' + str(eps_rec_new) + '.\n')

        # Find the new cluster centroids:
        centroids_new = np.zeros((k, n_variables))

        for j in range(0,k):
            centroids_new[j, :] = np.mean(nz_X_k[j], axis=0)

        eps_rec_var = abs((eps_rec_new - eps_rec) / eps_rec_new)

        # Judge the convergence of errors:
        if (eps_rec_var < r_tol):
            eps_rec_convergence = 1

        # Judge the convergence of centroids:
        if (len(centroids) == len(centroids_new)):
            centroids_var = abs((centroids_new - centroids) / (centroids_new + a_tol))

            # If all elements in the C_var is less than the error tolerance:
            if (centroids_var < r_tol).all():
                centroids_convergence = 1

        # If the convergence of centroids and reconstruction error is reached, the algorithm stops:
        if ((iteration > 1) and (centroids_convergence == 1) and (eps_rec_convergence == 1)):
            convergence = 1
            if verbose==True:
                print('Convergence reached in iteration: ' + str(iteration) + '\n')
            break

        # Update recontruction error and cluster centroids:
        centroids = centroids_new.copy()
        eps_rec = eps_rec_new.copy()

        # Initialize the new eigenvectors matrix:
        eigenvectors = []

        # Perform PCA in local clusters to update the eigenvectors:
        for j in range(0,k):

            # Perform PCA:
            centered_nz_X_k = nz_X_k[j] - np.mean(nz_X_k[j], axis=0)
            covariance_matrix = np.dot(np.transpose(centered_nz_X_k), centered_nz_X_k) / (n_observations-1)
            L, PCs = numpy.linalg.eig(covariance_matrix)
            PCs = np.real(PCs)

            eigenvectors.append(PCs[:,0:n_pcs])

            if verbose==True:
                print('Cluster ' + str(j) + ' dimensions:')
                print(np.shape(nz_X_k[j]))

        # Increment the iteration counter:
        iteration = iteration + 1

    if (convergence == 0):
        print('Convergence not reached in ' + str(iteration) + ' iterations.')

    # Degrade clusters if needed:
    if len(np.unique(idx)) != (np.max(idx)+1):
        (idx, k_new) = degrade_clusters(idx, verbose=False)

    # Check that the number of entries inside idx is the same as the number of observations:
    if len(idx) != n_observations:
        raise ValueError("The number of entires inside `idx` is not equal to the number of observations in the data set `X`.")

    return(np.asarray(idx))

def degrade_clusters(idx, verbose=False):
    """
    This function renumerates clusters if either of these two cases is true:

    - ``idx`` is composed of non-consecutive integers, or

    - the smallest cluster number in ``idx`` is not equal to ``0``.

    **Example:**

    Starting with an ``idx`` that is the following:
    ``[0, 0, 2, 0, 5, 10]`` this function turns this ``idx`` to:
    ``[0, 0, 1, 0, 2, 3]``, where clusters are numbered with consecutive integers.

    Alternatively, if ``idx`` is: ``[1, 1, 2, 2, 3, 3]`` this function turns
    this ``idx`` to: ``[0, 0, 1, 1, 2, 2]`` so that the smallest cluster number
    is equal to ``0``.

    :param idx:
        raw vector of cluster classifications.
    :param verbose: (optional)
        boolean for printing clustering details.

    :return:
        - **idx_degraded** degraded vector of cluster classifications. The first cluster has index 0.
        - **k_update** - the updated number of clusters.
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

    return (np.asarray(idx_degraded), k_update)

def flip_clusters(idx, dictionary):
    """
    This function flips the cluster labelling according to instructions provided
    in the dictionary. For a ``dictionary = {key : value}``, a cluster with a
    number ``key`` will get a number ``value``.

    :param idx:
        vector of cluster classifications.
        The first cluster has index 0.
    :param dictionary:
        a dictionary specifying the cluster numeration flipping instructions.

    :return:
        - **flipped_idx** - vector of cluster classifications. The first cluster has index 0.
    """

    flipped_idx = []

    for i in idx:
        if i in dictionary.keys():
            flipped_idx.append(dictionary[i])
        else:
            flipped_idx.append(i)

    return(np.asarray(flipped_idx))

def get_centroids(X, idx):
    """
    This function computes the centroids for the clustering specified in the
    ``idx`` vector.

    :param X:
        data set for computing the cluster centroids.
    :param idx:
        vector of cluster classifications.
        The first cluster has index 0.

    :raises ValueError:
        if the number of observations in the data set ``X`` does not match the
        number of elements in the ``idx`` vector.

    :return:
        - **centroids** - matrix of cluster centroids. It has size ``k`` times number of observations.
    """

    # Degrade clusters if needed:
    if len(np.unique(idx)) != (np.max(idx)+1):
        (idx, k_new) = degrade_clusters(idx, verbose=False)

    (n_observations, n_variables) = np.shape(X)

    # Check if the number of indices in `idx` is the same as the number of observations in a data set:
    if n_observations != len(idx):
        raise ValueError("The number of observations in the data set `X` must match the number of elements in `idx` vector.")

    # Find the number of clusters:
    k = len(np.unique(idx))

    # Initialize the centroids matrix:
    centroids = np.zeros((k, n_variables))

    # Compute the centroids:
    for i in range(0,k):
        indices = [ind for ind, e in enumerate(idx) if e == i]
        centroids[i, :] = np.mean(X[indices,:], axis=0)

    return(centroids)

def get_partition(X, idx, verbose=False):
    """
    This function partitions the observations from the original global data
    set ``X`` into local clusters according to ``idx`` provided. It returns a
    tuple of three variables ``(data_in_clusters, data_idx_in_clusters, k_new)``,
    where ``data_in_clusters`` are the original observations from ``X`` divided
    into clusters, ``data_idx_in_clusters`` are the indices of the original
    observations divided into clusters. If any cluster is empty or has less
    observations assigned to it that the number of variables, that cluster will
    be removed and the observations that were assigned to it will not appear
    in ``data_in_clusters`` nor in ``data_idx_in_clusters``. The new number of
    clusters ``k_new`` is computed taking into account any possibly removed
    clusters.

    :param X:
        data set to partition.
    :param idx:
        vector of cluster classifications.
        The first cluster has index 0.
    :param verbose: (optional)
        boolean for printing details.

    :return:
        - **data_in_clusters** - list of ``k_new`` arrays that contains original data set observations in each cluster.
        - **data_idx_in_clusters** - list of ``k_new`` arrays that contains indices of the original data set observations in each cluster.
        - **k_new** - the updated number of clusters.
    """

    try:
        (n_observations, n_variables) = np.shape(X)
    except:
        (n_observations, ) = np.shape(X)
        n_variables = 1

    # Remove empty clusters from indexing:
    if len(np.unique(idx)) != (np.max(idx)+1):
        (idx, _) = degrade_clusters(idx, verbose)
        if verbose==True:
            print('Empty clusters will be removed.')

    k = len(np.unique(idx))

    idx_clust = []
    n_points = np.zeros(k)
    data_in_clusters = []
    data_idx_in_clusters = []

    for i in range(0,k):

        indices_to_append = np.argwhere(idx==i).ravel()
        idx_clust.append(indices_to_append)
        n_points[i] = len(indices_to_append)

        if ((n_points[i] < n_variables) and (n_points[i] > 0)):
            if verbose==True:
                print('Too few points (' + str(int(n_points[i])) + ') in cluster ' + str(i) + ', cluster will be removed.')

    # Find those cluster numbers where the number of observations is not less than number of variables:
    nz_idx = np.argwhere(n_points >= n_variables).ravel()

    # Compute the new number of clusters taking into account removed clusters:
    k_new = len(nz_idx)

    for i in range(0,k_new):

        # Assign observations to clusters:
        data_idx_in_clusters.append(idx_clust[nz_idx[i]])
        data_in_clusters.append(X[data_idx_in_clusters[i],:])

    return(data_in_clusters, data_idx_in_clusters, k_new)

def get_populations(idx, verbose=False):
    """
    This function computes populations (number of observations) in clusters
    specified in the ``idx`` vector. As an example, if there are 100
    observations in the first cluster and 500 observations in the second cluster
    this function will return a list: ``[100, 500]``.

    :param idx:
        vector of cluster classifications.
        The first cluster has index 0.
    :param verbose: (optional)
        boolean for printing details.

    :return:
        - **populations** - list of cluster populations. Each entry referes to one cluster ordered according to ``idx``.
    """

    populations = []

    # Degrade clusters if needed:
    if len(np.unique(idx)) != (np.max(idx)+1):
        (idx, k_new) = degrade_clusters(idx, verbose)

    # Find the number of clusters:
    k = len(np.unique(idx))

    for i in range(0,k):

        populations.append((idx==i).sum())

    return(populations)
