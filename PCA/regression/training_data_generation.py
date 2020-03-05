import numpy as np
import random
import PCA.clustering

def train_test_split_fixed_number_from_idx(idx, perc, test_selection_option=1, verbose=False):
    """
    This function takes an `idx` classifications from a clustering technique and samples
    a fixed number `n_of_samples` of observations from every cluster as training data.

    `n_of_samples` is estimated based on the percentage provided.
    First, the total number of samples for training is estimated as a percentage `perc`
    from the total number of observations `n_obs`.
    Next, this number is devided equally into `k` clusters.

    There is a bar that no more than 50% of observations from any cluster
    will be taken for training. This is to avoid that too little samples
    will remain for test data from small clusters.

    Test data is then drawn from every cluster equally in a similar way,
    in a quantity equal to the number of remaining samples from the smallest cluster.

    Example:
    ----------
    If the full data has 10 observations with indices:

    `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]`

    and a clustering technique divided the observations into clusters in the following way:

    `[1, 1, 1, 1, 1, 1, 2, 2, 2, 2]`

    Then, if you request 40% of the data for training, the function may return:

    `idx_train = [0, 1, 7, 9]`

    as the training data (where observations 0 and 1 belong to cluster 1
    and observations 7 and 9 belong to cluster 2).

    Test data may then become:

    `idx_test = [3, 5, 6, 8]`

    Two options for sampling test data are implemented.
    If you select `test_selection_option=1`, all remaining samples
    that were not taken as training data become the test data.
    If you select `test_selection_option=2`, the smallest cluster is found and
    the remaining number of observations are taken as test data in that cluster.
    Next, the same number of observations is taken from all remaining larger clusters.

    Input:
    ----------
    `idx`         - vector of indices classifying observations to clusters.
    `perc`        - percentage of data to be selected as training data from each cluster.
                    Set perc=20 if you want 20%.
    `test_selection_option`
                  - select 1 if you want all remaining samples to become test data.
                    Select 2 if you want the same number of samples from each cluster to become test data.
    `verbose`     - boolean for printing clustering details.

    Output:
    ----------
    `idx_train`   - indices of the training data.
    `idx_test`    - indices of the (remaining) test data.
    """

    n_obs = len(idx)

    # Degrade clusters if needed:
    if len(np.unique(idx)) != (np.max(idx)-1):
        (idx, k_new) = PCA.clustering.degrade_clusters(idx, verbose=False)

    # Vector of indices 0..n_obs:
    idx_full = np.arange(0,n_obs)
    idx_train = []
    idx_test = []
    cluster_test = []

    # Find the number of clusters:
    k = np.size(np.unique(idx))

    # Fixed number of samples that will be taken from every cluster as the training data:
    n_of_samples = int(perc*n_obs/k/100)

    if verbose == True:
        print("The number of samples that will be select from each cluster is " + str(n_of_samples) + ".\n")

    smallest_cluster_size = n_obs

    # Get clusters and split them into training and test indices:
    for cl in range(0,k):
        cluster = []
        for i, id in enumerate(idx):
            if id == cl:
                cluster.append(idx_full[i])

        if len(cluster) < smallest_cluster_size:
            smallest_cluster_size = len(cluster)

        # Selection of training data:
        if int(0.5*len(cluster)) < n_of_samples:
            cluster_train = np.array(random.sample(cluster, int(0.5*len(cluster))))

            if verbose == True:
                print("Cluster " + str(cl+1) + ": taking " + str(int(0.5*len(cluster))) + " training samples out of " + str(len(cluster)) + " observations (%.1f" % (int(0.5*len(cluster))/len(cluster)*100) + "%).")
        else:
            cluster_train = np.array(random.sample(cluster, n_of_samples))

            if verbose == True:
                print("Cluster " + str(cl+1) + ": taking " + str(n_of_samples) + " training samples out of " + str(len(cluster)) + " observations (%.1f" % (n_of_samples/len(cluster)*100) + "%).")

        idx_train = np.concatenate((idx_train, cluster_train))

        # Selection of testing data - all data that remains is test data:
        if test_selection_option == 1:
            cluster_test = np.setdiff1d(cluster, cluster_train)
            idx_test = np.concatenate((idx_test, cluster_test))

        # Selection of testing data - equal samples from each cluster:
        if test_selection_option == 2:
            cluster_test.append(np.setdiff1d(cluster, cluster_train))

    if test_selection_option == 2:
        minimum_test_samples = n_obs
        for cl in range(0,k):
            if len(cluster_test[cl]) < minimum_test_samples:
                minimum_test_samples = len(cluster_test[cl])

        for cl in range(0,k):
            idx_test = np.concatenate((idx_test, random.sample(list(cluster_test[cl]), minimum_test_samples)))

            if verbose == True:
                print("Cluster " + str(cl+1) + ": taking " + str(minimum_test_samples) + " test samples out of " + str(len(cluster_test[cl])) + " remaining observations (%.1f" % (minimum_test_samples/len(cluster_test[cl])*100) + "%).")

    if verbose == True:
        print('\nSelected ' + str(np.size(idx_train)) + ' training samples (%.1f' % (np.size(idx_train)*100/n_obs) + '%) and ' + str(np.size(idx_test)) + ' test samples (%.1f' % (np.size(idx_test)*100/n_obs) + '%).\n')

    if (test_selection_option == 1) & (np.size(idx_test) + np.size(idx_train) != n_obs):
        raise ValueError("Sizes of train and test data do not sum up to the total number of observations.")

    idx_train = np.sort(idx_train.astype(int))
    idx_test = np.sort(idx_test.astype(int))

    return (idx_train, idx_test)

def train_test_split_percentage_from_idx(idx, perc, verbose=False):
    """
    This function takes an `idx` classifications from a clustering technique and samples
    a certain percentage `perc` from every cluster as the training data.
    The remaining percentage is the test data.

    For example, if the full data has 10 observations with indices:

    `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]`

    and you requested 40% of the data to be training data, the function may return:

    `idx_train = [0, 1, 2, 7]`

    and:

    `idx_test = [3, 4, 5, 6, 8, 9]`

    as the remaining 60% test data.

    Input:
    ----------
    `idx`         - vector of indices classifying observations to clusters.
    `perc`        - percentage of data to be selected as training data from each cluster.
                    Set perc=20 if you want 20%.
    `verbose`     - boolean for printing clustering details.

    Output:
    ----------
    `idx_train`   - indices of the training data.
    `idx_test`    - indices of the (remaining) test data.
    """

    n_obs = len(idx)
    idx_full = np.arange(0,n_obs)
    idx_train = []
    idx_test = []

    k = np.size(np.unique(idx))

    # Get clusters and split them into training and test indices:
    for cl in range(0,k):
        cluster = []
        for i, id in enumerate(idx):
            if id == cl:
                cluster.append(idx_full[i])

        if verbose == True:
            print('Number of observations in cluster ' + str(cl) + ': ' + str(len(cluster)))

        cluster_train = np.array(random.sample(cluster, int(len(cluster)*perc/100)))
        cluster_test = np.setdiff1d(cluster, cluster_train)

        idx_train = np.concatenate((idx_train, cluster_train))
        idx_test = np.concatenate((idx_test, cluster_test))

    if verbose == True:
        print('\nSelected ' + str(np.size(idx_train)) + ' training samples (' + str(perc) + '%) and ' + str(np.size(idx_test)) + ' test samples (' + str(100-perc) + '%).\n')

    if np.size(idx_test) + np.size(idx_train) != n_obs:
        raise ValueError("Size of train and test data do not sum up to the total number of observations.")

    idx_train = np.sort(idx_train.astype(int))
    idx_test = np.sort(idx_test.astype(int))

    return (idx_train, idx_test)

def train_test_split_random(n_obs, perc, idx_test=[], verbose=False):
    """
    This function splits dataset into training and testing using random sampling.

    For example if the full data has 10 observations whose indices are:

    `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]`

    and you request 40% of the data to be training data, the function may return:

    `idx_train = [0, 1, 2, 7]`

    and:

    `idx_test = [3, 4, 5, 6, 8, 9]`

    as the remaining 60% test data.

    Input:
    ----------
    `n_obs`       - number of observations in the original data set.
    `perc`        - percentage of data to be selected as training data from each cluster.
                    Set perc=20 if you want 20%.
    `idx_test`    - are the user-provided indices for test data. If specified,
                    the training data will be selected ignoring the indices in `idx_test`
                    and the test data will be return as the user-provided.
                    If not specified, all remaining samples become test data.
    `verbose`     - boolean for printing clustering details.

    Output:
    ----------
    `idx_train`   - indices of the training data.
    `idx_test`    - indices of the (remaining) test data.
    """

    idx_full = np.arange(0,n_obs)
    idx_test = np.array(idx_test)

    if len(idx_test) != 0:
        idx_full_no_test = np.setdiff1d(idx_full, idx_test)
        if int(len(idx_full)*perc/100) <= len(idx_full_no_test):
            idx_train = np.array(random.sample(idx_full_no_test.tolist(), int(len(idx_full)*perc/100)))
        else:
            raise ValueError("The training percentage specified is too high, there aren't enough samples.")
    else:
        idx_train = np.array(random.sample(idx_full.tolist(), int(len(idx_full)*perc/100)))
        idx_test = np.setdiff1d(idx_full, idx_train)

    n_train = len(idx_train)
    n_test = len(idx_test)

    if verbose == True:
        print('Selected ' + str(np.size(idx_train)) + ' training samples (' + str(round(n_train/n_obs*100, 1)) + '%) and ' + str(np.size(idx_test)) + ' test samples (' + str(round(n_test/n_obs*100,1)) + '%).\n')

    if len(idx_test) == 0 and (np.size(idx_test) + np.size(idx_train) != n_obs):
        raise ValueError("Size of train and test data do not sum up to the total number of observations.")

    idx_train = np.sort(idx_train.astype(int))
    idx_test = np.sort(idx_test.astype(int))

    return (idx_train, idx_test)

def test():
    """
    This function tests the `training_data_generation` module.
    """

    try:
        idx = [1, 1, 1, 1, 1, 1, 2, 2, 2, 2]
        (idx_train, idx_test) = train_test_split_fixed_number_from_idx(idx, 40, test_selection_option=1, verbose=False)
    except Exception:
        print('Test of train_test_split_fixed_number_from_idx failed.')
        return 0

    try:
        (idx_train, idx_test) = train_test_split_random(10, 40, idx_test=[1,2], verbose=False)
    except Exception:
        print('Test of train_test_split_random failed.')
        return 0

    try:
        (idx_train, idx_test) = train_test_split_random(10, 40, idx_test=[1,2,3,4,5,6], verbose=False)
    except Exception:
        print('Test of train_test_split_random failed.')
        return 0

    try:
        (idx_train, idx_test) = train_test_split_random(10, 40, idx_test=[1,2,3,4,5,6,7], verbose=False)
        print('Test of train_test_split_random failed.')
        return 0
    except Exception:
        pass

    print("Test passed.")
    return 1
