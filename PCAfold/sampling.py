import numpy as np
import random
from PCAfold import clustering_data

def _perform_checks(idx, idx_test, bar_50, random_seed, verbose):
    """
    This private function performs basic checks on the input parameters for the
    ``TrainTestSelect`` class.

    It will run at each class initialization.

    :param idx:
        vector of cluster classifications.
    :param idx_test: (optional)
        are the user-provided indices for test data. If specified, the training
        data will be selected ignoring the indices in ``idx_test`` and the test
        data will be returned the same as the user-provided ``idx_test``.
        If not specified, all remaining samples become test data.
    :param bar_50: (optional)
        boolean specifying whether the 50% bar should apply.
    :param random_seed: (optional)
        integer specifying random seed for random sample selection.
    :param verbose: (optional)
        boolean for printing sampling details.

    :raises ValueError:
        if ``idx`` vector has length zero.
    :raises ValueError:
        if ``idx_test`` vector has more observations than ``idx``.
    :raises ValueError:
        if ``bar_50`` is not a boolean.
    :raises ValueError:
        if ``random_seed`` is not an integer.
    :raises ValueError:
        if ``verbose`` is not a boolean.
    """

    if len(idx) == 0:
        raise ValueError("Parameter `idx` has length zero.")

    if len(idx_test) > len(idx):
        raise ValueError("Parameter `idx_test` has more observations than `idx`.")

    if not isinstance(bar_50, bool):
        raise ValueError("Parameter `bar_50` has to be a boolean.")

    if random_seed != None:
        if not isinstance(random_seed, int):
            raise ValueError("Parameter `random_seed` has to be an integer.")

    if not isinstance(verbose, bool):
        raise ValueError("Parameter `verbose` has to be a boolean.")

def _print_verbose_information(idx, idx_train, idx_test):
    """
    This private function prints detailed information on train and test sampling when
    ``verbose=True``.

    :param idx:
        vector of cluster classifications.
    :param idx_train:
        indices of the train data.
    :param idx_test:
        indices of the test data.
    """

    cluster_populations = clustering_data.get_populations(idx)
    k = np.size(np.unique(idx))
    n_observations = len(idx)

    for cl_id in range(0,k):
        train_indices = [t_id for t_id in idx_train if idx[t_id,]==cl_id]
        if cluster_populations[cl_id] != 0:
            print("Cluster " + str(cl_id+1) + ": taking " + str(len(train_indices)) + " train samples out of " + str(cluster_populations[cl_id]) + " observations (%.1f" % (len(train_indices)/(cluster_populations[cl_id])*100) + "%).")
        else:
            print("Cluster " + str(cl_id+1) + ": taking " + str(len(train_indices)) + " train samples out of " + str(cluster_populations[cl_id]) + " observations (%.1f" % (0) + "%).")
    print("")

    for cl_id in range(0,k):
        train_indices = [t_id for t_id in idx_train if idx[t_id,]==cl_id]
        test_indices = [t_id for t_id in idx_test if idx[t_id,]==cl_id]
        if (cluster_populations[cl_id] - len(train_indices)) != 0:
            print("Cluster " + str(cl_id+1) + ": taking " + str(len(test_indices)) + " test samples out of " + str(cluster_populations[cl_id] - len(train_indices)) + " remaining observations (%.1f" % (len(test_indices)/(cluster_populations[cl_id] - len(train_indices))*100) + "%).")
        else:
            print("Cluster " + str(cl_id+1) + ": taking " + str(len(test_indices)) + " test samples out of " + str(cluster_populations[cl_id] - len(train_indices)) + " remaining observations (%.1f" % (0) + "%).")

    print('\nSelected ' + str(np.size(idx_train)) + ' train samples (%.1f' % (np.size(idx_train)*100/n_observations) + '%) and ' + str(np.size(idx_test)) + ' test samples (%.1f' % (np.size(idx_test)*100/n_observations) + '%).\n')

class TrainTestSelect:
    """
    This class enables selecting train and test data samples.

    :param idx:
        vector of cluster classifications.
    :param idx_test: (optional)
        are the user-provided indices for test data. If specified, the training
        data will be selected ignoring the indices in ``idx_test`` and the test
        data will be returned the same as the user-provided ``idx_test``.
        If not specified, test samples will be selected according to a
        method specified by ``test_selection_option`` parameter.
    :param bar_50: (optional)
        boolean specifying whether the 50% bar should apply.
    :param random_seed: (optional)
        integer specifying random seed for random sample selection.
    :param verbose: (optional)
        boolean for printing sampling details.

    :raises ValueError:
        if ``idx`` vector has length zero.
    :raises ValueError:
        if ``idx_test`` vector has more observations than ``idx``.
    :raises ValueError:
        if ``bar_50`` is not a boolean.
    :raises ValueError:
        if ``random_seed`` is not an integer.
    :raises ValueError:
        if ``verbose`` is not a boolean.
    """

    def __init__(self, idx, idx_test=[], bar_50=True, random_seed=None, verbose=False):

        self.idx = idx
        self.idx_test = idx_test
        self.bar_50 = bar_50
        self.random_seed = random_seed
        self.verbose = verbose

        _perform_checks(self.idx, self.idx_test, self.bar_50, self.random_seed, self.verbose)

    def number(self, perc, test_selection_option=1):
        """
        This function takes an ``idx`` classifications from a clustering technique
        and samples a fixed number of observations from every cluster as training
        data. This in general results in under-representing large cluster and
        over-representing small clusters.

        The number of samples is estimated based on the percentage ``perc`` provided.
        First, the total number of samples for training is estimated as a percentage
        ``perc`` from the total number of observations ``n_observations``.
        Next, this number is divided equally into ``k`` clusters.

        By default, there is a bar that no more than 50% of observations from any
        cluster will be taken for training. This is to avoid that too little samples
        will remain for test data from small clusters. If the parameter ``bar_50`` is
        set to False, this function will allow to sample more than 50% of
        observations.

        Test data is then drawn from every cluster equally in a similar way, in a
        quantity equal to the number of remaining samples from the smallest cluster.

        **Example:**

        If the full data has 10 observations with indices:

        ``[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]``

        and a clustering technique divided the observations into clusters in the
        following way:

        ``[0, 0, 0, 0, 0, 0, 1, 1, 1, 1]``

        Then, if you request 40% of the data for training, the function may return:

        ``idx_train = [0, 1, 7, 9]``

        as the training data (where observations 0 and 1 belong to the first cluster
        and observations 7 and 9 belong to the second cluster).

        Test data may then become:

        ``idx_test = [3, 5, 6, 8]``

        Two options for sampling test data are implemented. If you select
        ``test_selection_option=1``, all remaining samples that were not taken as
        training data become the test data. If you select ``test_selection_option=2``,
        the smallest cluster is found and the remaining number of observations are
        taken as test data in that cluster. Next, the same number of observations is
        taken from all remaining larger clusters.

        :param perc:
            percentage of data to be selected as training data from each cluster.
            For instance, set ``perc=20`` if you want to select 20%.
        :param test_selection_option: (optional)
            option for how the test data is selected.
            Select ``test_selection_option=1`` if you want all remaining samples
            to become test data.
            Select ``test_selection_option=2`` if you want the same number of samples
            from each cluster to become test data.

        :raises ValueError:
            if ``test_selection_option`` is not equal to 1 or 2.

        :raises ValueError:
            if the perecentage specified is too high in combination with the
            user-provided ``idx_test`` vector and there aren't enough samples to
            select as train data.

        :raises ValueError:
            if the size of ``idx_train`` and the size of ``idx_test`` do not sum up
            to ``n_observations``. *This is for unit testing only.*

        :return:
            - **idx_train** - indices of the train data.
            - **idx_test** - indices of the test data.
        """

        # Check that `test_selection_option` parameter was passed correctly:
        _test_selection_option = [1,2]
        if test_selection_option not in _test_selection_option:
            raise ValueError("Test selection option can only be 1 or 2.")

        n_observations = len(self.idx)

        # Degrade clusters if needed:
        if len(np.unique(self.idx)) != (np.max(self.idx)+1):
            (self.idx, _) = clustering_data.degrade_clusters(self.idx, verbose=False)

        # Vector of indices 0..n_observations:
        idx_full = np.arange(0, n_observations)
        idx_train = []
        idx_test = []
        cluster_test = []

        # Find the number of clusters:
        k = np.size(np.unique(self.idx))

        # Fixed number of samples that will be taken from every cluster as the training data:
        n_of_samples = int(perc*n_observations/k/100)

        smallest_cluster_size = n_observations

        # Get clusters and split them into training and test indices:
        for cl_id in range(0,k):

            if self.random_seed != None:
                random.seed(self.random_seed)

            cluster = []
            for i, id in enumerate(self.idx):
                if id == cl_id:
                    cluster.append(idx_full[i])

            if len(cluster) < smallest_cluster_size:
                smallest_cluster_size = len(cluster)

            # Selection of training data:
            if int(0.5*len(cluster)) < n_of_samples:

                # If the 50% bar should apply, take only 50% of cluster observations:
                cluster_train = np.array(random.sample(cluster, int(0.5*len(cluster))))

            else:

                # Otherwise take the calculated number of samples:
                cluster_train = np.array(random.sample(cluster, n_of_samples))

            idx_train = np.concatenate((idx_train, cluster_train))

            # Selection of testing data - all data that remains is test data:
            if test_selection_option == 1:
                cluster_test = np.setdiff1d(cluster, cluster_train)
                idx_test = np.concatenate((idx_test, cluster_test))

            # Selection of testing data - equal samples from each cluster:
            if test_selection_option == 2:
                cluster_test.append(np.setdiff1d(cluster, cluster_train))

        if test_selection_option == 2:
            minimum_test_samples = n_observations
            for cl_id in range(0,k):
                if len(cluster_test[cl_id]) < minimum_test_samples:
                    minimum_test_samples = len(cluster_test[cl_id])

            for cl_id in range(0,k):
                idx_test = np.concatenate((idx_test, random.sample(list(cluster_test[cl_id]), minimum_test_samples)))

        idx_train = np.sort(idx_train.astype(int))
        idx_test = np.sort(idx_test.astype(int))

        if (test_selection_option == 1) & (np.size(idx_test) + np.size(idx_train) != n_observations):
            raise ValueError("Sizes of train and test data do not sum up to the total number of observations.")

        # Print detailed information on sampling:
        if self.verbose == True:
            _print_verbose_information(self.idx, idx_train, idx_test)

        return (idx_train, idx_test)

    def percentage(self, perc, test_selection_option=1):
        """
        This function takes an ``idx`` classifications from a clustering technique and
        samples a certain percentage ``perc`` from every cluster as the training data.
        The remaining percentage is the test data.

        *Note:*
        If the clusters sizes are comparable, using this function is not recommended
        as it might give similar train sample distribution as random sampling
        (``TrainTestSelect.random``). It might still be useful in cases where one
        cluster is significantly smaller than others and there is a chance that this
        cluster will not get reflected in the train data if random sampling was used.

        **Example:**

        If the full data has 10 observations with indices:

        ``[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]``

        and you requested 40% of the data to be training data, the function may
        return:

        ``idx_train = [0, 1, 2, 7]``

        and:

        ``idx_test = [3, 4, 5, 6, 8, 9]``

        as the remaining 60% test data.

        :param perc:
            percentage of data to be selected as training data from each cluster.
            For instance, set ``perc=20`` if you want to select 20%.
        :param test_selection_option: (optional)
            option for how the test data is selected.
            Select ``test_selection_option=1`` if you want all remaining samples
            to become test data.
            Select ``test_selection_option=2`` if you want ... *[to be determined]*

        :raises ValueError:
            if the perecentage specified is too high in combination with the
            user-provided ``idx_test`` vector and there aren't enough samples to
            select as train data.

        :raises ValueError:
            if the size of ``idx_train`` and the size of ``idx_test`` do not sum up
            to ``n_observations``. *This is for unit testing only.*

        :return:
            - **idx_train** - indices of the train data.
            - **idx_test** - indices of the test data.
        """

        # Degrade clusters if needed:
        if len(np.unique(self.idx)) != (np.max(self.idx)+1):
            (self.idx, _) = clustering_data.degrade_clusters(self.idx, verbose=False)

        n_observations = len(self.idx)
        idx_full = np.arange(0,n_observations)
        idx_train = []
        idx_test = []

        # Find the number of clusters:
        k = np.size(np.unique(self.idx))

        # Get clusters and split them into training and test indices:
        for cl_id in range(0,k):

            cluster = []

            if self.random_seed != None:
                random.seed(self.random_seed)

            for i, id in enumerate(self.idx):
                if id == cl_id:
                    cluster.append(idx_full[i])

            cluster_train = np.array(random.sample(cluster, int(len(cluster)*perc/100)))
            cluster_test = np.setdiff1d(cluster, cluster_train)

            idx_train = np.concatenate((idx_train, cluster_train))
            idx_test = np.concatenate((idx_test, cluster_test))

        idx_train = np.sort(idx_train.astype(int))
        idx_test = np.sort(idx_test.astype(int))

        if np.size(idx_test) + np.size(idx_train) != n_observations:
            raise ValueError("Size of train and test data do not sum up to the total number of observations.")

        # Print detailed information on sampling:
        if self.verbose == True:
            _print_verbose_information(self.idx, idx_train, idx_test)

        return (idx_train, idx_test)

    def manual(self, sampling_dictionary, sampling_type='percentage', test_selection_option=1):
        """
        This function takes an ``idx`` classifications from a clustering technique
        and a dictionary ``sampling_dictionary`` in which you manually specify what
        ``'percentage'`` (or what ``'number'``) of samples will be
        selected as the train data from each cluster. The dictionary keys are
        cluster numerations as per ``idx`` and the dictionary values are either
        percentage or number of train samples to be selected. The default dictionary
        values are percentage but you can select ``sampling_type='number'`` in order
        to interpret the values as a number of samples.

        By default, there is a bar that no more than 50% of observations from any
        cluster will be taken for training. This is to avoid that too little samples
        will remain for test data from small clusters. If the parameter ``bar_50`` is
        set to False, this function will allow to sample more than 50% of
        observations.

        *Note:*
        This function does not run ``degrade_clusters`` to avoid disambiguity
        between cluster numeration inside ``idx`` and inside the keys of the
        ``sampling_dictionary``! It will however check whether keys are consistent
        with ``idx`` entries and if yes it will continue running. If the ``idx``
        classifies for running ``degrade_clusters`` this information will be printed
        as a suggestion.

        **Example:**

        If the full data has 10 observations with indices:

        ``[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]``

        and a clustering technique divided the observations into clusters in the
        following way:

        ``[0, 0, 0, 0, 0, 0, 1, 1, 1, 1]``

        and the dictionary is: ``sampling_dictionary = {0:3, 1:1}`` with the values
        representing a ``number``, the function may return:

        ``idx_train = [2, 3, 5, 9]``

        so that 3 samples are taken from the first cluster and 1 sample is taken
        from the second cluster.

        :param sampling_dictionary:
            dictionary specifying manual sampling. Keys are cluster numbers and
            values are either ``percentage`` or ``number`` of samples to be taken from
            that cluster. Keys should match the cluster numbering as per ``idx``.
        :param sampling_type: (optional)
            string specifying whether percentage or number is given in the
            ``sampling_dictionary``. Available options: ``percentage`` or ``number``.
            The default is ``percentage``.
        :param test_selection_option: (optional)
            option for how the test data is selected.
            Select ``test_selection_option=1`` if you want all remaining samples
            to become test data.
            Select ``test_selection_option=2`` if you want ... *[to be determined]*

        :raises ValueError:
            if ``sampling_type`` is not ``'percentage'`` or ``'number'``.

        :raises ValueError:
            if the number of entries in ``sampling_dictionary`` does not match the
            number of clusters specified in ``idx``.

        :raises ValueError:
            if the number of samples exceeds 50% in any cluster when ``bar_50=True``.

        :raises ValueError:
            if the perecentage specified is too high in combination with the
            user-provided ``idx_test`` vector and there aren't enough samples to
            select as train data.

        :raises ValueError:
            if the size of ``idx_train`` and the size of ``idx_test`` do not sum up
            to ``n_observations``. *This is for unit testing only.*

        :return:
            - **idx_train** - indices of the train data.
            - **idx_test** - indices of the test data.
        """

        _sampling_type = ['percentage', 'number']

        # Check if degrading clusters is needed and if yes print a message:
        if len(np.unique(self.idx)) != (np.max(self.idx)+1):
            print("----------\nConsider running `degrade_clusters` on `idx`!\n----------")

        n_observations = len(self.idx)

        # Find the number of clusters:
        k = np.size(np.unique(self.idx))

        # Vector of indices 0..n_observations:
        idx_full = np.arange(0,n_observations)
        idx_train = []
        idx_test = []

        # Check that sampling_type is passed correctly:
        if sampling_type not in _sampling_type:
            raise ValueError("Variable `sampling_type` has to be one of the following: 'percentage' or 'number'.")

        # Check that dictionary has consistend number of entries with respect to `idx`:
        if len(np.unique(self.idx)) != len(sampling_dictionary.keys()):
            raise ValueError("The number of entries inside `sampling_dictionary` does not match the number of clusters specified in `idx`.")

        # Check that no percentage is higher than 50%:
        if sampling_type == 'percentage' and self.bar_50 == True:
            for key, value in sampling_dictionary.items():
                if value > 50:
                    raise ValueError("Error in cluster number " + str(key) + ". Percentage in `sampling_dictionary` cannot be higher than 50%.")

        # Check that keys and values are properly defined:
        for key, value in sampling_dictionary.items():

            # Check that all keys are present in the `idx`:
            if key not in np.unique(self.idx):
                raise ValueError("Key " + str(key) + " does not match an entry in `idx`.")

            # Check that keys are non-negative integers:
            if not (isinstance(key, int) and key >= 0):
                raise ValueError("Error in cluster number " + str(key) + ". Key must be a non-negative integer.")

            # Check that percentage is between 0 and 50:
            if sampling_type == 'percentage' and self.bar_50 == True:
                if not (value >= 0 and value <= 50):
                    raise ValueError("Error in cluster number " + str(key) + ". The percentage must be between 0% and 50%.")

            # Check that percentage is between 0 and 100:
            if sampling_type == 'percentage' and self.bar_50 == False:
                if not (value >= 0 and value <= 100):
                    raise ValueError("Error in cluster number " + str(key) + ". The percentage must be between 0% and 100%.")

            # Check that number is a non-negative integer:
            if sampling_type == 'number':
                if not (isinstance(value, int) and value >= 0):
                    raise ValueError("Error in cluster number " + str(key) + ". The number must be a non-negative integer.")

        # Sampling the user-specified percentage of observations:
        if sampling_type == 'percentage':

            # Get clusters and split them into training and test indices:
            for key, value in sampling_dictionary.items():

                cluster = []

                if self.random_seed != None:
                    random.seed(self.random_seed)

                for i, id in enumerate(self.idx):
                    if id == key:
                        cluster.append(idx_full[i])

                cluster_train = np.array(random.sample(cluster, int(len(cluster)*value/100)))
                cluster_test = np.setdiff1d(cluster, cluster_train)

                idx_train = np.concatenate((idx_train, cluster_train))
                idx_test = np.concatenate((idx_test, cluster_test))

        # Sampling the user-specified number of observations:
        if sampling_type == 'number':

            # Get clusters and split them into training and test indices:
            for key, value in sampling_dictionary.items():

                cluster = []

                if self.random_seed != None:
                    random.seed(self.random_seed)

                for i, id in enumerate(self.idx):
                    if id == key:
                        cluster.append(idx_full[i])

                # Check that the number of requested observations from that cluster does not exceed 50% of observations in that cluster:
                if value > 0.5*len(cluster) and self.bar_50==True:
                    raise ValueError("Error in cluster number " + str(key) + ". The number of samples in `sampling_dictionary` cannot exceed 50% of observations in a cluster.")

                if value > len(cluster) and self.bar_50==False:
                    raise ValueError("Error in cluster number " + str(key) + ". The number of samples in `sampling_dictionary` cannot exceed the number of observations in a cluster.")

                cluster_train = np.array(random.sample(cluster, value))
                cluster_test = np.setdiff1d(cluster, cluster_train)

                idx_train = np.concatenate((idx_train, cluster_train))
                idx_test = np.concatenate((idx_test, cluster_test))

        idx_train = np.sort(idx_train.astype(int))
        idx_test = np.sort(idx_test.astype(int))

        if np.size(idx_test) + np.size(idx_train) != n_observations:
            raise ValueError("Size of train and test data do not sum up to the total number of observations.")

        # Print detailed information on sampling:
        if self.verbose == True:
            _print_verbose_information(self.idx, idx_train, idx_test)

        return (idx_train, idx_test)

    def random(self, perc, test_selection_option=1):
        """
        This function takes an ``idx`` classifications from a clustering technique and
        samples train data at random from the entire data set.

        *Note:*
        If the parameter ``idx_test`` is not provided, test data will be selected as
        all remaining samples that didn't go into train data. Optionally, you may
        provide the parameter ``idx_test`` that forces the function to maintain this
        set of test data and the train data will be selected ignoring the indices in
        ``idx_test``. This may be useful if training a machine learning model on
        fixed test samples is desired.

        **Example:**

        If the full data has 10 observations whose indices are:

        ``[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]``

        and you request 40% of the data to be training data, the function may return:

        ``idx_train = [0, 1, 2, 7]``

        and:

        ``idx_test = [3, 4, 5, 6, 8, 9]``

        as the remaining 60% test data.

        :param perc:
            percentage of data to be selected as training data from each cluster.
            Set ``perc=20`` if you want 20%.
        :param test_selection_option: (optional)
            option for how the test data is selected.
            Select ``test_selection_option=1`` if you want all remaining samples
            to become test data.
            Select ``test_selection_option=2`` if you want ... *[to be determined]*

        :raises ValueError:
            if the perecentage specified is too high in combination with the
            user-provided ``idx_test`` vector and there aren't enough samples to
            select as train data.

        :raises ValueError:
            if the size of ``idx_train`` and the size of ``idx_test`` do not sum up
            to ``n_observations``. *This is for unit testing only.*

        :return:
            - **idx_train** - indices of the train data.
            - **idx_test** - indices of the test data.
        """

        # Degrade clusters if needed:
        if len(np.unique(self.idx)) != (np.max(self.idx)+1):
            (self.idx, _) = clustering_data.degrade_clusters(self.idx, verbose=False)

        n_observations = len(self.idx)

        # Find the number of clusters:
        k = np.size(np.unique(self.idx))

        idx_full = np.arange(0,n_observations)
        idx_test = np.array(self.idx_test)

        if self.random_seed != None:
            random.seed(self.random_seed)

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

        idx_train = np.sort(idx_train.astype(int))
        idx_test = np.sort(idx_test.astype(int))

        if len(idx_test) == 0 and (np.size(idx_test) + np.size(idx_train) != n_observations):
            raise ValueError("Size of train and test data do not sum up to the total number of observations.")

        # Print detailed information on sampling:
        if self.verbose == True:
            _print_verbose_information(self.idx, idx_train, idx_test)

        return (idx_train, idx_test)

















def test():
    """
    This function performs regression testing of this module.
    """

    # Tests of `TrainTestSelect` class init: -----------------------------------
    try:
        TrainTestSelect(np.array([0,0,0,0,0,0,0,1,1,1,1]), idx_test=[], bar_50=1, random_seed=None, verbose=False)
        print('Test (01) of `TrainTestSelect` class failed.')
        return 0
    except Exception:
        pass

    try:
        TrainTestSelect(np.array([0,0,0,0,0,0,0,1,1,1,1]), idx_test=[], bar_50=True, random_seed=0.4, verbose=False)
        print('Test (02) of `TrainTestSelect` class failed.')
        return 0
    except Exception:
        pass

    try:
        TrainTestSelect(np.array([0,0,0,0,0,0,0,1,1,1,1]), idx_test=[], bar_50=True, random_seed=100, verbose=2)
        print('Test (03) of `TrainTestSelect` class failed.')
        return 0
    except Exception:
        pass

    try:
        TrainTestSelect(np.array([0,0,0,0,0,0,0,1,1,1,1]), idx_test=np.array([0,0,0,0,0,0,0,1,1,1,1,1,1]), bar_50=True, random_seed=100, verbose=False)
        print('Test (04) of `TrainTestSelect` class failed.')
        return 0
    except Exception:
        pass

    try:
        TrainTestSelect(np.array([]), idx_test=[], bar_50=True, random_seed=None, verbose=False)
        print('Test (05) of `TrainTestSelect` class failed.')
        return 0
    except Exception:
        pass

    try:
        TrainTestSelect(np.array([0,0,0,0,0,0,0,1,1,1,1]))
    except Exception:
        print('Test (06) of `TrainTestSelect` class failed.')
        return 0

    try:
        TrainTestSelect(np.array([1,1,1,1,2,2,2,2]))
    except Exception:
        print('Test (07) of `TrainTestSelect` class failed.')
        return 0

    # Tests of `TrainTestSelect.number`: ---------------------------------------

    sampling = TrainTestSelect(np.array([0,0,0,0,0,0,0,1,1,1,1]), idx_test=[], bar_50=True, random_seed=None, verbose=False)

    try:
        (idx_train, idx_test) = sampling.number(40)
    except Exception:
        print('Test (01) of `TrainTestSelect.number` failed.')
        return 0

    # Tests of `TrainTestSelect.percentage`: -----------------------------------

    sampling = TrainTestSelect(np.array([0,0,0,0,0,0,0,1,1,1,1]), idx_test=[], bar_50=True, random_seed=None, verbose=False)

    try:
        (idx_train, idx_test) = sampling.percentage(20)
    except Exception:
        print('Test (01) of `TrainTestSelect.percentage` failed.')
        return 0

    # Tests of `TrainTestSelect.manual`: -----------------------------

    sampling = TrainTestSelect(np.array([0,0,0,0,0,0,0,1,1,1,1]), idx_test=[], bar_50=True, random_seed=None, verbose=False)

    try:
        (idx_train, idx_test) = sampling.manual({1:1, 2:1})
        print('Test (01) of `TrainTestSelect.manual` failed.')
        return 0
    except Exception:
        pass

    try:
        (idx_train, idx_test) = sampling.manual({0:1, 1:1})
    except Exception:
        print('Test (02) of `TrainTestSelect.manual` failed.')
        return 0

    try:
        (idx_train, idx_test) = sampling.manual({0:1, 1:1}, sampling_type='perc')
        print('Test (03) of `TrainTestSelect.manual` failed.')
        return 0
    except Exception:
        pass

    sampling = TrainTestSelect(np.array([0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1]), idx_test=[], bar_50=True, random_seed=None, verbose=False)

    try:
        (idx_train, idx_test) = sampling.manual({0:10, 1:10}, 'percentage')
    except Exception:
        print('Test (04) of `TrainTestSelect.manual` failed.')
        return 0

    try:
        (idx_train, idx_test) = sampling.manual({0:50, 1:50}, 'percentage')
    except Exception:
        print('Test (05) of `TrainTestSelect.manual` failed.')
        return 0

    try:
        (idx_train, idx_test) = sampling.manual({0:60, 1:60}, 'percentage')
        print('Test (06) of `TrainTestSelect.manual` failed.')
        return 0
    except Exception:
        pass

    try:
        (idx_train, idx_test) = sampling.manual({0:20, 1:20}, 'number')
        print('Test (07) of `TrainTestSelect.manual` failed.')
        return 0
    except Exception:
        pass

    try:
        (idx_train, idx_test) = sampling.manual({0:5, 1:6}, 'number')
        print('Test (08) of `TrainTestSelect.manual` failed.')
        return 0
    except Exception:
        pass

    try:
        (idx_train, idx_test) = sampling.manual({0:2, 1:2}, 'number')
    except Exception:
        print('Test (09) of `TrainTestSelect.manual` failed.')
        return 0

    try:
        (idx_train, idx_test) = sampling.manual({0:5, 1:5}, 'number')
    except Exception:
        print('Test (10) of `TrainTestSelect.manual` failed.')
        return 0

    try:
        (idx_train, idx_test) = sampling.manual({0:2.2, 1:1}, 'number')
        print('Test (11) of `TrainTestSelect.manual` failed.')
        return 0
    except Exception:
        pass

    try:
        (idx_train, idx_test) = sampling.manual({0:20, 1:-20}, 'percentage')
        print('Test (12) of `TrainTestSelect.manual` failed.')
        return 0
    except Exception:
        pass

    # Tests of `TrainTestSelect.random`: --------------------------------------

    sampling = TrainTestSelect(np.array([1, 1, 1, 1, 1, 1, 2, 2, 2, 2]), idx_test=[1,2], bar_50=True, random_seed=None, verbose=False)

    try:
        (idx_train, idx_test) = sampling.random(40,)
    except Exception:
        print('Test (01) of `TrainTestSelect.random` failed.')
        return 0

    sampling = TrainTestSelect(np.array([1, 1, 1, 1, 1, 1, 2, 2, 2, 2]), idx_test=[1,2,3,4,5,6], bar_50=True, random_seed=None, verbose=False)

    try:
        (idx_train, idx_test) = sampling.random(40)
    except Exception:
        print('Test (02) of `TrainTestSelect.random` failed.')
        return 0

    sampling = TrainTestSelect(np.array([1, 1, 1, 1, 1, 1, 2, 2, 2, 2]), idx_test=[1,2,3,4,5,6,7], bar_50=True, random_seed=None, verbose=False)

    try:
        (idx_train, idx_test) = sampling.random(40)
        print('Test (03) of `TrainTestSelect.random` failed.')
        return 0
    except Exception:
        pass

    print("Test passed.")
    return 1
