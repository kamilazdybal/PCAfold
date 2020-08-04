import numpy as np
import random
from PCAfold import clustering_data

def __print_verbose_information(idx, idx_train, idx_test):
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

    **Example:**

    .. code::

      from PCAfold import TrainTestSelect
      import numpy as np

      idx = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1])
      selection = TrainTestSelect(idx, idx_test=[5,9], random_seed=100, verbose=True)

    :param idx:
        vector of cluster classifications.
    :param idx_test: (optional)
        vector or list of user-provided indices for test data. If specified, train
        data will be selected ignoring the indices in ``idx_test`` and the test
        data will be returned the same as the user-provided ``idx_test``.
        If not specified, test samples will be selected according to the
        ``test_selection_option`` parameter (see documentation for each sampling function).
        Setting fixed ``idx_test`` parameter may be useful if training a machine
        learning model on specific test samples is desired.
    :param random_seed: (optional)
        integer specifying random seed for random sample selection.
    :param verbose: (optional)
        boolean for printing sampling details.

    :raises ValueError:
        if ``idx`` vector has length zero, or is not a list or ``numpy.ndarray``.
    :raises ValueError:
        if ``idx_test`` vector has more unique observations than ``idx``, or is not a list or ``numpy.ndarray``.
    :raises ValueError:
        if ``random_seed`` is not an integer or ``None``.
    :raises ValueError:
        if ``verbose`` is not a boolean.
    """

    def __init__(self, idx, idx_test=[], random_seed=None, verbose=False):

        if len(idx) == 0:
            raise ValueError("Parameter `idx` has length zero.")
        else:
            if len(np.unique(idx_test)) > len(idx):
                raise ValueError("Parameter `idx` has less observations than current `idx_test`.")
            else:
                if isinstance(idx, list) or isinstance(idx, np.ndarray):
                    self.__idx = idx
                else:
                    raise ValueError("Parameter `idx` has to be a list or numpy.ndarray.")

        if len(np.unique(idx_test)) > len(idx):
            raise ValueError("Parameter `idx_test` has more unique observations than `idx`.")
        else:
            if isinstance(idx_test, list) or isinstance(idx_test, np.ndarray):
                self.__idx_test = idx_test
            else:
                raise ValueError("Parameter `idx_test` has to be a list or numpy.ndarray.")

        if random_seed != None:
            if not isinstance(random_seed, int):
                raise ValueError("Parameter `random_seed` has to be an integer or None.")
            if isinstance(random_seed, bool):
                raise ValueError("Parameter `random_seed` has to be an integer or None.")
            else:
                self.__random_seed = random_seed
        else:
            self.__random_seed = random_seed

        if not isinstance(verbose, bool):
            raise ValueError("Parameter `verbose` has to be a boolean.")
        else:
            self.__verbose = verbose

        if len(np.unique(idx_test)) != 0:
            self.__using_user_defined_idx_test = True
            if self.verbose==True:
                print('User defined test samples will be used. Parameter `test_selection_option` will be ignored.\n')
        else:
            self.__using_user_defined_idx_test = False

    @property
    def idx(self):
        return self.__idx

    @property
    def idx_test(self):
        return self.__idx_test

    @property
    def random_seed(self):
        return self.__random_seed

    @property
    def verbose(self):
        return self.__verbose

    @idx.setter
    def idx(self, new_idx):
        if len(new_idx) == 0:
            raise ValueError("Parameter `idx` has length zero.")
        else:
            if len(np.unique(self.idx_test)) > len(new_idx):
                raise ValueError("Parameter `idx` has less observations than current `idx_test`.")
            else:
                if isinstance(new_idx, list) or isinstance(new_idx, np.ndarray):
                    self.__idx = new_idx
                else:
                    raise ValueError("Parameter `idx` has to be a list or numpy.ndarray.")

    @idx_test.setter
    def idx_test(self, new_idx_test):
        if len(new_idx_test) > len(self.idx):
            raise ValueError("Parameter `idx_test` has more unique observations than `idx`.")
        else:
            if isinstance(new_idx_test, list) or isinstance(new_idx_test, np.ndarray):
                self.__idx_test = new_idx_test
            else:
                raise ValueError("Parameter `idx_test` has to be a list or numpy.ndarray.")

            if len(np.unique(new_idx_test)) != 0:
                self.__using_user_defined_idx_test = True
                if self.verbose==True:
                    print('User defined test samples will be used. Parameter `test_selection_option` will be ignored.\n')
            else:
                self.__using_user_defined_idx_test = False

    @random_seed.setter
    def random_seed(self, new_random_seed):
        if new_random_seed != None:
            if not isinstance(new_random_seed, int):
                raise ValueError("Parameter `random_seed` has to be an integer or None.")
            if isinstance(new_random_seed, bool):
                raise ValueError("Parameter `random_seed` has to be an integer or None.")
            else:
                self.__random_seed = new_random_seed
        else:
            self.__random_seed = new_random_seed

    @verbose.setter
    def verbose(self, new_verbose):
        if not isinstance(new_verbose, bool):
            raise ValueError("Parameter `verbose` has to be a boolean.")
        else:
            self.__verbose = new_verbose

    def number(self, perc, test_selection_option=1):
        """
        This function uses classifications into :math:`k` clusters and samples
        fixed number of observations from every cluster as training data.
        In general, this results in a balanced representation of features
        identified by a clustering algorithm.

        **Example:**

        .. code::

          from PCAfold import TrainTestSelect
          import numpy as np

          idx = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1])
          selection = TrainTestSelect(idx)
          (idx_train, idx_test) = selection.number(20, test_selection_option=1)

        **Train data:**

        The number of train samples is estimated based on the percentage
        ``perc`` provided.
        First, the total number of samples for training is estimated as a
        percentage ``perc`` from the total number of observations ``n_observations`` in a data set.
        Next, this number is divided equally into :math:`k` clusters. The
        result ``n_of_samples`` is the number of samples that will be selected
        from each cluster:

        .. math::

            \\verb|n_of_samples| = \\verb|int| \Big( \\frac{\\verb|perc| \cdot \\verb|n_observations|}{k \cdot 100} \Big)

        **Test data:**

        Two options for sampling test data are implemented. If you select
        ``test_selection_option=1`` all remaining samples that were not taken as
        train data become the test data. If you select ``test_selection_option=2``,
        the smallest cluster is found and the remaining number of observations
        :math:`m` are taken as test data in that cluster. Next, the same number
        of samples :math:`m` is taken from all remaining larger clusters.

        The scheme below presents graphically how train and test data can be selected using ``test_selection_option`` parameter:

        .. image:: ../images/sampling-test-selection-option-number.png
          :width: 700
          :align: center

        Here :math:`n` and :math:`m` are fixed numbers for each cluster.
        In general, :math:`n \\neq m`.

        :param perc:
            percentage of data to be selected as training data from the entire data set.
            For instance, set ``perc=20`` if you want to select 20%.
        :param test_selection_option: (optional)
            option for how the test data is selected.
            Select ``test_selection_option=1`` if you want all remaining samples
            to become test data.
            Select ``test_selection_option=2`` if you want to select a subset
            of the remaining samples as test data.

        :raises ValueError:
            if ``perc`` is not a number between 0-100.

        :raises ValueError:
            if ``test_selection_option`` is not equal to 1 or 2.

        :raises ValueError:
            if the percentage specified is too high in combination with the
            user-provided ``idx_test`` vector and there aren't enough samples to
            select as train data.

        :return:
            - **idx_train** - indices of the train data.
            - **idx_test** - indices of the test data.
        """

        # Check if `perc` parameter was passed correctly:
        if (perc < 0) or (perc > 100):
            raise ValueError("Percentage has to be between 0-100.")

        # Check if `test_selection_option` parameter was passed correctly:
        _test_selection_option = [1,2]
        if test_selection_option not in _test_selection_option:
            raise ValueError("Test selection option can only be 1 or 2.")

        # Degrade clusters if needed:
        if len(np.unique(self.idx)) != (np.max(self.idx)+1):
            (self.idx, _) = clustering_data.degrade_clusters(self.idx, verbose=False)

        # Initialize vector of indices 0..n_observations:
        n_observations = len(self.idx)
        idx_full = np.arange(0, n_observations)
        idx_test = np.unique(np.array(self.idx_test))
        idx_full_no_test = np.setdiff1d(idx_full, idx_test)

        # Find the number of clusters:
        k = np.size(np.unique(self.idx))

        # Calculate fixed number of samples that will be taken from every cluster as the training data:
        n_of_samples = int(perc*n_observations/k/100)

        # Initialize auxiliary variables:
        idx_train = []
        cluster_test = []

        # Get clusters and split them into train and test indices:
        for cl_id in range(0,k):

            if self.random_seed != None:
                random.seed(self.random_seed)

            # Variable `cluster` contains indices of observations that are allowed to be selected as train samples in a particular cluster:
            cluster = []
            for i, id in enumerate(self.idx[idx_full_no_test]):
                if id == cl_id:
                    cluster.append(idx_full_no_test[i])

            # Selection of training data:
            if int(len(cluster)) < n_of_samples:
                raise ValueError("The requested percentage requires taking more samples from cluster " + str(cl_id+1) + " than there are available observations in that cluster. Consider lowering the percentage or use a different sampling function.")
            else:
                cluster_train = np.array(random.sample(cluster, n_of_samples))
                idx_train = np.concatenate((idx_train, cluster_train))

            if self.__using_user_defined_idx_test==False:

                # Selection of test data - all data that remains is test data:
                if test_selection_option == 1:

                    cluster_test = np.setdiff1d(cluster, cluster_train)
                    idx_test = np.concatenate((idx_test, cluster_test))

        # Selection of test data - equal samples from each cluster:
                if test_selection_option == 2:

                    cluster_test.append(np.setdiff1d(cluster, cluster_train))

        if self.__using_user_defined_idx_test==False:

            if test_selection_option == 2:

                # Search for the smallest number of remaining observations in any cluster:
                minimum_test_samples = n_observations
                for cl_id in range(0,k):
                    if len(cluster_test[cl_id]) < minimum_test_samples:
                        minimum_test_samples = len(cluster_test[cl_id])

                # Sample that amount from every cluster:
                for cl_id in range(0,k):
                    idx_test = np.concatenate((idx_test, random.sample(list(cluster_test[cl_id]), minimum_test_samples)))

        idx_train = np.sort(idx_train.astype(int))
        idx_test = np.sort(idx_test.astype(int))

        # Print detailed information on sampling:
        if self.verbose == True:
            __print_verbose_information(self.idx, idx_train, idx_test)

        return (idx_train, idx_test)

    def percentage(self, perc, test_selection_option=1):
        """
        This function uses classifications into :math:`k` clusters and
        samples a certain percentage ``perc`` from every cluster as the training data.

        **Example:**

        .. code:: python

          from PCAfold import TrainTestSelect
          import numpy as np

          idx = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1])
          selection = TrainTestSelect(idx)
          (idx_train, idx_test) = selection.percentage(20, test_selection_option=1)

        *Note:*
        If the cluster sizes are comparable, this function will give a similar
        train sample distribution as random sampling (``TrainTestSelect.random``).
        This sampling can be useful in cases where one cluster is significantly
        smaller than others and there is a chance that this cluster will not get
        covereed in the train data if random sampling was used.

        **Train data:**

        The number of train samples is estimated based on the percentage ``perc`` provided.
        First, the size of the :math:`i^{th}` cluster is estimated ``cluster_size_i``
        and then a percentage ``perc`` of that number is selected.

        **Test data:**

        Two options for sampling test data are implemented. If you select
        ``test_selection_option=1`` all remaining samples that were not taken as
        train data become the test data. If you select
        ``test_selection_option=2`` the same procedure will be used to select
        test data as was used to select train data (only allowed if the number of samples
        taken as train data from any cluster did not exceed 50% of observations
        in that cluster).

        The scheme below presents graphically how train and test data can be
        selected using ``test_selection_option`` parameter:

        .. image:: ../images/sampling-test-selection-option-percentage.png
          :width: 700
          :align: center

        Here :math:`p` is the percentage ``perc`` provided.

        :param perc:
            percentage of data to be selected as training data from each cluster.
            For instance, set ``perc=20`` if you want to select 20%.
        :param test_selection_option: (optional)
            option for how the test data is selected.
            Select ``test_selection_option=1`` if you want all remaining samples
            to become test data.
            Select ``test_selection_option=2`` if you want to select a subset
            of the remaining samples as test data.

        :raises ValueError:
            if ``perc`` is not a number between 0-100.

        :raises ValueError:
            if ``test_selection_option`` is not equal to 1 or 2.

        :raises ValueError:
            if the perecentage specified is too high in combination with the
            user-provided ``idx_test`` vector and there aren't enough samples to
            select as train data.

        :return:
            - **idx_train** - indices of the train data.
            - **idx_test** - indices of the test data.
        """

        # Check if `perc` parameter was passed correctly:
        if (perc < 0) or (perc > 100):
            raise ValueError("Percentage has to be between 0-100.")

        # Check if `test_selection_option` parameter was passed correctly:
        _test_selection_option = [1,2]
        if test_selection_option not in _test_selection_option:
            raise ValueError("Test selection option can only be 1 or 2.")

        # Degrade clusters if needed:
        if len(np.unique(self.idx)) != (np.max(self.idx)+1):
            (self.idx, _) = clustering_data.degrade_clusters(self.idx, verbose=False)

        # Initialize vector of indices 0..n_observations:
        n_observations = len(self.idx)
        idx_full = np.arange(0,n_observations)
        idx_test = np.unique(np.array(self.idx_test))
        idx_full_no_test = np.setdiff1d(idx_full, idx_test)

        # Find the number of clusters:
        k = np.size(np.unique(self.idx))

        # Initialize auxiliary variables:
        idx_train = []

        # Get cluster populations:
        cluster_populations = clustering_data.get_populations(self.idx)

        # Get clusters and split them into training and test indices:
        for cl_id in range(0,k):

            if self.random_seed != None:
                random.seed(self.random_seed)

            # Variable `cluster` contains indices of observations that are allowed to be selected as train samples in a particular cluster:
            cluster = []
            for i, id in enumerate(self.idx[idx_full_no_test]):
                if id == cl_id:
                    cluster.append(idx_full_no_test[i])

            # Selection of training data:
            if int(len(cluster)) < int(cluster_populations[cl_id]*perc/100):
                raise ValueError("The requested percentage requires taking more samples from cluster " + str(cl_id+1) + " than there are available observations in that cluster. Consider lowering the percentage or use a different sampling function.")
            else:
                cluster_train = np.array(random.sample(cluster, int(cluster_populations[cl_id]*perc/100)))
                idx_train = np.concatenate((idx_train, cluster_train))

            if self.__using_user_defined_idx_test==False:

                # Selection of test data - all data that remains is test data:
                if test_selection_option == 1:

                    cluster_test = np.setdiff1d(cluster, cluster_train)
                    idx_test = np.concatenate((idx_test, cluster_test))

                if test_selection_option == 2:

                    # Check if there is enough test samples to select:
                    if perc > 50:
                        raise ValueError("Percentage is larger than 50% and test samples cannot be selected with `test_selection_option=2`.")
                    else:
                        cluster_test = np.setdiff1d(cluster, cluster_train)
                        cluster_test_sampled = np.array(random.sample(list(cluster_test), int(cluster_populations[cl_id]*perc/100)))
                        idx_test = np.concatenate((idx_test, cluster_test_sampled))

        idx_train = np.sort(idx_train.astype(int))
        idx_test = np.sort(idx_test.astype(int))

        # Print detailed information on sampling:
        if self.verbose == True:
            __print_verbose_information(self.idx, idx_train, idx_test)

        return (idx_train, idx_test)

    def manual(self, sampling_dictionary, sampling_type='percentage', test_selection_option=1):
        """
        This function uses classifications into :math:`k` clusters
        and a dictionary ``sampling_dictionary`` in which you manually specify what
        ``'percentage'`` (or what ``'number'``) of samples will be
        selected as the train data from each cluster. The dictionary keys are
        cluster classifications as per ``idx`` and the dictionary values are either
        percentage or number of train samples to be selected. The default dictionary
        values are percentage but you can select ``sampling_type='number'`` in order
        to interpret the values as a number of samples.

        **Example:**

        .. code:: python

          from PCAfold import TrainTestSelect
          import numpy as np

          idx = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1])
          selection = TrainTestSelect(idx)
          (idx_train, idx_test) = selection.manual({0:1, 1:1, 2:1}, sampling_type='number', test_selection_option=1)

        *Note:*
        This function does not run ``degrade_clusters`` to avoid disambiguity
        between cluster classifications inside ``idx`` and inside the keys of the
        ``sampling_dictionary``! It will however check whether keys are consistent
        with ``idx`` entries and if yes it will continue running. If the ``idx``
        classifies for running ``degrade_clusters`` this information will be printed
        as a suggestion.

        **Train data:**

        The number of train samples selected from each cluster is estimated based
        on the ``sampling_dictionary``. For ``key : value``, percentage ``value``
        (or number ``value``) of samples will be selected from cluster ``key``.

        **Test data:**

        Two options for sampling test data are implemented.
        If you select ``test_selection_option=1`` all remaining samples that
        were not taken as train data become the test data.
        If you select
        ``test_selection_option=2`` the same procedure will be used to select
        test data as was used to select train data (only allowed if the number
        of samples taken as train data from any cluster did not exceed 50%
        of observations in that cluster).

        The scheme below presents graphically how train and test data can be
        selected using ``test_selection_option`` parameter:

        .. image:: ../images/sampling-test-selection-option-manual.png
          :width: 700
          :align: center

        Here it is understood that :math:`n_1` train samples were requested from
        the first cluster, :math:`n_2` from the second cluster and :math:`n_3`
        from the third cluster. This can be achieved by setting:

        .. code:: python

            sampling_dictionary = {0:n_1, 1:n_2, 2:n_3}

        :param sampling_dictionary:
            dictionary specifying manual sampling. Keys are cluster classifications and
            values are either ``percentage`` or ``number`` of samples to be taken from
            that cluster. Keys should match the cluster classifications as per ``idx``.
        :param sampling_type: (optional)
            string specifying whether percentage or number is given in the
            ``sampling_dictionary``. Available options: ``percentage`` or ``number``.
            The default is ``percentage``.
        :param test_selection_option: (optional)
            option for how the test data is selected.
            Select ``test_selection_option=1`` if you want all remaining samples
            to become test data.
            Select ``test_selection_option=2`` if you want to select a subset
            of the remaining samples as test data.

        :raises ValueError:
            if ``sampling_type`` is not ``'percentage'`` or ``'number'``.

        :raises ValueError:
            if the number of entries in ``sampling_dictionary`` does not match the
            number of clusters specified in ``idx``.

        :raises ValueError:
            if any ``value`` in ``sampling_dictionary`` is not a number between 0-100 when ``sampling_type='percentage'``.

        :raises ValueError:
            if any ``value`` in ``sampling_dictionary`` is not a non-negative integer when ``sampling_type='number'``.

        :raises ValueError:
            if ``test_selection_option`` is not equal to 1 or 2.

        :raises ValueError:
            if the perecentage specified is too high in combination with the
            user-provided ``idx_test`` vector and there aren't enough samples to
            select as train data.

        :return:
            - **idx_train** - indices of the train data.
            - **idx_test** - indices of the test data.
        """

        # Check that sampling_type is passed correctly:
        _sampling_type = ['percentage', 'number']
        if sampling_type not in _sampling_type:
            raise ValueError("Variable `sampling_type` has to be one of the following: 'percentage' or 'number'.")

        # Check that dictionary has consistend number of entries with respect to `idx`:
        if len(np.unique(self.idx)) != len(sampling_dictionary.keys()):
            raise ValueError("The number of entries inside `sampling_dictionary` does not match the number of clusters specified in `idx`.")

        # Check that keys and values are properly defined:
        for key, value in sampling_dictionary.items():

            # Check that all keys are present in the `idx`:
            if key not in np.unique(self.idx):
                raise ValueError("Key " + str(key) + " does not match an entry in `idx`.")

            # Check that keys are non-negative integers:
            if not (isinstance(key, int) and key >= 0):
                raise ValueError("Error in cluster number " + str(key) + ". Key must be a non-negative integer.")

            # Check that percentage is between 0 and 100:
            if sampling_type == 'percentage':
                if not (value >= 0 and value <= 100):
                    raise ValueError("Error in cluster number " + str(key) + ". The percentage has to be between 0-100.")

            # Check that number is a non-negative integer:
            if sampling_type == 'number':
                if not (isinstance(value, int) and value >= 0):
                    raise ValueError("Error in cluster number " + str(key) + ". The number must be a non-negative integer.")

        # Check that `test_selection_option` parameter was passed correctly:
        _test_selection_option = [1,2]
        if test_selection_option not in _test_selection_option:
            raise ValueError("Test selection option can only be 1 or 2.")

        # Check if degrading clusters is needed and if yes print a message:
        if len(np.unique(self.idx)) != (np.max(self.idx)+1):
            print("----------\nConsider running `degrade_clusters` on `idx`!\n----------")

        # Initialize vector of indices 0..n_observations:
        n_observations = len(self.idx)
        idx_full = np.arange(0,n_observations)
        idx_test = np.unique(np.array(self.idx_test))
        idx_full_no_test = np.setdiff1d(idx_full, idx_test)

        # Find the number of clusters:
        k = np.size(np.unique(self.idx))

        # Initialize auxiliary variables:
        idx_train = []

        # Get cluster populations:
        cluster_populations = clustering_data.get_populations(self.idx)

        # Sampling the user-specified percentage of observations:
        if sampling_type == 'percentage':

            # Get clusters and split them into training and test indices:
            for key, value in sampling_dictionary.items():

                if self.random_seed != None:
                    random.seed(self.random_seed)

                cluster = []
                for i, id in enumerate(self.idx[idx_full_no_test]):
                    if id == key:
                        cluster.append(idx_full_no_test[i])

                # Selection of training data:
                cluster_train = np.array(random.sample(cluster, int(len(cluster)*value/100)))
                idx_train = np.concatenate((idx_train, cluster_train))

                if self.__using_user_defined_idx_test==False:

                    # Selection of test data - all data that remains is test data:
                    if test_selection_option == 1:

                        cluster_test = np.setdiff1d(cluster, cluster_train)
                        idx_test = np.concatenate((idx_test, cluster_test))

                    if test_selection_option == 2:

                        if value > 50:
                            raise ValueError("Percentage in cluster " + str(key+1) + " is larger than 50% and test samples cannot be selected with `test_selection_option=2`.")

                        cluster_test = np.setdiff1d(cluster, cluster_train)
                        cluster_test_sampled = np.array(random.sample(list(cluster_test), int(len(cluster)*value/100)))
                        idx_test = np.concatenate((idx_test, cluster_test_sampled))

        # Sampling the user-specified number of observations:
        if sampling_type == 'number':

            # Get clusters and split them into training and test indices:
            for key, value in sampling_dictionary.items():

                if self.random_seed != None:
                    random.seed(self.random_seed)

                cluster = []
                for i, id in enumerate(self.idx[idx_full_no_test]):
                    if id == key:
                        cluster.append(idx_full_no_test[i])

                if value > len(cluster):
                    raise ValueError("Error in cluster number " + str(key+1) + ". The number of samples in `sampling_dictionary` cannot exceed the number of observations in a cluster.")

                cluster_train = np.array(random.sample(cluster, value))
                idx_train = np.concatenate((idx_train, cluster_train))

                if self.__using_user_defined_idx_test==False:

                    # Selection of test data - all data that remains is test data:
                    if test_selection_option == 1:

                        cluster_test = np.setdiff1d(cluster, cluster_train)
                        idx_test = np.concatenate((idx_test, cluster_test))

                    if test_selection_option == 2:

                        if value > int(cluster_populations[key]*0.5):
                            raise ValueError("Number of samples in cluster " + str(key+1) + " is larger than 50% and test samples cannot be selected with `test_selection_option=2`.")

                        cluster_test = np.setdiff1d(cluster, cluster_train)
                        cluster_test_sampled = np.array(random.sample(list(cluster_test), value))
                        idx_test = np.concatenate((idx_test, cluster_test_sampled))

        idx_train = np.sort(idx_train.astype(int))
        idx_test = np.sort(idx_test.astype(int))

        # Print detailed information on sampling:
        if self.verbose == True:
            __print_verbose_information(self.idx, idx_train, idx_test)

        return (idx_train, idx_test)

    def random(self, perc, test_selection_option=1):
        """
        This function samples train data at random from the entire data set.

        **Example:**

        .. code:: python

          from PCAfold import TrainTestSelect
          import numpy as np

          idx = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1])
          selection = TrainTestSelect(idx)
          (idx_train, idx_test) = selection.random(20, test_selection_option=1)

        Due to the nature of this sampling technique, it is not necessary to
        have ``idx`` classifications since random samples can also be selected
        from unclassified data sets. You can achieve that by generating a dummy
        ``idx`` vector that has the same number of observations
        ``n_observations`` as your data set. For instance:

        .. code:: python

          from PCAfold import TrainTestSelect
          import numpy as np

          idx = np.zeros(n_observations)
          selection = TrainTestSelect(idx)
          (idx_train, idx_test) = selection.random(20, test_selection_option=1)

        **Train data:**

        The total number of train samples is computed as a percentage ``perc``
        from the total number of observations in a data set. These samples are
        then drawn at random from the entire data set, independent of cluster
        classifications.

        **Test data:**

        Two options for sampling test data are implemented. If you select
        ``test_selection_option=1`` all remaining samples that were not taken
        as train data become the test data. If you select
        ``test_selection_option=2`` the same procedure is used to select
        test data as was used to select train data
        (only allowed if ``perc`` is less than 50%).

        The scheme below presents graphically how train and test data can be
        selected using ``test_selection_option`` parameter:

        .. image:: ../images/sampling-test-selection-option-random.png
          :width: 700
          :align: center

        Here :math:`p` is the percentage ``perc`` provided.

        :param perc:
            percentage of data to be selected as training data from each cluster.
            Set ``perc=20`` if you want 20%.
        :param test_selection_option: (optional)
            option for how the test data is selected.
            Select ``test_selection_option=1`` if you want all remaining samples
            to become test data.
            Select ``test_selection_option=2`` if you want to select a subset
            of the remaining samples as test data.

        :raises ValueError:
            if ``perc`` is not a number between 0-100.

        :raises ValueError:
            if ``test_selection_option`` is not equal to 1 or 2.

        :raises ValueError:
            if the perecentage specified is too high in combination with the
            user-provided ``idx_test`` vector and there aren't enough samples to
            select as train data.

        :return:
            - **idx_train** - indices of the train data.
            - **idx_test** - indices of the test data.
        """

        # Check if `perc` parameter was passed correctly:
        if (perc < 0) or (perc > 100):
            raise ValueError("Percentage has to be between 0-100.")

        # Check that `test_selection_option` parameter was passed correctly:
        _test_selection_option = [1,2]
        if test_selection_option not in _test_selection_option:
            raise ValueError("Test selection option can only be 1 or 2.")

        # Degrade clusters if needed:
        if len(np.unique(self.idx)) != (np.max(self.idx)+1):
            (self.idx, _) = clustering_data.degrade_clusters(self.idx, verbose=False)

        # Initialize vector of indices 0..n_observations:
        n_observations = len(self.idx)
        idx_full = np.arange(0, n_observations)
        idx_test = np.unique(np.array(self.idx_test))
        idx_full_no_test = np.setdiff1d(idx_full, idx_test)

        # Find the number of clusters:
        k = np.size(np.unique(self.idx))

        if self.random_seed != None:
            random.seed(self.random_seed)

        if self.__using_user_defined_idx_test==True:

            if int(len(idx_full)*perc/100) <= len(idx_full_no_test):
                idx_train = np.array(random.sample(idx_full_no_test.tolist(), int(len(idx_full)*perc/100)))
            else:
                raise ValueError("The training percentage specified is too high, there aren't enough samples.")

        else:

            if test_selection_option==1:

                idx_train = np.array(random.sample(idx_full.tolist(), int(len(idx_full)*perc/100)))
                idx_test = np.setdiff1d(idx_full, idx_train)

            elif test_selection_option==2:

                idx_train = np.array(random.sample(idx_full.tolist(), int(len(idx_full)*perc/100)))

                # Check if there is enough test samples to select:
                if perc > 50:
                    raise ValueError("Percentage is larger than 50% and test samples cannot be selected with `test_selection_option=2`.")
                else:
                    test_pool = np.setdiff1d(idx_full, idx_train)
                    idx_test = np.array(random.sample(list(test_pool), int(len(idx_full)*perc/100)))

        idx_train = np.sort(idx_train.astype(int))
        idx_test = np.sort(idx_test.astype(int))

        # Print detailed information on sampling:
        if self.verbose == True:
            __print_verbose_information(self.idx, idx_train, idx_test)

        return (idx_train, idx_test)
