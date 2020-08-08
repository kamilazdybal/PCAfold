from PCAfold import TrainTestSelect
from PCAfold import preprocess
from PCAfold import cluster_biased_pca
from PCAfold import normalized_local_variance
from PCAfold import PCA
import numpy as np

def test():
    """
    This function runs all regression tests available.
    """

    test_clustering()
    test_sampling()

def test_clustering():
    """
    This function performs regression testing of ``clustering_data`` module.
    """

    # ##########################################################################

    # Check if `idx` output vectors are of type numpy.ndarray and of size (_,):

    # ##########################################################################
    try:
        idx_1 = preprocess.variable_bins(np.array([1,2,3,4,5,6,7,8,9,10]), 4, verbose=False)
    except:
        print('Test of variable_bins failed.')
        return 0
    if not isinstance(idx_1, np.ndarray):
        print('Test of variable_bins failed.')
        return 0
    try:
        (n_observations,) = np.shape(idx_1)
    except:
        print('Test of variable_bins failed.')
        return 0

    try:
        idx_2 = preprocess.predefined_variable_bins(np.array([1,2,3,4,5,6,7,8,9,10]), [3.5, 8.5], verbose=False)
    except:
        print('Test of predefined_variable_bins failed.')
        return 0
    if not isinstance(idx_2, np.ndarray):
        print('Test of predefined_variable_bins failed.')
        return 0
    try:
        (n_observations,) = np.shape(idx_2)
    except:
        print('Test of predefined_variable_bins failed.')
        return 0

    try:
        idx_3 = preprocess.mixture_fraction_bins(np.array([0.1, 0.15, 0.2, 0.25, 0.6, 0.8, 1]), 2, 0.2)
    except:
        print('Test of mixture_fraction_bins failed.')
        return 0
    if not isinstance(idx_3, np.ndarray):
        print('Test of mixture_fraction_bins failed.')
        return 0
    try:
        (n_observations,) = np.shape(idx_3)
    except:
        print('Test of mixture_fraction_bins failed.')
        return 0

    try:
        idx_5 = preprocess.vqpca(np.array([[1,2,3,4,5,6,7,8,9,10],[2,3,4,5,6,7,8,9,10,11]]).T, k=2, n_pcs=1, scaling_criteria='NONE', idx_0=[], maximum_number_of_iterations=20, verbose=False)
    except:
        print('Test of vqpca failed.')
        return 0
    if not isinstance(idx_5, np.ndarray):
        print('Test of vqpca failed.')
        return 0
    try:
        (n_observations,) = np.shape(idx_5)
    except:
        print('Test of vqpca failed.')
        return 0

    try:
        idx_6 = preprocess.pc_source_bins(np.array([-100, -20, -0.1, 0, 0.1, 1, 10, 20, 200, 300, 400]), k=4, split_at_zero=True, verbose=False)
    except:
        print('Test of pc_source_bins failed.')
        return 0
    if not isinstance(idx_6, np.ndarray):
        print('Test of pc_source_bins failed.')
        return 0
    try:
        (n_observations,) = np.shape(idx_6)
    except:
        print('Test of pc_source_bins failed.')
        return 0

    # Test degrade_clusters function:
    (idx, k) = preprocess.degrade_clusters([1,1,2,2,3,3], verbose=False)
    if np.min(idx) != 0:
        print('Test of degrade_clusters failed.')
        return 0
    if k != 3:
        print('Test of degrade_clusters failed.')
        return 0

    print('Test of `clustering_data` module passed.')

def test_sampling():
    """
    This function performs regression testing of `sampling` module.
    """

    # ##########################################################################

    # Basic sanity tests:

    # ##########################################################################

    # Check that sizes of idx_train and idx_test always sum up to the total numer
    # of observations when test_selection_option=1:
    try:
        idx = np.array([0,0,0,0,0,0,0,1,1,1,1])
        sam = TrainTestSelect(idx)
        (idx_train, idx_test) = sam.number(10)
        n_observations = len(idx)
        if np.size(idx_test) + np.size(idx_train) != n_observations:
            print('Sanity test (01) failed.')
            return 0
        (idx_train, idx_test) = sam.percentage(10)
        if np.size(idx_test) + np.size(idx_train) != n_observations:
            print('Sanity test (02) failed.')
            return 0
        (idx_train, idx_test) = sam.manual({0:1,1:4}, sampling_type='number')
        if np.size(idx_test) + np.size(idx_train) != n_observations:
            print('Sanity test (03) failed.')
            return 0
        (idx_train, idx_test) = sam.random(10)
        if np.size(idx_test) + np.size(idx_train) != n_observations:
            print('Sanity test (04) failed.')
            return 0
    except Exception:
        pass

    # Check that sizes of idx_train and idx_test sum up to less than the total numer
    # for these cases with test_selection_option=2:
    try:
        idx = np.array([0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1])
        sam = TrainTestSelect(idx)
        (idx_train, idx_test) = sam.number(10, test_selection_option=2)
        n_observations = len(idx)
        if np.size(idx_test) + np.size(idx_train) >= n_observations:
            print('Sanity test (05) failed.')
            return 0
        (idx_train, idx_test) = sam.percentage(10, test_selection_option=2)
        if np.size(idx_test) + np.size(idx_train) >= n_observations:
            print('Sanity test (06) failed.')
            return 0
        (idx_train, idx_test) = sam.manual({0:1,1:4}, sampling_type='number', test_selection_option=2)
        if np.size(idx_test) + np.size(idx_train) >= n_observations:
            print('Sanity test (07) failed.')
            return 0
        (idx_train, idx_test) = sam.random(10, test_selection_option=2)
        if np.size(idx_test) + np.size(idx_train) >= n_observations:
            print('Sanity test (08) failed.')
            return 0
    except Exception:
        pass

    # Check that indices in idx_train are never in idx_test and vice versa:
    try:
        idx = np.array([0,0,0,0,0,0,0,1,1,1,1,2,2,2,2,2,2,2,2,5,5,5,5,10,10,10,10,10])
        sam = TrainTestSelect(idx)
        (idx_train, idx_test) = sam.number(10, test_selection_option=1)
        if len(np.setdiff1d(idx_train, idx_test)) != 0:
            print('Sanity test (09) failed.')
            return 0
        (idx_train, idx_test) = sam.percentage(10, test_selection_option=1)
        if len(np.setdiff1d(idx_train, idx_test)) != 0:
            print('Sanity test (19) failed.')
            return 0
        (idx_train, idx_test) = sam.manual({0:1,1:4}, sampling_type='number', test_selection_option=1)
        if len(np.setdiff1d(idx_train, idx_test)) != 0:
            print('Sanity test (20) failed.')
            return 0
        (idx_train, idx_test) = sam.random(10, test_selection_option=1)
        if len(np.setdiff1d(idx_train, idx_test)) != 0:
            print('Sanity test (21) failed.')
            return 0
        (idx_train, idx_test) = sam.number(10, test_selection_option=2)
        if len(np.setdiff1d(idx_train, idx_test)) != 0:
            print('Sanity test (22) failed.')
            return 0
        (idx_train, idx_test) = sam.percentage(10, test_selection_option=2)
        if len(np.setdiff1d(idx_train, idx_test)) != 0:
            print('Sanity test (23) failed.')
            return 0
        (idx_train, idx_test) = sam.manual({0:1,1:4}, sampling_type='number', test_selection_option=2)
        if len(np.setdiff1d(idx_train, idx_test)) != 0:
            print('Sanity test (24) failed.')
            return 0
        (idx_train, idx_test) = sam.random(10, test_selection_option=2)
        if len(np.setdiff1d(idx_train, idx_test)) != 0:
            print('Sanity test (25) failed.')
            return 0
        (idx_train, idx_test) = sam.manual({0:10,1:10}, sampling_type='percentage', test_selection_option=1)
        if len(np.setdiff1d(idx_train, idx_test)) != 0:
            print('Sanity test (26) failed.')
            return 0
        (idx_train, idx_test) = sam.manual({0:10,1:10}, sampling_type='percentage', test_selection_option=2)
        if len(np.setdiff1d(idx_train, idx_test)) != 0:
            print('Sanity test (26) failed.')
            return 0
    except Exception:
        pass

    # ##########################################################################

    # Tests of `TrainTestSelect` class init:

    # ##########################################################################

    try:
        TrainTestSelect(np.array([0,0,0,0,0,0,0,1,1,1,1]), idx_test=[], random_seed=0.4, verbose=False)
        print('Test (02) of `TrainTestSelect` class failed.')
        return 0
    except Exception:
        pass

    try:
        TrainTestSelect(np.array([0,0,0,0,0,0,0,1,1,1,1]), idx_test=[], random_seed=100, verbose=2)
        print('Test (03) of `TrainTestSelect` class failed.')
        return 0
    except Exception:
        pass

    try:
        TrainTestSelect(np.array([0,0,0,1,1]), idx_test=np.array([1,2,3,4,5,6,7,8]), random_seed=100, verbose=False)
        print('Test (04) of `TrainTestSelect` class failed.')
        return 0
    except Exception:
        pass

    try:
        TrainTestSelect(np.array([]), idx_test=[], random_seed=None, verbose=False)
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

    try:
        TrainTestSelect(np.array([0,0,0,0,0,0,0,1,1,1,1]), idx_test=np.array([0,0,0,0,0,0,0,1,1,1,2,2,2,2,2,2]))
    except Exception:
        print('Test (08) of `TrainTestSelect` class failed.')
        return 0

    try:
        sam = TrainTestSelect(np.array([0,0,0,0,0,0,0,1,1,1,1]))
        sam.idx = np.array([1,1,1,1,1,1,2,2,2,2,2,2,2,2])
        sam.idx_test = np.arange(1,10,1)
        sam.random_seed = 100
        sam.random_seed = None
        sam.random_seed = -1
        sam.verbose = False
        sam.verbose = True
        sam.idx_test = []
    except Exception:
        print('Test (09) of `TrainTestSelect` class failed.')
        return 0

    try:
        sam = TrainTestSelect(np.array([0,0,0,0,0,0,0,1,1,1,1]))
        sam.idx_test = np.arange(1,100,1)
        print('Test (10) of `TrainTestSelect` class failed.')
        return 0
    except Exception:
        pass

    try:
        sam = TrainTestSelect(np.array([0,0,0,0,0,0,0,1,1,1,1]))
        sam.random_seed = 10.1
        print('Test (11) of `TrainTestSelect` class failed.')
        return 0
    except Exception:
        pass

    try:
        sam = TrainTestSelect(np.array([0,0,0,0,0,0,0,1,1,1,1]))
        sam.random_seed = False
        print('Test (12) of `TrainTestSelect` class failed.')
        return 0
    except Exception:
        pass

    try:
        sam = TrainTestSelect(np.array([0,0,0,0,0,0,0,1,1,1,1]))
        sam.verbose = 10
        print('Test (13) of `TrainTestSelect` class failed.')
        return 0
    except Exception:
        pass

    try:
        sam = TrainTestSelect(np.array([0,0,0,0,0,0,0,1,1,1,1]))
        sam.idx = []
        print('Test (14) of `TrainTestSelect` class failed.')
        return 0
    except Exception:
        pass

    try:
        sam = TrainTestSelect(np.array([0,0,0,0,0,0,0,1,1,1,1]))
        sam.idx_test = [0,1,2,3,4,5,6]
        sam.idx = np.array([0,1])
        print('Test (15) of `TrainTestSelect` class failed.')
        return 0
    except Exception:
        pass

    try:
        sam = TrainTestSelect(np.array([0,0,0,0,0,0,0,1,1,1,1]))
        sam.idx_test = 'hello'
        print('Test (16) of `TrainTestSelect` class failed.')
        return 0
    except Exception:
        pass

    try:
        sam = TrainTestSelect(np.array([0,0,0,0,0,0,0,1,1,1,1]))
        sam.idx = 'hello'
        print('Test (17) of `TrainTestSelect` class failed.')
        return 0
    except Exception:
        pass

    # ##########################################################################

    # Tests of `TrainTestSelect.number`:

    # ##########################################################################

    idx_number = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,1])
    sampling = TrainTestSelect(idx_number, idx_test=[], random_seed=None, verbose=False)

    try:
        (idx_train, idx_test) = sampling.number(40, test_selection_option=1)
    except Exception:
        print('Test (01) of `TrainTestSelect.number` failed.')
        return 0

    try:
        (idx_train, idx_test) = sampling.number(70, test_selection_option=1)
    except Exception:
        print('Test (02) of `TrainTestSelect.number` failed.')
        return 0

    try:
        (idx_train, idx_test) = sampling.number(80, test_selection_option=1)
        print('Test (03) of `TrainTestSelect.number` failed.')
        return 0
    except Exception:
        pass

    try:
        (idx_train, idx_test) = sampling.number(40, test_selection_option=2)
    except Exception:
        print('Test (04) of `TrainTestSelect.number` failed.')
        return 0

    try:
        (idx_train, idx_test) = sampling.number(70, test_selection_option=2)
    except Exception:
        print('Test (05) of `TrainTestSelect.number` failed.')
        return 0

    try:
        (idx_train, idx_test) = sampling.number(80, test_selection_option=2)
        print('Test (06) of `TrainTestSelect.number` failed.')
        return 0
    except Exception:
        pass

    try:
        (idx_train, idx_test) = sampling.number(0, test_selection_option=2)
    except Exception:
        print('Test (07) of `TrainTestSelect.number` failed.')
        return 0

    try:
        (idx_train, idx_test) = sampling.number(-2, test_selection_option=2)
        print('Test (08) of `TrainTestSelect.number` failed.')
        return 0
    except Exception:
        pass

    try:
        (idx_train, idx_test) = sampling.number(102, test_selection_option=2)
        print('Test (09) of `TrainTestSelect.number` failed.')
        return 0
    except Exception:
        pass

    # ##########################################################################

    # Tests of `TrainTestSelect.percentage`:

    # ##########################################################################

    idx_percentage = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,1])
    sampling = TrainTestSelect(idx_percentage, idx_test=[], random_seed=None, verbose=False)

    try:
        (idx_train, idx_test) = sampling.percentage(0, test_selection_option=1)
    except Exception:
        print('Test (01) of `TrainTestSelect.percentage` failed.')
        return 0

    try:
        (idx_train, idx_test) = sampling.percentage(20, test_selection_option=1)
    except Exception:
        print('Test (02) of `TrainTestSelect.percentage` failed.')
        return 0

    try:
        (idx_train, idx_test) = sampling.percentage(60, test_selection_option=1)
    except Exception:
        print('Test (03) of `TrainTestSelect.percentage` failed.')
        return 0

    try:
        (idx_train, idx_test) = sampling.percentage(100, test_selection_option=1)
    except Exception:
        print('Test (04) of `TrainTestSelect.percentage` failed.')
        return 0

    try:
        (idx_train, idx_test) = sampling.percentage(10, test_selection_option=2)
    except Exception:
        print('Test (05) of `TrainTestSelect.percentage` failed.')
        return 0

    try:
        (idx_train, idx_test) = sampling.percentage(50, test_selection_option=2)
    except Exception:
        print('Test (06) of `TrainTestSelect.percentage` failed.')
        return 0

    try:
        (idx_train, idx_test) = sampling.percentage(60, test_selection_option=2)
        print('Test (07) of `TrainTestSelect.percentage` failed.')
        return 0
    except Exception:
        pass

    try:
        (idx_train, idx_test) = sampling.percentage(100, test_selection_option=2)
        print('Test (07) of `TrainTestSelect.percentage` failed.')
        return 0
    except Exception:
        pass

    # ##########################################################################

    # Tests of `TrainTestSelect.manual`:

    # ##########################################################################

    idx_manual = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,1])
    sampling = TrainTestSelect(idx_manual, idx_test=[], random_seed=None, verbose=False)

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

    try:
        (idx_train, idx_test) = sampling.manual({0:10, 1:10}, sampling_type='percentage', test_selection_option=1)
    except Exception:
        print('Test (04) of `TrainTestSelect.manual` failed.')
        return 0

    try:
        (idx_train, idx_test) = sampling.manual({0:50, 1:50}, sampling_type='percentage', test_selection_option=1)
    except Exception:
        print('Test (05) of `TrainTestSelect.manual` failed.')
        return 0

    try:
        (idx_train, idx_test) = sampling.manual({0:60, 1:60}, sampling_type='percentage', test_selection_option=1)
    except Exception:
        print('Test (06) of `TrainTestSelect.manual` failed.')
        return 0

    try:
        (idx_train, idx_test) = sampling.manual({0:20, 1:20}, sampling_type='number', test_selection_option=1)
        print('Test (07) of `TrainTestSelect.manual` failed.')
        return 0
    except Exception:
        pass

    try:
        (idx_train, idx_test) = sampling.manual({0:5, 1:6}, sampling_type='number', test_selection_option=1)
    except Exception:
        print('Test (08) of `TrainTestSelect.manual` failed.')
        return 0

    try:
        (idx_train, idx_test) = sampling.manual({0:2, 1:2}, sampling_type='number', test_selection_option=1)
    except Exception:
        print('Test (09) of `TrainTestSelect.manual` failed.')
        return 0

    try:
        (idx_train, idx_test) = sampling.manual({0:5, 1:5}, sampling_type='number', test_selection_option=1)
    except Exception:
        print('Test (10) of `TrainTestSelect.manual` failed.')
        return 0

    try:
        (idx_train, idx_test) = sampling.manual({0:2.2, 1:1}, sampling_type='number', test_selection_option=1)
        print('Test (11) of `TrainTestSelect.manual` failed.')
        return 0
    except Exception:
        pass

    try:
        (idx_train, idx_test) = sampling.manual({0:20, 1:-20}, sampling_type='percentage', test_selection_option=1)
        print('Test (12) of `TrainTestSelect.manual` failed.')
        return 0
    except Exception:
        pass

    try:
        (idx_train, idx_test) = sampling.manual({0:10, 1:10}, sampling_type='percentage', test_selection_option=2)
    except Exception:
        print('Test (13) of `TrainTestSelect.manual` failed.')
        return 0

    try:
        (idx_train, idx_test) = sampling.manual({0:50, 1:50}, sampling_type='percentage', test_selection_option=2)
    except Exception:
        print('Test (14) of `TrainTestSelect.manual` failed.')
        return 0

    try:
        (idx_train, idx_test) = sampling.manual({0:51, 1:10}, sampling_type='percentage', test_selection_option=2)
        print('Test (15) of `TrainTestSelect.manual` failed.')
        return 0
    except Exception:
        pass

    try:
        (idx_train, idx_test) = sampling.manual({0:15, 1:2}, sampling_type='number', test_selection_option=2)
        print('Test (16) of `TrainTestSelect.manual` failed.')
        return 0
    except Exception:
        pass

    try:
        (idx_train, idx_test) = sampling.manual({0:1, 1:0}, sampling_type='number', test_selection_option=2)
    except Exception:
        print('Test (17) of `TrainTestSelect.manual` failed.')
        return 0

    # ##########################################################################

    # Tests of `TrainTestSelect.random`:

    # ##########################################################################

    idx_random = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,1])
    sampling = TrainTestSelect(idx_random, idx_test=[], random_seed=None, verbose=False)

    try:
        (idx_train, idx_test) = sampling.random(40, test_selection_option=1)
    except Exception:
        print('Test (01) of `TrainTestSelect.random` failed.')
        return 0

    try:
        (idx_train, idx_test) = sampling.random(0, test_selection_option=1)
    except Exception:
        print('Test (02) of `TrainTestSelect.random` failed.')
        return 0

    try:
        (idx_train, idx_test) = sampling.random(51, test_selection_option=1)
    except Exception:
        print('Test (03) of `TrainTestSelect.random` failed.')
        return 0

    try:
        (idx_train, idx_test) = sampling.random(100, test_selection_option=1)
    except Exception:
        print('Test (04) of `TrainTestSelect.random` failed.')
        return 0

    try:
        (idx_train, idx_test) = sampling.random(101, test_selection_option=1)
        print('Test (05) of `TrainTestSelect.random` failed.')
        return 0
    except Exception:
        pass

    try:
        (idx_train, idx_test) = sampling.random(-1, test_selection_option=1)
        print('Test (06) of `TrainTestSelect.random` failed.')
        return 0
    except Exception:
        pass

    try:
        (idx_train, idx_test) = sampling.random(0, test_selection_option=2)
    except Exception:
        print('Test (07) of `TrainTestSelect.random` failed.')
        return 0

    try:
        (idx_train, idx_test) = sampling.random(10, test_selection_option=2)
    except Exception:
        print('Test (08) of `TrainTestSelect.random` failed.')
        return 0

    try:
        (idx_train, idx_test) = sampling.random(90, test_selection_option=2)
        print('Test (09) of `TrainTestSelect.random` failed.')
        return 0
    except Exception:
        pass

    try:
        (idx_train, idx_test) = sampling.random(51, test_selection_option=2)
        print('Test (10) of `TrainTestSelect.random` failed.')
        return 0
    except Exception:
        pass

    try:
        (idx_train, idx_test) = sampling.random(101, test_selection_option=2)
        print('Test (11) of `TrainTestSelect.random` failed.')
        return 0
    except Exception:
        pass

    try:
        idx_random = np.array([0,0,0,0,0,0,0,1,1,1,1,1,1,1,0,0,0,0,0,0,0])
        idx_test = [1,2,3,4,5,6]
        sampling = TrainTestSelect(idx_random, idx_test=idx_test, random_seed=None, verbose=False)
        (idx_train, idx_test) = sampling.random(70, test_selection_option=1)
    except Exception:
        print('Test (12) of `TrainTestSelect.random` failed.')
        return 0

    try:
        idx_random = np.array([0,0,0,0,0,0,0,1,1,1,1,1,1,1,0,0,0,0,0,0,0])
        idx_test = [1,2,3,4,5,6]
        sampling = TrainTestSelect(idx_random, idx_test=idx_test, random_seed=None, verbose=False)
        (idx_train, idx_test) = sampling.random(80, test_selection_option=1)
        print('Test (13) of `TrainTestSelect.random` failed.')
        return 0
    except Exception:
        pass

    try:
        idx_random = np.array([0,0,0,0,0,0,0,1,1,1,1,1,1,1,0,0,0,0,0,0,0])
        idx_test = [1,2,3,4,5,6]
        sampling = TrainTestSelect(idx_random, idx_test=idx_test, random_seed=None, verbose=False)
        (idx_train, idx_test) = sampling.random(70, test_selection_option=2)
    except Exception:
        print('Test (14) of `TrainTestSelect.random` failed.')
        return 0

    try:
        idx_random = np.array([0,0,0,0,0,0,0,1,1,1,1,1,1,1,0,0,0,0,0,0,0])
        idx_test = [1,2,3,4,5,6]
        sampling = TrainTestSelect(idx_random, idx_test=idx_test, random_seed=None, verbose=False)
        (idx_train, idx_test) = sampling.random(80, test_selection_option=2)
        print('Test (15) of `TrainTestSelect.random` failed.')
        return 0
    except Exception:
        pass

    print("Tests of `sampling` module passed.")
