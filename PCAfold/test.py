# Import available classes:
from PCAfold import DataSampler
from PCAfold import PCA

# Import moduless:
from PCAfold import preprocess
from PCAfold import reduction
from PCAfold import analysis

import numpy as np

def test():
    """
    This function runs all regression tests available.
    """

    # Functionalities within `preprocess` module:
    result_preprocess = test_preprocess()

    # Functionalities within `reduction` module:
    result_reduction = test_reduction()

    # Functionalities within `analysis` module:
    result_analysis = test_analysis()

    if ((result_preprocess == 0) or (result_reduction == 0) or (result_analysis == 0)):
        print('Tests of all modules failed.')
        return 0
    else:
        print('Tests of all modules passed.')

################################################################################
# ------------------------------------------------------------------------------
################################################################################
#
# Module: `preprocess`
#
################################################################################
# ------------------------------------------------------------------------------
################################################################################

def test_preprocess():
    """
    This function performs regression testing of ``preprocess`` module.
    """

    result_manipulation = test_manipulation()
    result_clustering = test_clustering()
    result_sampling = test_sampling()

    if ((result_manipulation == 0) or (result_clustering == 0) or (result_sampling == 0)):
        print('Tests of `preprocess` module failed.')
        return 0
    else:
        print('Tests of `preprocess` module passed.')
        return 1

################################################################################
#
# Data Manipulation
#
################################################################################

def test_manipulation():
    """
    This function performs regression testing of data manipulation functions
    within ``preprocess`` module.
    """

    return 1











################################################################################
#
# Data Clustering
#
################################################################################

def test_clustering():
    """
    This function performs regression testing of clustering functions within
    ``preprocess`` module.
    """

    # --------------------------------------------------------------------------

    # Check if `idx` output vectors are of type numpy.ndarray and of size (_,):

    # --------------------------------------------------------------------------
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
        idx_5 = preprocess.vqpca(np.array([[1,2,3,4,5,6,7,8,9,10],[2,3,4,5,6,7,8,9,10,11]]).T, k=2, n_components=1, scal_crit='NONE', idx_0=[], maximum_number_of_iterations=20, verbose=False)
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
        idx_6 = preprocess.source_bins(np.array([-100, -20, -0.1, 0, 0.1, 1, 10, 20, 200, 300, 400]), k=4, split_at_zero=True, verbose=False)
    except:
        print('Test of source_bins failed.')
        return 0
    if not isinstance(idx_6, np.ndarray):
        print('Test of source_bins failed.')
        return 0
    try:
        (n_observations,) = np.shape(idx_6)
    except:
        print('Test of source_bins failed.')
        return 0

    # Test degrade_clusters function:
    (idx, k) = preprocess.degrade_clusters([1,1,2,2,3,3], verbose=False)
    if np.min(idx) != 0:
        print('Test of degrade_clusters failed.')
        return 0
    if k != 3:
        print('Test of degrade_clusters failed.')
        return 0

    print('Tests of clustering functionalities within `preprocess` module passed.')
    return 1

################################################################################
#
# Data Sampling
#
################################################################################

def test_sampling():
    """
    This function performs regression testing of sampling functions within
    ``preprocess`` module.
    """

    # --------------------------------------------------------------------------

    # Basic sanity tests:

    # --------------------------------------------------------------------------

    # Check that sizes of idx_train and idx_test always sum up to the total numer
    # of observations when test_selection_option=1:
    try:
        idx = np.array([0,0,0,0,0,0,0,1,1,1,1])
        sam = DataSampler(idx)
        (idx_train, idx_test) = sam.number(10)
        n_observations = len(idx)
        if np.size(idx_test) + np.size(idx_train) != n_observations:
            print('Sanity test of sampling functions (01) failed.')
            return 0
        (idx_train, idx_test) = sam.percentage(10)
        if np.size(idx_test) + np.size(idx_train) != n_observations:
            print('Sanity test of sampling functions (02) failed.')
            return 0
        (idx_train, idx_test) = sam.manual({0:1,1:4}, sampling_type='number')
        if np.size(idx_test) + np.size(idx_train) != n_observations:
            print('Sanity test of sampling functions (03) failed.')
            return 0
        (idx_train, idx_test) = sam.random(10)
        if np.size(idx_test) + np.size(idx_train) != n_observations:
            print('Sanity test of sampling functions (04) failed.')
            return 0
    except Exception:
        print('Sanity test of sampling functions (05) failed.')
        return 0

    # Check that sizes of idx_train and idx_test sum up to less than the total numer
    # for these cases with test_selection_option=2:
    try:
        idx = np.array([0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1])
        sam = DataSampler(idx)
        (idx_train, idx_test) = sam.number(10, test_selection_option=2)
        n_observations = len(idx)
        if np.size(idx_test) + np.size(idx_train) >= n_observations:
            print('Sanity test of sampling functions (06) failed.')
            return 0
        (idx_train, idx_test) = sam.percentage(10, test_selection_option=2)
        if np.size(idx_test) + np.size(idx_train) >= n_observations:
            print('Sanity test of sampling functions (07) failed.')
            return 0
        (idx_train, idx_test) = sam.manual({0:1,1:4}, sampling_type='number', test_selection_option=2)
        if np.size(idx_test) + np.size(idx_train) >= n_observations:
            print('Sanity test of sampling functions (08) failed.')
            return 0
        (idx_train, idx_test) = sam.random(10, test_selection_option=2)
        if np.size(idx_test) + np.size(idx_train) >= n_observations:
            print('Sanity test of sampling functions (09) failed.')
            return 0
    except Exception:
        print('Sanity test of sampling functions (10) failed.')
        return 0

    # Check that indices in idx_train are never in idx_test and vice versa:
    try:
        idx = np.array([0,0,0,0,0,0,0,0,0,1,1,1,1,2,2,2,2,2,2,2,2,5,5,5,5,10,10,10,10,10])
        sam = DataSampler(idx)

        (idx_train, idx_test) = sam.number(40, test_selection_option=1)
        if len(np.setdiff1d(idx_test, idx_train)) == 0:
            print('Sanity test of sampling functions (11) failed.')
            return 0

        (idx_train, idx_test) = sam.percentage(40, test_selection_option=1)
        if len(np.setdiff1d(idx_test, idx_train)) == 0:
            print('Sanity test of sampling functions (12) failed.')
            return 0

        (idx_train, idx_test) = sam.manual({0:1,1:4,2:1,3:1,4:1}, sampling_type='number', test_selection_option=1)
        if len(np.setdiff1d(idx_test, idx_train)) == 0:
            print('Sanity test of sampling functions (13) failed.')
            return 0

        (idx_train, idx_test) = sam.random(40, test_selection_option=1)
        if len(np.setdiff1d(idx_test, idx_train)) == 0:
            print('Sanity test of sampling functions (14) failed.')
            return 0

        (idx_train, idx_test) = sam.number(40, test_selection_option=2)
        if len(np.setdiff1d(idx_test, idx_train)) == 0:
            print('Sanity test of sampling functions (15) failed.')
            return 0

        (idx_train, idx_test) = sam.percentage(40, test_selection_option=2)
        if len(np.setdiff1d(idx_test, idx_train)) == 0:
            print('Sanity test of sampling functions (16) failed.')
            return 0

        (idx_train, idx_test) = sam.manual({0:1,1:1,2:1,3:1,4:1}, sampling_type='number', test_selection_option=2)
        if len(np.setdiff1d(idx_test, idx_train)) == 0:
            print('Sanity test of sampling functions (17) failed.')
            return 0

        (idx_train, idx_test) = sam.random(10, test_selection_option=2)
        if len(np.setdiff1d(idx_test, idx_train)) == 0:
            print('Sanity test of sampling functions (18) failed.')
            return 0

        (idx_train, idx_test) = sam.manual({0:10,1:10,2:60,3:10,4:80}, sampling_type='percentage', test_selection_option=1)
        if len(np.setdiff1d(idx_test, idx_train)) == 0:
            print('Sanity test of sampling functions (19) failed.')
            return 0

        (idx_train, idx_test) = sam.manual({0:40,1:40,2:40,3:40,4:40}, sampling_type='percentage', test_selection_option=2)
        if len(np.setdiff1d(idx_test, idx_train)) == 0:
            print('Sanity test of sampling functions (20) failed.')
            return 0

    except Exception:
        print('Sanity test of sampling functions (21) failed.')
        return 0

    # Check that idx_test can be predefined by user and check that idx_test
    # returned is always the same as the user predefined vector:
    try:

        idx = np.array([0,0,0,0,0,0,0,0,0,1,1,1,1,2,2,2,2,2,2,2,2,5,5,5,5,10,10,10,10,10])
        idx_test_predefined = [0, 10, 20, 21]
        sam = DataSampler(idx, idx_test=idx_test_predefined)

        (idx_train, idx_test) = sam.number(40, test_selection_option=1)
        if len(np.setdiff1d(idx_test, idx_train)) == 0:
            print('Sanity test of sampling functions (22) failed.')
            return 0
        if not np.array_equal(idx_test_predefined, idx_test):
            print('Sanity test of sampling functions (23) failed.')
            return 0

        (idx_train, idx_test) = sam.percentage(40, test_selection_option=1)
        if len(np.setdiff1d(idx_test, idx_train)) == 0:
            print('Sanity test of sampling functions (24) failed.')
            return 0
        if not np.array_equal(idx_test_predefined, idx_test):
            print('Sanity test of sampling functions (25) failed.')
            return 0

        (idx_train, idx_test) = sam.manual({0:1,1:1,2:1,3:1,4:1}, sampling_type='number', test_selection_option=1)
        if len(np.setdiff1d(idx_test, idx_train)) == 0:
            print('Sanity test of sampling functions (26) failed.')
            return 0
        if not np.array_equal(idx_test_predefined, idx_test):
            print('Sanity test of sampling functions (27) failed.')
            return 0

        (idx_train, idx_test) = sam.random(40, test_selection_option=1)
        if len(np.setdiff1d(idx_test, idx_train)) == 0:
            print('Sanity test of sampling functions (28) failed.')
            return 0
        if not np.array_equal(idx_test_predefined, idx_test):
            print('Sanity test of sampling functions (29) failed.')
            return 0

        (idx_train, idx_test) = sam.number(40, test_selection_option=2)
        if len(np.setdiff1d(idx_test, idx_train)) == 0:
            print('Sanity test of sampling functions (30) failed.')
            return 0
        if not np.array_equal(idx_test_predefined, idx_test):
            print('Sanity test of sampling functions (31) failed.')
            return 0

        (idx_train, idx_test) = sam.percentage(40, test_selection_option=2)
        if len(np.setdiff1d(idx_test, idx_train)) == 0:
            print('Sanity test of sampling functions (32) failed.')
            return 0
        if not np.array_equal(idx_test_predefined, idx_test):
            print('Sanity test of sampling functions (33) failed.')
            return 0

        (idx_train, idx_test) = sam.manual({0:1,1:1,2:1,3:1,4:1}, sampling_type='number', test_selection_option=2)
        if len(np.setdiff1d(idx_test, idx_train)) == 0:
            print('Sanity test of sampling functions (34) failed.')
            return 0
        if not np.array_equal(idx_test_predefined, idx_test):
            print('Sanity test of sampling functions (35) failed.')
            return 0

        (idx_train, idx_test) = sam.random(10, test_selection_option=2)
        if len(np.setdiff1d(idx_test, idx_train)) == 0:
            print('Sanity test of sampling functions (36) failed.')
            return 0
        if not np.array_equal(idx_test_predefined, idx_test):
            print('Sanity test of sampling functions (37) failed.')
            return 0

        (idx_train, idx_test) = sam.manual({0:20,1:80,2:60,3:80,4:80}, sampling_type='percentage', test_selection_option=1)
        if len(np.setdiff1d(idx_test, idx_train)) == 0:
            print('Sanity test of sampling functions (38) failed.')
            return 0
        if not np.array_equal(idx_test_predefined, idx_test):
            print('Sanity test of sampling functions (39) failed.')
            return 0

        (idx_train, idx_test) = sam.manual({0:40,1:40,2:40,3:40,4:40}, sampling_type='percentage', test_selection_option=2)
        if len(np.setdiff1d(idx_test, idx_train)) == 0:
            print('Sanity test of sampling functions (40) failed.')
            return 0
        if not np.array_equal(idx_test_predefined, idx_test):
            print('Sanity test of sampling functions (41) failed.')
            return 0

    except Exception:
        print('Sanity test of sampling functions (42) failed.')
        return 0

    # --------------------------------------------------------------------------

    # Tests of `DataSampler` class init:

    # --------------------------------------------------------------------------

    try:
        DataSampler(np.array([0,0,0,0,0,0,0,1,1,1,1]), idx_test=[], random_seed=0.4, verbose=False)
        print('Test (02) of `DataSampler` class failed.')
        return 0
    except Exception:
        pass

    try:
        DataSampler(np.array([0,0,0,0,0,0,0,1,1,1,1]), idx_test=[], random_seed=100, verbose=2)
        print('Test (03) of `DataSampler` class failed.')
        return 0
    except Exception:
        pass

    try:
        DataSampler(np.array([0,0,0,1,1]), idx_test=np.array([1,2,3,4,5,6,7,8]), random_seed=100, verbose=False)
        print('Test (04) of `DataSampler` class failed.')
        return 0
    except Exception:
        pass

    try:
        DataSampler(np.array([]), idx_test=[], random_seed=None, verbose=False)
        print('Test (05) of `DataSampler` class failed.')
        return 0
    except Exception:
        pass

    try:
        DataSampler(np.array([0,0,0,0,0,0,0,1,1,1,1]))
    except Exception:
        print('Test (06) of `DataSampler` class failed.')
        return 0

    try:
        DataSampler(np.array([1,1,1,1,2,2,2,2]))
    except Exception:
        print('Test (07) of `DataSampler` class failed.')
        return 0

    try:
        DataSampler(np.array([0,0,0,0,0,0,0,1,1,1,1]), idx_test=np.array([0,0,0,0,0,0,0,1,1,1,2,2,2,2,2,2]))
    except Exception:
        print('Test (08) of `DataSampler` class failed.')
        return 0

    try:
        sam = DataSampler(np.array([0,0,0,0,0,0,0,1,1,1,1]))
        sam.idx = np.array([1,1,1,1,1,1,2,2,2,2,2,2,2,2])
        sam.idx_test = np.arange(1,10,1)
        sam.random_seed = 100
        sam.random_seed = None
        sam.random_seed = -1
        sam.verbose = False
        sam.verbose = True
        sam.idx_test = []
    except Exception:
        print('Test (09) of `DataSampler` class failed.')
        return 0

    try:
        sam = DataSampler(np.array([0,0,0,0,0,0,0,1,1,1,1]))
        sam.idx_test = np.arange(1,100,1)
        print('Test (10) of `DataSampler` class failed.')
        return 0
    except Exception:
        pass

    try:
        sam = DataSampler(np.array([0,0,0,0,0,0,0,1,1,1,1]))
        sam.random_seed = 10.1
        print('Test (11) of `DataSampler` class failed.')
        return 0
    except Exception:
        pass

    try:
        sam = DataSampler(np.array([0,0,0,0,0,0,0,1,1,1,1]))
        sam.random_seed = False
        print('Test (12) of `DataSampler` class failed.')
        return 0
    except Exception:
        pass

    try:
        sam = DataSampler(np.array([0,0,0,0,0,0,0,1,1,1,1]))
        sam.verbose = 10
        print('Test (13) of `DataSampler` class failed.')
        return 0
    except Exception:
        pass

    try:
        sam = DataSampler(np.array([0,0,0,0,0,0,0,1,1,1,1]))
        sam.idx = []
        print('Test (14) of `DataSampler` class failed.')
        return 0
    except Exception:
        pass

    try:
        sam = DataSampler(np.array([0,0,0,0,0,0,0,1,1,1,1]))
        sam.idx_test = [0,1,2,3,4,5,6]
        sam.idx = np.array([0,1])
        print('Test (15) of `DataSampler` class failed.')
        return 0
    except Exception:
        pass

    try:
        sam = DataSampler(np.array([0,0,0,0,0,0,0,1,1,1,1]))
        sam.idx_test = 'hello'
        print('Test (16) of `DataSampler` class failed.')
        return 0
    except Exception:
        pass

    try:
        sam = DataSampler(np.array([0,0,0,0,0,0,0,1,1,1,1]))
        sam.idx = 'hello'
        print('Test (17) of `DataSampler` class failed.')
        return 0
    except Exception:
        pass

    # --------------------------------------------------------------------------

    # Tests of `DataSampler.number`:

    # --------------------------------------------------------------------------

    idx_number = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,1])
    sampling = DataSampler(idx_number, idx_test=[], random_seed=None, verbose=False)

    try:
        (idx_train, idx_test) = sampling.number(40, test_selection_option=1)
    except Exception:
        print('Test (01) of `DataSampler.number` failed.')
        return 0

    try:
        (idx_train, idx_test) = sampling.number(70, test_selection_option=1)
    except Exception:
        print('Test (02) of `DataSampler.number` failed.')
        return 0

    try:
        (idx_train, idx_test) = sampling.number(80, test_selection_option=1)
        print('Test (03) of `DataSampler.number` failed.')
        return 0
    except Exception:
        pass

    try:
        (idx_train, idx_test) = sampling.number(40, test_selection_option=2)
    except Exception:
        print('Test (04) of `DataSampler.number` failed.')
        return 0

    try:
        (idx_train, idx_test) = sampling.number(70, test_selection_option=2)
    except Exception:
        print('Test (05) of `DataSampler.number` failed.')
        return 0

    try:
        (idx_train, idx_test) = sampling.number(80, test_selection_option=2)
        print('Test (06) of `DataSampler.number` failed.')
        return 0
    except Exception:
        pass

    try:
        (idx_train, idx_test) = sampling.number(0, test_selection_option=2)
    except Exception:
        print('Test (07) of `DataSampler.number` failed.')
        return 0

    try:
        (idx_train, idx_test) = sampling.number(-2, test_selection_option=2)
        print('Test (08) of `DataSampler.number` failed.')
        return 0
    except Exception:
        pass

    try:
        (idx_train, idx_test) = sampling.number(102, test_selection_option=2)
        print('Test (09) of `DataSampler.number` failed.')
        return 0
    except Exception:
        pass

    try:
        sampling.idx_test = np.array([0,0,1,10,20,30,31])
        (idx_train, idx_test) = sampling.number(40, test_selection_option=1)
    except Exception:
        print('Test (10) of `DataSampler.number` failed.')
        return 0

    try:
        sampling.idx_test = np.array([0,1,2,3,10,20,21,22,23,24,30,31])
        (idx_train, idx_test) = sampling.number(46, test_selection_option=1)
        print('Test (11) of `DataSampler.number` failed.')
        return 0
    except Exception:
        pass

    # --------------------------------------------------------------------------

    # Tests of `DataSampler.percentage`:

    # --------------------------------------------------------------------------

    idx_percentage = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,1])
    sampling = DataSampler(idx_percentage, idx_test=[], random_seed=None, verbose=False)

    try:
        (idx_train, idx_test) = sampling.percentage(0, test_selection_option=1)
    except Exception:
        print('Test (01) of `DataSampler.percentage` failed.')
        return 0

    try:
        (idx_train, idx_test) = sampling.percentage(20, test_selection_option=1)
    except Exception:
        print('Test (02) of `DataSampler.percentage` failed.')
        return 0

    try:
        (idx_train, idx_test) = sampling.percentage(60, test_selection_option=1)
    except Exception:
        print('Test (03) of `DataSampler.percentage` failed.')
        return 0

    try:
        (idx_train, idx_test) = sampling.percentage(100, test_selection_option=1)
    except Exception:
        print('Test (04) of `DataSampler.percentage` failed.')
        return 0

    try:
        (idx_train, idx_test) = sampling.percentage(10, test_selection_option=2)
    except Exception:
        print('Test (05) of `DataSampler.percentage` failed.')
        return 0

    try:
        (idx_train, idx_test) = sampling.percentage(50, test_selection_option=2)
    except Exception:
        print('Test (06) of `DataSampler.percentage` failed.')
        return 0

    try:
        (idx_train, idx_test) = sampling.percentage(60, test_selection_option=2)
        print('Test (07) of `DataSampler.percentage` failed.')
        return 0
    except Exception:
        pass

    try:
        (idx_train, idx_test) = sampling.percentage(100, test_selection_option=2)
        print('Test (08) of `DataSampler.percentage` failed.')
        return 0
    except Exception:
        pass

    try:
        sampling.idx_test = np.array([0,0,1,10,20,30,31])
        (idx_train, idx_test) = sampling.percentage(40, test_selection_option=1)
    except Exception:
        print('Test (09) of `DataSampler.percentage` failed.')
        return 0

    try:
        sampling.idx_test = np.array([0,1,2,3,10,20,21,22,23,24,30,31])
        (idx_train, idx_test) = sampling.percentage(80, test_selection_option=1)
        print('Test (10) of `DataSampler.percentage` failed.')
        return 0
    except Exception:
        pass

    # --------------------------------------------------------------------------

    # Tests of `DataSampler.manual`:

    # --------------------------------------------------------------------------

    idx_manual = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,1])
    sampling = DataSampler(idx_manual, idx_test=[], random_seed=None, verbose=False)

    try:
        (idx_train, idx_test) = sampling.manual({1:1, 2:1})
        print('Test (01) of `DataSampler.manual` failed.')
        return 0
    except Exception:
        pass

    try:
        (idx_train, idx_test) = sampling.manual({0:1, 1:1})
    except Exception:
        print('Test (02) of `DataSampler.manual` failed.')
        return 0

    try:
        (idx_train, idx_test) = sampling.manual({0:1, 1:1}, sampling_type='perc')
        print('Test (03) of `DataSampler.manual` failed.')
        return 0
    except Exception:
        pass

    try:
        (idx_train, idx_test) = sampling.manual({0:10, 1:10}, sampling_type='percentage', test_selection_option=1)
    except Exception:
        print('Test (04) of `DataSampler.manual` failed.')
        return 0

    try:
        (idx_train, idx_test) = sampling.manual({0:50, 1:50}, sampling_type='percentage', test_selection_option=1)
    except Exception:
        print('Test (05) of `DataSampler.manual` failed.')
        return 0

    try:
        (idx_train, idx_test) = sampling.manual({0:60, 1:60}, sampling_type='percentage', test_selection_option=1)
    except Exception:
        print('Test (06) of `DataSampler.manual` failed.')
        return 0

    try:
        (idx_train, idx_test) = sampling.manual({0:20, 1:20}, sampling_type='number', test_selection_option=1)
        print('Test (07) of `DataSampler.manual` failed.')
        return 0
    except Exception:
        pass

    try:
        (idx_train, idx_test) = sampling.manual({0:5, 1:6}, sampling_type='number', test_selection_option=1)
    except Exception:
        print('Test (08) of `DataSampler.manual` failed.')
        return 0

    try:
        (idx_train, idx_test) = sampling.manual({0:2, 1:2}, sampling_type='number', test_selection_option=1)
    except Exception:
        print('Test (09) of `DataSampler.manual` failed.')
        return 0

    try:
        (idx_train, idx_test) = sampling.manual({0:5, 1:5}, sampling_type='number', test_selection_option=1)
    except Exception:
        print('Test (10) of `DataSampler.manual` failed.')
        return 0

    try:
        (idx_train, idx_test) = sampling.manual({0:2.2, 1:1}, sampling_type='number', test_selection_option=1)
        print('Test (11) of `DataSampler.manual` failed.')
        return 0
    except Exception:
        pass

    try:
        (idx_train, idx_test) = sampling.manual({0:20, 1:-20}, sampling_type='percentage', test_selection_option=1)
        print('Test (12) of `DataSampler.manual` failed.')
        return 0
    except Exception:
        pass

    try:
        (idx_train, idx_test) = sampling.manual({0:10, 1:10}, sampling_type='percentage', test_selection_option=2)
    except Exception:
        print('Test (13) of `DataSampler.manual` failed.')
        return 0

    try:
        (idx_train, idx_test) = sampling.manual({0:50, 1:50}, sampling_type='percentage', test_selection_option=2)
    except Exception:
        print('Test (14) of `DataSampler.manual` failed.')
        return 0

    try:
        (idx_train, idx_test) = sampling.manual({0:51, 1:10}, sampling_type='percentage', test_selection_option=2)
        print('Test (15) of `DataSampler.manual` failed.')
        return 0
    except Exception:
        pass

    try:
        (idx_train, idx_test) = sampling.manual({0:15, 1:2}, sampling_type='number', test_selection_option=2)
        print('Test (16) of `DataSampler.manual` failed.')
        return 0
    except Exception:
        pass

    try:
        (idx_train, idx_test) = sampling.manual({0:1, 1:0}, sampling_type='number', test_selection_option=2)
    except Exception:
        print('Test (17) of `DataSampler.manual` failed.')
        return 0

    try:
        sampling.idx_test = np.array([0,0,1,10,20,30,31])
        (idx_train, idx_test) = sampling.manual({0:2, 1:2}, sampling_type='number', test_selection_option=1)
    except Exception:
        print('Test (18) of `DataSampler.manual` failed.')
        return 0

    try:
        sampling.idx_test = np.array([0,1,2,3,10,20,21,22,23,24,30,31])
        (idx_train, idx_test) = sampling.manual({0:2, 1:8}, sampling_type='number', test_selection_option=1)
        print('Test (19) of `DataSampler.manual` failed.')
        return 0
    except Exception:
        pass

    try:
        sampling.idx_test = np.array([0,1,2,3,10,20,21,22,23,24,30,31])
        (idx_train, idx_test) = sampling.manual({0:70, 1:10}, sampling_type='percentage', test_selection_option=1)
    except Exception:
        print('Test (20) of `DataSampler.manual` failed.')
        return 0

    try:
        sampling.idx_test = np.array([0,1,2,3,10,20,21,22,23,24,30,31])
        (idx_train, idx_test) = sampling.manual({0:75, 1:10}, sampling_type='number', test_selection_option=1)
        print('Test (21) of `DataSampler.manual` failed.')
        return 0
    except Exception:
        pass

    # --------------------------------------------------------------------------

    # Tests of `DataSampler.random`:

    # --------------------------------------------------------------------------

    idx_random = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,1])
    sampling = DataSampler(idx_random, idx_test=[], random_seed=None, verbose=False)

    try:
        (idx_train, idx_test) = sampling.random(40, test_selection_option=1)
    except Exception:
        print('Test (01) of `DataSampler.random` failed.')
        return 0

    try:
        (idx_train, idx_test) = sampling.random(0, test_selection_option=1)
    except Exception:
        print('Test (02) of `DataSampler.random` failed.')
        return 0

    try:
        (idx_train, idx_test) = sampling.random(51, test_selection_option=1)
    except Exception:
        print('Test (03) of `DataSampler.random` failed.')
        return 0

    try:
        (idx_train, idx_test) = sampling.random(100, test_selection_option=1)
    except Exception:
        print('Test (04) of `DataSampler.random` failed.')
        return 0

    try:
        (idx_train, idx_test) = sampling.random(101, test_selection_option=1)
        print('Test (05) of `DataSampler.random` failed.')
        return 0
    except Exception:
        pass

    try:
        (idx_train, idx_test) = sampling.random(-1, test_selection_option=1)
        print('Test (06) of `DataSampler.random` failed.')
        return 0
    except Exception:
        pass

    try:
        (idx_train, idx_test) = sampling.random(0, test_selection_option=2)
    except Exception:
        print('Test (07) of `DataSampler.random` failed.')
        return 0

    try:
        (idx_train, idx_test) = sampling.random(10, test_selection_option=2)
    except Exception:
        print('Test (08) of `DataSampler.random` failed.')
        return 0

    try:
        (idx_train, idx_test) = sampling.random(90, test_selection_option=2)
        print('Test (09) of `DataSampler.random` failed.')
        return 0
    except Exception:
        pass

    try:
        (idx_train, idx_test) = sampling.random(51, test_selection_option=2)
        print('Test (10) of `DataSampler.random` failed.')
        return 0
    except Exception:
        pass

    try:
        (idx_train, idx_test) = sampling.random(101, test_selection_option=2)
        print('Test (11) of `DataSampler.random` failed.')
        return 0
    except Exception:
        pass

    try:
        idx_random = np.array([0,0,0,0,0,0,0,1,1,1,1,1,1,1,0,0,0,0,0,0,0])
        idx_test = [1,2,3,4,5,6]
        sampling = DataSampler(idx_random, idx_test=idx_test, random_seed=None, verbose=False)
        (idx_train, idx_test) = sampling.random(70, test_selection_option=1)
    except Exception:
        print('Test (12) of `DataSampler.random` failed.')
        return 0

    try:
        idx_random = np.array([0,0,0,0,0,0,0,1,1,1,1,1,1,1,0,0,0,0,0,0,0])
        idx_test = [1,2,3,4,5,6]
        sampling = DataSampler(idx_random, idx_test=idx_test, random_seed=None, verbose=False)
        (idx_train, idx_test) = sampling.random(80, test_selection_option=1)
        print('Test (13) of `DataSampler.random` failed.')
        return 0
    except Exception:
        pass

    try:
        idx_random = np.array([0,0,0,0,0,0,0,1,1,1,1,1,1,1,0,0,0,0,0,0,0])
        idx_test = [1,2,3,4,5,6]
        sampling = DataSampler(idx_random, idx_test=idx_test, random_seed=None, verbose=False)
        (idx_train, idx_test) = sampling.random(70, test_selection_option=2)
    except Exception:
        print('Test (14) of `DataSampler.random` failed.')
        return 0

    try:
        idx_random = np.array([0,0,0,0,0,0,0,1,1,1,1,1,1,1,0,0,0,0,0,0,0])
        idx_test = [1,2,3,4,5,6]
        sampling = DataSampler(idx_random, idx_test=idx_test, random_seed=None, verbose=False)
        (idx_train, idx_test) = sampling.random(80, test_selection_option=2)
        print('Test (15) of `DataSampler.random` failed.')
        return 0
    except Exception:
        pass

    print("Tests of sampling functionalities within `preprocess` module passed.")
    return 1

################################################################################
# ------------------------------------------------------------------------------
################################################################################
#
# Module: `reduction`
#
################################################################################
# ------------------------------------------------------------------------------
################################################################################

def test_reduction():
    """
    This function performs regression testing of `reduction` module.
    """

    print('Tests of `reduction` module passed.')
    return 1









################################################################################
# ------------------------------------------------------------------------------
################################################################################
#
# Module: `analysis`
#
################################################################################
# ------------------------------------------------------------------------------
################################################################################

def test_analysis():
    """
    This function performs regression testing of ``analysis`` module.
    """

    print('Tests of `analysis` module passed.')
    return 1
