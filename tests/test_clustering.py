import unittest
import numpy as np
from PCAfold import preprocess


class TestClustering(unittest.TestCase):

################################################################################
#
# Clustering functions
#
################################################################################

    def test_variable_bins_allowed_calls(self):

        try:
            (idx, borders) = preprocess.variable_bins(np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]), 4, verbose=False)
            self.assertTrue(True)
        except:
            self.assertTrue(False)

        self.assertTrue(isinstance(idx, np.ndarray))
        self.assertTrue(idx.ndim == 1)

    def test_variable_bins_not_allowed_calls(self):

        with self.assertRaises(ValueError):
            (idx, borders) = preprocess.variable_bins(np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]), 0, verbose=False)

        with self.assertRaises(ValueError):
            (idx, borders) = preprocess.variable_bins(np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]), -1, verbose=False)

        with self.assertRaises(ValueError):
            (idx, borders) = preprocess.variable_bins(np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]), 4, verbose=1)

        with self.assertRaises(ValueError):
            (idx, borders) = preprocess.variable_bins(np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]), 4, verbose='True')

    def test_predefined_variable_bins_allowed_calls(self):

        try:
            (idx, borders) = preprocess.predefined_variable_bins(np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]), [3.5, 8.5], verbose=False)
            self.assertTrue(True)
        except:
            self.assertTrue(False)

        self.assertTrue(isinstance(idx, np.ndarray))
        self.assertTrue(idx.ndim == 1)

    def test_predefined_variable_bins_not_allowed_calls(self):

        with self.assertRaises(ValueError):
            (idx, borders) = preprocess.predefined_variable_bins(np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]), [3, 11], verbose=False)

        with self.assertRaises(ValueError):
            (idx, borders) = preprocess.predefined_variable_bins(np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]), [0, 6], verbose=False)

        with self.assertRaises(ValueError):
            (idx, borders) = preprocess.predefined_variable_bins(np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]), [3, 8], verbose=1)

        with self.assertRaises(ValueError):
            (idx, borders) = preprocess.predefined_variable_bins(np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]), [3, 8], verbose='True')

    def test_mixture_fraction_bins_allowed_calls(self):

        try:
            (idx, borders) = preprocess.mixture_fraction_bins(np.array([0.1, 0.15, 0.2, 0.25, 0.6, 0.8, 1]), 2, 0.2)
            self.assertTrue(True)
        except:
            self.assertTrue(False)

        try:
            (idx, borders) = preprocess.mixture_fraction_bins(np.array([0.1, 0.15, 0.2, 0.25, 0.6, 0.8, 1]), 1, 0.2)
            self.assertTrue(True)
        except:
            self.assertTrue(False)

        self.assertTrue(isinstance(idx, np.ndarray))
        self.assertTrue(idx.ndim == 1)

    def test_mixture_fraction_bins_not_allowed_calls(self):

        with self.assertRaises(ValueError):
            (idx, borders) = preprocess.mixture_fraction_bins(np.array([0.1, 0.15, 0.2, 0.25, 0.6, 0.8, 1]), 0, 0.2)

        with self.assertRaises(ValueError):
            (idx, borders) = preprocess.mixture_fraction_bins(np.array([0.1, 0.15, 0.2, 0.25, 0.6, 0.8, 1]), -1, 0.2)

        with self.assertRaises(ValueError):
            (idx, borders) = preprocess.mixture_fraction_bins(np.array([0.1, 0.15, 0.2, 0.25, 0.6, 0.8, 1]), 2, 0.2, verbose=1)

        with self.assertRaises(ValueError):
            (idx, borders) = preprocess.mixture_fraction_bins(np.array([0.1, 0.15, 0.2, 0.25, 0.6, 0.8, 1]), 2, 0.2, verbose='True')

    def test_zero_neighborhood_bins_allowed_calls(self):

        try:
            (idx, borders) = preprocess.zero_neighborhood_bins(np.array([-100, -20, -0.1, 0, 0.1, 1, 10, 20, 200, 300, 400]), k=4, split_at_zero=True, verbose=False)
            self.assertTrue(True)
        except:
            self.assertTrue(False)

        self.assertTrue(isinstance(idx, np.ndarray))
        self.assertTrue(idx.ndim == 1)

    def test_zero_neighborhood_bins_not_allowed_calls(self):

        with self.assertRaises(ValueError):
            (idx, borders) = preprocess.zero_neighborhood_bins(np.array([-100, -20, -0.1, 0, 0.1, 1, 10, 20, 200, 300, 400]), k=0, split_at_zero=True, verbose=False)

        with self.assertRaises(ValueError):
            (idx, borders) = preprocess.zero_neighborhood_bins(np.array([-100, -20, -0.1, 0, 0.1, 1, 10, 20, 200, 300, 400]), k=-1, split_at_zero=True, verbose=False)

        with self.assertRaises(ValueError):
            (idx, borders) = preprocess.zero_neighborhood_bins(np.array([-100, -20, -0.1, 0, 0.1, 1, 10, 20, 200, 300, 400]), k=4, split_at_zero=True, verbose=1)

        with self.assertRaises(ValueError):
            (idx, borders) = preprocess.zero_neighborhood_bins(np.array([-100, -20, -0.1, 0, 0.1, 1, 10, 20, 200, 300, 400]), k=4, split_at_zero=True, verbose='True')

################################################################################
#
# Auxiliary functions
#
################################################################################

    def test_degrade_clusters_allowed_calls(self):

        try:
            idx_undegraded = [1, 1, 2, 2, 3, 3]
            idx_degraded = [0, 0, 1, 1, 2, 2]
            (idx, k) = preprocess.degrade_clusters(idx_undegraded, verbose=False)
            self.assertTrue(np.min(idx) == 0)
            self.assertTrue(k == 3)
            self.assertTrue(list(idx) == idx_degraded)
        except:
            self.assertTrue(False)

        try:
            idx_undegraded = [-1, -1, 1, 1, 2, 2, 3, 3]
            idx_degraded = [0, 0, 1, 1, 2, 2, 3, 3]
            (idx, k) = preprocess.degrade_clusters(idx_undegraded, verbose=False)
            self.assertTrue(np.min(idx) == 0)
            self.assertTrue(k == 4)
            self.assertTrue(list(idx) == idx_degraded)
        except:
            self.assertTrue(False)

        try:
            idx_undegraded = [-1, 1, 3, -1, 1, 1, 2, 2, 3, 3]
            idx_degraded = [0, 1, 3, 0, 1, 1, 2, 2, 3, 3]
            (idx, k) = preprocess.degrade_clusters(idx_undegraded, verbose=False)
            self.assertTrue(np.min(idx) == 0)
            self.assertTrue(k == 4)
            self.assertTrue(list(idx) == idx_degraded)
        except:
            self.assertTrue(False)

        try:
            idx = np.array([-1,-1,0,0,0,0,1,1,1,1,5])
            (idx, k) = preprocess.degrade_clusters(idx, verbose=False)
            self.assertTrue(np.min(idx) == 0)
            self.assertTrue(k == 4)
        except:
            self.assertTrue(False)

    def test_degrade_clusters_not_allowed_calls(self):

        idx_test = [0,0,0,1,1,1,True,2,2,2]
        with self.assertRaises(ValueError):
            (idx, k) = preprocess.degrade_clusters(idx_test, verbose=False)

        idx_test = [0,0,0,1,1,1,5.1,2,2,2]
        with self.assertRaises(ValueError):
            (idx, k) = preprocess.degrade_clusters(idx_test, verbose=False)

        idx_test = np.array([0,0,0,1.1,1,1,2,2,2])
        with self.assertRaises(ValueError):
            (idx, k) = preprocess.degrade_clusters(idx_test, verbose=False)

        idx_test = np.array([-1.2,0,0,0,1,1,1,2,2,2])
        with self.assertRaises(ValueError):
            (idx, k) = preprocess.degrade_clusters(idx_test, verbose=False)

        with self.assertRaises(ValueError):
            (idx, k) = preprocess.degrade_clusters(1, verbose=False)

        with self.assertRaises(ValueError):
            (idx, k) = preprocess.degrade_clusters('list', verbose=False)

    def test_flip_clusters_allowed_calls(self):

        try:
            idx_unflipped = np.array([0,0,0,1,1,1,2,2,2])
            idx_flipped = np.array([0,0,0,2,2,2,1,1,1])
            idx = preprocess.flip_clusters(idx_unflipped, dictionary={1:2, 2:1})
            comparison = idx_flipped == idx
            self.assertTrue(comparison.all())
        except:
            self.assertTrue(False)

        try:
            idx_unflipped = np.array([0,0,0,1,1,1,2,2,2])
            idx_flipped = np.array([0,0,0,10,10,10,20,20,20])
            idx = preprocess.flip_clusters(idx_unflipped, dictionary={1:10, 2:20})
            comparison = idx_flipped == idx
            self.assertTrue(comparison.all())
        except:
            self.assertTrue(False)

    def test_flip_clusters_not_allowed_calls(self):

        idx_unflipped = np.array([0,0,0,1,1,1,2,2,2])
        with self.assertRaises(ValueError):
            idx = preprocess.flip_clusters(idx_unflipped, dictionary={3:2,2:3})

        with self.assertRaises(ValueError):
            idx = preprocess.flip_clusters(idx_unflipped, dictionary={0:1,1:1.5})

    def test_get_centroids_allowed_calls(self):

        try:
            x = np.array([[1,2,10],[1,2,10],[1,2,10]])
            idx = np.array([0,0,0])
            idx_centroids = np.array([[1, 2, 10]])
            centroids = preprocess.get_centroids(x, idx)
            comparison = (idx_centroids == centroids)
            self.assertTrue(comparison.all())
        except Exception:
            self.assertTrue(False)

        try:
            x = np.array([[1,2,10],[1,2,10],[20,30,40]])
            idx = np.array([0,0,1])
            idx_centroids = np.array([[1, 2, 10], [20,30,40]])
            centroids = preprocess.get_centroids(x, idx)
            comparison = (idx_centroids == centroids)
            self.assertTrue(comparison.all())
        except Exception:
            self.assertTrue(False)

    def test_get_centroidss_not_allowed_calls(self):

        X = np.random.rand(100,10)
        idx = np.zeros((90,))

        with self.assertRaises(ValueError):
            centroids = preprocess.get_centroids(X, idx)

        X = np.random.rand(100,10)
        idx = np.zeros((110,))

        with self.assertRaises(ValueError):
            centroids = preprocess.get_centroids(X, idx)

    def test_get_partition_allowed_calls(self):

        try:
            x = np.array([[1,2,10],[1,2,10],[1,2,10]])
            idx = np.array([0,0,0])
            pre_x_in_clusters = [np.array([[1,2,10],[1,2,10],[1,2,10]])]
            pre_idx_in_clusters = [np.array([0,1,2])]
            (x_in_clusters, idx_in_clusters) = preprocess.get_partition(x, idx)
            comparison_1 = (pre_x_in_clusters[0] == x_in_clusters[0])
            self.assertTrue(comparison_1.all())
            comparison_2 = (pre_idx_in_clusters[0] == idx_in_clusters[0])
            self.assertTrue(comparison_2.all())
        except Exception:
            self.assertTrue(False)

        try:
            x = np.array([[1,2,10],[1,2,10],[30,40,50]])
            idx = np.array([0,0,1])
            pre_x_in_clusters = [np.array([[1,2,10],[1,2,10]]), np.array([[30,40,50]])]
            pre_idx_in_clusters = [np.array([0,1]), np.array([2])]
            (x_in_clusters, idx_in_clusters) = preprocess.get_partition(x, idx)
            comparison_1 = (pre_x_in_clusters[0] == x_in_clusters[0])
            comparison_2 = (pre_x_in_clusters[1] == x_in_clusters[1])
            self.assertTrue(comparison_1.all())
            self.assertTrue(comparison_2.all())
            comparison_3 = (pre_idx_in_clusters[0] == idx_in_clusters[0])
            comparison_4 = (pre_idx_in_clusters[1] == idx_in_clusters[1])
            self.assertTrue(comparison_3.all())
            self.assertTrue(comparison_4.all())
        except Exception:
            self.assertTrue(False)

    def test_get_parition_not_allowed_calls(self):

        X = np.random.rand(100,10)
        idx = np.zeros((90,))

        with self.assertRaises(ValueError):
            (x_in_clusters, idx_in_clusters) = preprocess.get_partition(X, idx)

        X = np.random.rand(100,10)
        idx = np.zeros((110,))

        with self.assertRaises(ValueError):
            (x_in_clusters, idx_in_clusters) = preprocess.get_partition(X, idx)

    def test_get_populations_allowed_calls(self):

        x = np.linspace(-1,1,100)

        try:
            (idx, borders) = preprocess.variable_bins(x, 4, verbose=False)
            idx_populations = [25, 25, 25, 25]
            populations = preprocess.get_populations(idx)
            self.assertTrue(populations == idx_populations)
        except Exception:
            self.assertTrue(False)

        try:
            (idx, borders) = preprocess.variable_bins(x, 5, verbose=False)
            idx_populations = [20, 20, 20, 20, 20]
            populations = preprocess.get_populations(idx)
            self.assertTrue(populations == idx_populations)
        except Exception:
            self.assertTrue(False)

        try:
            (idx, borders) = preprocess.variable_bins(x, 2, verbose=False)
            idx_populations = [50, 50]
            populations = preprocess.get_populations(idx)
            self.assertTrue(populations == idx_populations)
        except Exception:
            self.assertTrue(False)

        try:
            (idx, borders) = preprocess.variable_bins(x, 1, verbose=False)
            idx_populations = [100]
            populations = preprocess.get_populations(idx)
            self.assertTrue(populations == idx_populations)
        except Exception:
            self.assertTrue(False)

        try:
            idx_populations = [1]
            populations = preprocess.get_populations(np.array([0]))
            self.assertTrue(populations == idx_populations)
        except Exception:
            self.assertTrue(False)
