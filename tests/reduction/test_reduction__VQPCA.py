import unittest
import numpy as np
from PCAfold import preprocess
from PCAfold import reduction
from PCAfold import analysis

class Reduction(unittest.TestCase):

    def test_reduction__VQPCA__allowed_calls(self):

        X = np.random.rand(400,10)

        try:
            vqpca = reduction.VQPCA(X, 3, 2, random_state=100)
            vqpca = reduction.VQPCA(X, 3, 2, idx_init='uniform', random_state=100)
            vqpca = reduction.VQPCA(X, 3, 2, idx_init='uniform', max_iter=20, random_state=100)
        except Exception:
            self.assertTrue(False)

        try:
            idx_init = np.ones((400,1))
            idx_init[20:40,:] = 2
            idx_init[200:300,:] = 3
            idx_init = idx_init.astype(int)
            vqpca = reduction.VQPCA(X, 3, 2, idx_init=idx_init)

            idx_init = np.ones((400,))
            idx_init[20:40] = 2
            idx_init[200:300] = 3
            idx_init = idx_init.astype(int)
            vqpca = reduction.VQPCA(X, 3, 2, idx_init=idx_init)
        except Exception:
            self.assertTrue(False)

# ------------------------------------------------------------------------------

    def test_reduction__VQPCA__allowed_attribute_call(self):

        X = np.random.rand(400,10)

        try:
            vqpca = reduction.VQPCA(X, 3, 2, random_state=100)
            vqpca.idx
            vqpca.collected_idx
            vqpca.converged
            vqpca.A
            vqpca.principal_components
            vqpca.reconstruction_errors_in_clusters
        except Exception:
            self.assertTrue(False)

# ------------------------------------------------------------------------------

    def test_reduction__VQPCA__not_allowed_attribute_setting(self):

        X = np.random.rand(400,10)
        vqpca = reduction.VQPCA(X, 3, 2, random_state=100)

        with self.assertRaises(AttributeError):
            vqpca.idx = 1
        with self.assertRaises(AttributeError):
            vqpca.collected_idx = 1
        with self.assertRaises(AttributeError):
            vqpca.converged = 1
        with self.assertRaises(AttributeError):
            vqpca.A = 1
        with self.assertRaises(AttributeError):
            vqpca.principal_components = 1
        with self.assertRaises(AttributeError):
            vqpca.reconstruction_errors_in_clusters = 1

# ------------------------------------------------------------------------------

    def test_reduction__VQPCA__not_allowed_calls(self):

        X = np.random.rand(400,10)

        with self.assertRaises(ValueError):
            vqpca = reduction.VQPCA(X, 0, 2)

        with self.assertRaises(ValueError):
            vqpca = reduction.VQPCA(X, 2, 0)

        with self.assertRaises(ValueError):
            vqpca = reduction.VQPCA(X, -1, 2)

        with self.assertRaises(ValueError):
            vqpca = reduction.VQPCA(X, 2, -1)

        with self.assertRaises(ValueError):
            vqpca = reduction.VQPCA(X, 2, 2, scaling='hello')

        with self.assertRaises(ValueError):
            vqpca = reduction.VQPCA(X, 2, 2, idx_init='hello')

        with self.assertRaises(ValueError):
            vqpca = reduction.VQPCA(X, 2, 2, max_iter=0)

        with self.assertRaises(ValueError):
            vqpca = reduction.VQPCA(X, 2, 2, max_iter=-1)

        with self.assertRaises(ValueError):
            vqpca = reduction.VQPCA(X, 2, 2, random_state='hello')

# ------------------------------------------------------------------------------

    def test_reduction__VQPCA__computation(self):

        X = np.random.rand(400,10)

        # Test that random seed is working properly:
        try:
            vqpca_1 = reduction.VQPCA(X, 3, 2, idx_init='random', random_state=100)
            idx_1 = vqpca_1.idx
            vqpca_2 = reduction.VQPCA(X, 3, 2, idx_init='random', random_state=100)
            idx_2 = vqpca_2.idx
            self.assertTrue(np.array_equal(idx_1, idx_2))
        except Exception:
            self.assertTrue(False)

        # Test that random seed is working properly:
        try:
            vqpca_1 = reduction.VQPCA(X, 3, 2, idx_init='random', random_state=100)
            idx_1 = vqpca_1.idx
            vqpca_2 = reduction.VQPCA(X, 3, 2, idx_init='random', random_state=200)
            idx_2 = vqpca_2.idx
            self.assertTrue(not np.array_equal(idx_1, idx_2))
        except Exception:
            self.assertTrue(False)

        # Test that uniform cluster initialization is working properly:
        try:
            vqpca_1 = reduction.VQPCA(X, 3, 2, idx_init='uniform')
            idx_1 = vqpca_1.idx
            vqpca_2 = reduction.VQPCA(X, 3, 2, idx_init='uniform')
            idx_2 = vqpca_2.idx
            self.assertTrue(np.array_equal(idx_1, idx_2))
        except Exception:
            self.assertTrue(False)

# ------------------------------------------------------------------------------

    def test_reduction__VQPCA__computation_on_line_dataset(self):

        # Generate a synthetic line dataset:
        n_observations = 1000
        x = np.linspace(0, 50, n_observations)
        phi_1 = np.zeros((n_observations,1))

        k1_count = 0
        k2_count = 0
        k3_count = 0

        for observation, x_value in enumerate(x):
            if x_value <= 10:
                phi_1[observation] = 2 * x_value
                k1_count += 1
            elif x_value > 10 and x_value <= 35:
                phi_1[observation] = - x_value + 30
                k2_count += 1
            elif x_value > 35:
                phi_1[observation] = 4 * x_value - 145
                k3_count += 1

        data_set_2d = np.hstack((x[:,None], phi_1))

        try:
            # Run VQPCA partitioning into three clusters:
            vqpca = reduction.VQPCA(data_set_2d,
                                    n_clusters=3,
                                    n_components=1,
                                    scaling='auto',
                                    idx_init='uniform',
                                    max_iter=100,
                                    verbose=False)
            idx_2d = vqpca.idx
            converged = vqpca.converged

            self.assertTrue(converged==True)

            # Check that clusters have been constructed as expected:
            self.assertTrue(idx_2d[0] != idx_2d[k1_count])
            self.assertTrue(idx_2d[k1_count] != idx_2d[k1_count+k2_count])

            for i in range(1,k1_count):
                self.assertTrue(idx_2d[0] == idx_2d[i])
            for i in range(k1_count+1,k1_count+k2_count):
                self.assertTrue(idx_2d[k1_count] == idx_2d[i])
            for i in range(k1_count+k2_count+1,k1_count+k2_count+k3_count):
                self.assertTrue(idx_2d[k1_count+k2_count] == idx_2d[i])
        except Exception:
            self.assertTrue(False)

        try:
            # Run VQPCA partitioning into three clusters:
            vqpca = reduction.VQPCA(data_set_2d,
                                    n_clusters=3,
                                    n_components=1,
                                    scaling='auto',
                                    idx_init='random',
                                    max_iter=100,
                                    verbose=False)
            idx_2d = vqpca.idx
            converged = vqpca.converged

            self.assertTrue(converged==True)

            # Check that clusters have been constructed as expected:
            self.assertTrue(idx_2d[0] != idx_2d[k1_count])
            self.assertTrue(idx_2d[k1_count] != idx_2d[k1_count+k2_count])

            for i in range(1,k1_count):
                self.assertTrue(idx_2d[0] == idx_2d[i])
            for i in range(k1_count+1,k1_count+k2_count):
                self.assertTrue(idx_2d[k1_count] == idx_2d[i])
            for i in range(k1_count+k2_count+1,k1_count+k2_count+k3_count):
                self.assertTrue(idx_2d[k1_count+k2_count] == idx_2d[i])
        except Exception:
            self.assertTrue(False)

        try:

            idx_init = np.zeros((n_observations,))
            idx_init[200:500] = 1
            idx_init[500:800] = 2
            idx_init = idx_init.astype(int)

            # Run VQPCA partitioning into three clusters:
            vqpca = reduction.VQPCA(data_set_2d,
                                    n_clusters=3,
                                    n_components=1,
                                    scaling='auto',
                                    idx_init=idx_init,
                                    max_iter=100,
                                    verbose=False)
            idx_2d = vqpca.idx
            converged = vqpca.converged

            self.assertTrue(converged==True)

            # Check that clusters have been constructed as expected:
            self.assertTrue(idx_2d[0] != idx_2d[k1_count])
            self.assertTrue(idx_2d[k1_count] != idx_2d[k1_count+k2_count])

            for i in range(1,k1_count):
                self.assertTrue(idx_2d[0] == idx_2d[i])
            for i in range(k1_count+1,k1_count+k2_count):
                self.assertTrue(idx_2d[k1_count] == idx_2d[i])
            for i in range(k1_count+k2_count+1,k1_count+k2_count+k3_count):
                self.assertTrue(idx_2d[k1_count+k2_count] == idx_2d[i])
        except Exception:
            self.assertTrue(False)

# ------------------------------------------------------------------------------

    def test_reduction__VQPCA__computation_on_planar_dataset(self):

        n_observations = 200
        x_3d = np.tile(np.linspace(0,50,n_observations), n_observations)
        y_3d = np.zeros((n_observations,1))
        phi_3d_1 = np.zeros((n_observations*n_observations,1))

        k1_count = 0
        k2_count = 0
        k3_count = 0

        for i in range(1,n_observations):
            y_3d = np.vstack((y_3d, np.ones((n_observations,1))*i))
        y_3d = y_3d.ravel()
        for observation, x_value in enumerate(x_3d):
            y_value = y_3d[observation]
            if x_value <= 10:
                phi_3d_1[observation] = 2 * x_value + y_value
                k1_count += 1
            elif x_value > 10 and x_value <= 35:
                phi_3d_1[observation] = 10 * x_value + y_value - 80
                k2_count += 1
            elif x_value > 35:
                phi_3d_1[observation] = 5 * x_value + y_value + 95
                k3_count += 1

        data_set_3d = np.hstack((x_3d[:,None], y_3d[:,None], phi_3d_1))

        try:
            # Run VQPCA partitioning into three clusters:
            vqpca = reduction.VQPCA(data_set_3d,
                                    n_clusters=3,
                                    n_components=2,
                                    scaling='auto',
                                    idx_init='uniform',
                                    max_iter=100,
                                    verbose=False)
            idx_3d = vqpca.idx
            converged = vqpca.converged

            self.assertTrue(converged==True)

            # Check that clusters have been constructed as expected:




        except Exception:
            self.assertTrue(False)

# ------------------------------------------------------------------------------
