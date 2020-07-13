# distutils: language = c++

import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport exp, abs
from scipy.spatial import cKDTree


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
def kreg_evaluate(np.ndarray[double, ndim=2] x_np,
                  np.ndarray[double, ndim=2] y_np,
                  np.ndarray[double, ndim=2] x_train_np,
                  np.ndarray[double, ndim=2] y_train_np,
                  np.ndarray[double, ndim=2] s_np):
    cdef int n = x_train_np.shape[0]   # number of basis (training) points
    cdef int d = x_train_np.shape[1]   # number of independent variable dimensions
    cdef int m = x_np.shape[0]         # number of evaluation (testing) points
    cdef int p = y_np.shape[1]         # number of quantities to evaluate
    cdef double [:, :] x_train = x_train_np
    cdef double [:, :] y_train = y_train_np
    cdef double [:, :] s = s_np
    cdef double [:, :] x = x_np
    cdef double [:, :] y = y_np
    cdef double [:] sum_ky = np.zeros(p)
    cdef double sum_k = 0.
    cdef double u = 0.
    cdef double kj = 0.
    for i in range(m):
        sum_k = 0
        for j in range(n):
            u = 0
            for l in range(d):
                u += (x_train_np[j, l] - x[i, l]) * (x_train_np[j, l] - x[i, l]) / (s[i, l]*s[i, l])
            kj = exp(-u)
            sum_k += kj
            for q in range(p):
                sum_ky[q] += kj * y_train[j, q]
        for q in range(p):
            y[i, q] = sum_ky[q] / sum_k
            sum_ky[q] = 0.

class KReg:
    """
    A class for building and evaluating Nadaraya-Watson kernel regression models using a Gaussian kernel:
      kernel(query_point_vector) = exp{ -( || IV_train_vector - query_point_vector ||_2 / bandwidth )^2 }

    Examples:
        model = KReg(IV_train, DV_train)
        model.predict(query_points, bandwidth)

    :param IV_train: independent variable training data (n_observations x n_independent_variables)
    :param DV_train: dependent variable training data (n_observations x n_dependent_variables)
    :param internal_dtype: (optional) default array type is float
    :param supress_warning: (optional) default is False, switch to True to turn off printed warnings
    """
    def __init__(self, IV_train, DV_train, internal_dtype=float, supress_warning=False):
        assert IV_train.ndim == 2, "independent variable array must be 2D: n_observations x n_variables."
        assert DV_train.ndim == 2, "dependent variable array must be 2D: n_observations x n_variables."
        assert IV_train.shape[0] == DV_train.shape[0], "number of observations for independent and dependent variables must match."

        if not isinstance(IV_train[0][0],internal_dtype) or not isinstance(DV_train[0][0],internal_dtype):
            if not supress_warning:
                print("WARNING: casting training data as",internal_dtype)

        self._IV_train = IV_train.astype(internal_dtype)
        self._DV_train = DV_train.astype(internal_dtype)
        self._internal_dtype = internal_dtype

    @property
    def IV_train(self):
        return self._IV_train

    @property
    def DV_train(self):
        return self._DV_train

    @property
    def internal_dtype(self):
        return self._internal_dtype

    def compute_constant_bandwidth(self, query_points, bandwidth):
        """
        Format a single bandwidth value into the 2D array (matching query_points shape) kreg_evaluate expects
        :param query_points: array of independent variable points to query the model (n_points x n_independent_variables)
        :param bandwidth: single value for the bandwidth used in a Gaussian kernel
        :return: an array of bandwidth values matching the shape of query_points
        """
        return bandwidth*np.ones_like(query_points, dtype=self._internal_dtype)

    def compute_bandwidth_isotropic(self, query_points, bandwidth):
        """
        Format a 1D array of bandwidth values for each point in query_points into the 2D array (matching query_points shape) kreg_evaluate expects
        :param query_points: array of independent variable points to query the model (n_points x n_independent_variables)
        :param bandwidth: 1D array of bandwidth values length n_points
        :return: an array of bandwidth values matching the shape of query_points (repeats the bandwidth array for each independent variable)
        """
        assert bandwidth.ravel().size == query_points.shape[0], "provided bandwidth array must be of length equal to the number of rows in query_points."
        return np.tile(bandwidth, query_points.shape[1]).reshape(query_points.T.shape).T.astype(self._internal_dtype)

    def compute_bandwidth_anisotropic(self, query_points, bandwidth):
        """
        Format a 1D array of bandwidth values for each independent variable into the 2D array (matching query_points shape) kreg_evaluate expects
        :param query_points: array of independent variable points to query the model (n_points x n_independent_variables)
        :param bandwidth: 1D array of bandwidth values length n_independent_variables
        :return: an array of bandwidth values matching the shape of query_points (repeats the bandwidth array for each point in query_points)
        """
        assert bandwidth.ravel().size == query_points.shape[1], "provided bandwidth array must be of length equal to the number of independent variables."
        return np.tile(bandwidth,query_points.shape[0]).reshape(query_points.shape).astype(self._internal_dtype)

    def compute_nearest_neighbors_bandwidth_isotropic(self, query_points, n_neighbors):
        """
        Compute a variable bandwidth for each point in query_points based on the Euclidean distance to the n_neighbors nearest neighbor
        :param query_points: array of independent variable points to query the model (n_points x n_independent_variables)
        :param n_neighbors: integer value for the number of nearest neighbors to consider in computing a bandwidth (distance)
        :return: an array of bandwidth values matching the shape of query_points (repeats the computed variable bandwidths for each independent variable)
        """
        tree = cKDTree(self._IV_train)
        query_bandwidth = tree.query(query_points,k=n_neighbors)[0]
        if n_neighbors==1:
            variable_bandwidth = query_bandwidth
        else:
            variable_bandwidth = query_bandwidth[:, n_neighbors - 1]

        threshold = 1.e-16
        variable_bandwidth[variable_bandwidth<threshold] = threshold # remove zero values
        return self.compute_bandwidth_isotropic(query_points, variable_bandwidth)

    def compute_nearest_neighbors_bandwidth_anisotropic(self, query_points, n_neighbors):
        """
        Compute a variable bandwidth for each point in query_points and each independent variable separately based on the distance to the n_neighbors nearest neighbor in each independent variable dimension
        :param query_points: array of independent variable points to query the model (n_points x n_independent_variables)
        :param n_neighbors: integer value for the number of nearest neighbors to consider in computing a bandwidth (distance)
        :return: an array of bandwidth values matching the shape of query_points (based on the distances to the n_neighbors nearest neighbor in each independent variable dimension and for each point in query_points)
        """
        variable_bandwidth = np.zeros_like(query_points, dtype=self._internal_dtype)
        for i in range(query_points.shape[1]):
            tree = cKDTree(np.expand_dims(self._IV_train[:,i],axis=1))
            query_bandwidth = tree.query(np.expand_dims(query_points[:,i],axis=1),k=n_neighbors)[0]
            if n_neighbors==1:
                variable_bandwidth[:,i] = query_bandwidth
            else:
                variable_bandwidth[:,i] = query_bandwidth[:, n_neighbors - 1]

        threshold = 1.e-16
        variable_bandwidth[variable_bandwidth<threshold] = threshold # remove zero values
        return variable_bandwidth

    def predict(self, query_points, bandwidth, n_neighbors=None):
        """
        Calculate dependent variable predictions at query_points.

        :param query_points: array of independent variable points to query the model (n_points x n_independent_variables)
        :param bandwidth: value(s) to use for the bandwidth in the Gaussian kernel. Supported formats include:
                            - single value: constant bandwidth applied to each query point and independent variable dimension.
                            - 2D array shape (n_points x n_independent_variables): an array of bandwidths for each independent variable dimension of each query point.
                            - string "nearest_neighbors_isotropic": This option requires the argument n_neighbors to be specified for which a bandwidth will be calculated
                                                                    for each query point based on the Euclidean distance to the n_neighbors nearest IV_train point.
                            - string "nearest_neighbors_anisotropic": This option requires the argument n_neighbors to be specified for which a bandwidth will be calculated
                                                                     for each query point based on the distance in each (separate) independent variable dimension to the n_neighbors nearest IV_train point.

        :return: dependent variable predictions for the query_points
        """
        assert query_points.ndim == 2, "query_points array must be 2D: n_observations x n_variables."
        assert query_points.shape[1] == self._IV_train.shape[1], "Number of query_points independent variables inconsistent with model."

        if isinstance(bandwidth,np.ndarray):
            if bandwidth.ndim == 2:
                assert bandwidth.shape == query_points.shape, "Shape of two-dimensional bandwidth array must match the shape of query_points."
                bandwidth_array = bandwidth.astype(self._internal_dtype)
            else:
                raise ValueError("An array for bandwidth must be the same shape as query_points.")
        elif isinstance(bandwidth,int) or isinstance(bandwidth,float):
            bandwidth_array = self.compute_constant_bandwidth(query_points, bandwidth)
        elif bandwidth=="nearest_neighbors_isotropic":
            assert n_neighbors is not None, "nearest neighbors method requires n_neighbors be specified."
            bandwidth_array = self.compute_nearest_neighbors_bandwidth_isotropic(query_points, n_neighbors)
        elif bandwidth=="nearest_neighbors_anisotropic":
            assert n_neighbors is not None, "nearest neighbors method requires n_neighbors be specified."
            bandwidth_array = self.compute_nearest_neighbors_bandwidth_anisotropic(query_points, n_neighbors)
        else:
            raise ValueError("Unsupported bandwidth type.")

        DV_points = np.zeros((query_points.shape[0], self._DV_train.shape[1]),dtype=self._internal_dtype)
        kreg_evaluate(query_points.astype(self._internal_dtype), DV_points, self._IV_train, self._DV_train, bandwidth_array)
        return DV_points
