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
                u += (x_train[j, l] - x[i, l]) * (x_train[j, l] - x[i, l]) / (s[i, l]*s[i, l])
            kj = exp(-u)
            sum_k += kj
            for q in range(p):
                sum_ky[q] += kj * y_train[j, q]
        for q in range(p):
            y[i, q] = sum_ky[q] / sum_k
            sum_ky[q] = 0.

class KReg:
    """
    A class for building and evaluating Nadaraya-Watson kernel regression models using a Gaussian kernel.
    The regression estimator :math:`\\mathcal{K}(u; \\sigma)` evaluated at independent variables :math:`u` can be
    expressed using a set of :math:`n` observations of independent variables (:math:`x`) and dependent variables
    (:math:`y`) as follows

    .. math::

            \\mathcal{K}(u; \\sigma) = \\frac{\\sum_{i=1}^{n} \\mathcal{W}_i(u; \\sigma) y_i}{\\sum_{i=1}^{n} \\mathcal{W}_i(u; \\sigma)}

    where a Gaussian kernel of bandwidth :math:`\\sigma` is used as

    .. math::
        \\mathcal{W}_i(u; \\sigma) = \\exp \\left( \\frac{-|| x_i - u ||_2^2}{\\sigma^2} \\right)

    Both constant and variable bandwidths are supported. Kernels with anisotropic bandwidths are calculated as

    .. math::
        \\mathcal{W}_i(u; \\sigma) = \\exp \\left( -|| \\text{diag}(\\sigma)^{-1} (x_i - u) ||_2^2 \\right)

    where :math:`\\sigma` is a vector of bandwidths per independent variable.

    **Example:**

    .. code::

      from PCAfold import KReg
      import numpy as np

      indepvars = np.expand_dims(np.linspace(0,np.pi,11),axis=1)
      depvars = np.cos(indepvars)
      query = np.expand_dims(np.linspace(0,np.pi,21),axis=1)

      model = KReg(indepvars, depvars)
      predicted = model.predict(query, 'nearest_neighbors_isotropic', n_neighbors=1)

    :param indepvars:
        ``numpy.ndarray`` specifying the independent variable training data, :math:`x` in equations above. It should be of size ``(n_observations,n_independent_variables)``.
    :param depvars:
        ``numpy.ndarray`` specifying the dependent variable training data, :math:`y` in equations above. It should be of size ``(n_observations,n_dependent_variables)``.
    :param internal_dtype:
        (optional, default float) data type to enforce in training and evaluating
    :param supress_warning:
        (optional, default False) if True, turns off printed warnings
    """
    def __init__(self, indepvars, depvars, internal_dtype=float, supress_warning=False):
        assert indepvars.ndim == 2, "independent variable array must be 2D: n_observations x n_variables."
        assert depvars.ndim == 2, "dependent variable array must be 2D: n_observations x n_variables."
        assert indepvars.shape[0] == depvars.shape[0], "number of observations for independent and dependent variables must match."

        if not isinstance(indepvars[0][0], internal_dtype) or not isinstance(depvars[0][0], internal_dtype):
            if not supress_warning:
                print("WARNING: casting training data as",internal_dtype)

        self._indepvars = indepvars.astype(internal_dtype)
        self._depvars = depvars.astype(internal_dtype)
        self._internal_dtype = internal_dtype

    @property
    def indepvars(self):
        return self._indepvars

    @property
    def depvars(self):
        return self._depvars

    @property
    def internal_dtype(self):
        return self._internal_dtype

    def compute_constant_bandwidth(self, query_points, bandwidth):
        """
        Format a single bandwidth value into a 2D array matching the shape of ``query_points``

        :param query_points:
            array of independent variable points to query the model (n_points x n_independent_variables)
        :param bandwidth:
            single value for the bandwidth used in a Gaussian kernel

        :return:
            an array of bandwidth values matching the shape of ``query_points``
        """
        return bandwidth*np.ones_like(query_points, dtype=self._internal_dtype)

    def compute_bandwidth_isotropic(self, query_points, bandwidth):
        """
        Format a 1D array of bandwidth values for each point in ``query_points`` into a 2D array matching the shape of ``query_points``

        :param query_points:
            array of independent variable points to query the model (n_points x n_independent_variables)
        :param bandwidth:
            1D array of bandwidth values length n_points

        :return:
            an array of bandwidth values matching the shape of ``query_points`` (repeats the bandwidth array for each independent variable)
        """
        assert bandwidth.ravel().size == query_points.shape[0], "provided bandwidth array must be of length equal to the number of rows in query_points."
        return np.tile(bandwidth, query_points.shape[1]).reshape(query_points.T.shape).T.astype(self._internal_dtype)

    def compute_bandwidth_anisotropic(self, query_points, bandwidth):
        """
        Format a 1D array of bandwidth values for each independent variable into the 2D array matching the shape of ``query_points``

        :param query_points:
            array of independent variable points to query the model (n_points x n_independent_variables)
        :param bandwidth:
            1D array of bandwidth values length n_independent_variables

        :return:
            an array of bandwidth values matching the shape of ``query_points`` (repeats the bandwidth array for each point in ``query_points``)
        """
        assert bandwidth.ravel().size == query_points.shape[1], "provided bandwidth array must be of length equal to the number of independent variables."
        return np.tile(bandwidth,query_points.shape[0]).reshape(query_points.shape).astype(self._internal_dtype)

    def compute_nearest_neighbors_bandwidth_isotropic(self, query_points, n_neighbors):
        """
        Compute a variable bandwidth for each point in ``query_points`` based on the Euclidean distance to the ``n_neighbors`` nearest neighbor

        :param query_points:
            array of independent variable points to query the model (n_points x n_independent_variables)
        :param n_neighbors:
            integer value for the number of nearest neighbors to consider in computing a bandwidth (distance)

        :return:
            an array of bandwidth values matching the shape of ``query_points`` (varies for each point, constant across independent variables)
        """
        tree = cKDTree(self._indepvars)
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
        Compute a variable bandwidth for each point in ``query_points`` and each independent variable separately based
        on the distance to the ``n_neighbors`` nearest neighbor in each independent variable dimension

        :param query_points:
            array of independent variable points to query the model (n_points x n_independent_variables)
        :param n_neighbors:
            integer value for the number of nearest neighbors to consider in computing a bandwidth (distance)

        :return:
            an array of bandwidth values matching the shape of ``query_points`` (varies for each point and independent variable)
        """
        variable_bandwidth = np.zeros_like(query_points, dtype=self._internal_dtype)
        for i in range(query_points.shape[1]):
            tree = cKDTree(np.expand_dims(self._indepvars[:, i], axis=1))
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
        Calculate dependent variable predictions at ``query_points``.

        :param query_points:
            ``numpy.ndarray`` specifying the independent variable points to query the model. It should be of size ``(n_points,n_independent_variables)``.
        :param bandwidth:
            value(s) to use for the bandwidth in the Gaussian kernel. Supported formats include:

            - single value: constant bandwidth applied to each query point and independent variable dimension.

            - 2D array shape (n_points x n_independent_variables): an array of bandwidths for each independent variable dimension of each query point.

            - string "nearest_neighbors_isotropic": This option requires the argument ``n_neighbors`` to be specified for which a bandwidth will be calculated for each query point based on the Euclidean distance to the ``n_neighbors`` nearest ``indepvars`` point.

            - string "nearest_neighbors_anisotropic": This option requires the argument ``n_neighbors`` to be specified for which a bandwidth will be calculated for each query point based on the distance in each (separate) independent variable dimension to the ``n_neighbors`` nearest ``indepvars`` point.

        :return: dependent variable predictions for the ``query_points``
        """
        assert query_points.ndim == 2, "query_points array must be 2D: n_observations x n_variables."
        assert query_points.shape[1] == self._indepvars.shape[1], "Number of query_points independent variables inconsistent with model."

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

        depvar_points = np.zeros((query_points.shape[0], self._depvars.shape[1]), dtype=self._internal_dtype)
        kreg_evaluate(query_points.astype(self._internal_dtype), depvar_points, self._indepvars, self._depvars, bandwidth_array)
        return depvar_points
