"""reduction.py: module for data reduction."""

__author__ = "Kamila Zdybal, Elizabeth Armstrong, Alessandro Parente and James C. Sutherland"
__copyright__ = "Copyright (c) 2020, Kamila Zdybal and Elizabeth Armstrong"
__credits__ = ["Department of Chemical Engineering, University of Utah, Salt Lake City, Utah, USA", "Universite Libre de Bruxelles, Aero-Thermo-Mechanics Laboratory, Brussels, Belgium"]
__license__ = "MIT"
__version__ = "1.0.0"
__maintainer__ = ["Kamila Zdybal", "Elizabeth Armstrong"]
__email__ = ["kamilazdybal@gmail.com", "Elizabeth.Armstrong@chemeng.utah.edu", "James.Sutherland@chemeng.utah.edu"]
__status__ = "Production"

import numpy as np
import copy as cp
import pandas as pd
from scipy import linalg as lg
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
from PCAfold import preprocess
from PCAfold import DataSampler
from PCAfold.styles import *
from PCAfold.preprocess import _scalings_list

################################################################################
#
# Principal Component Analysis (PCA)
#
################################################################################

class PCA:
    """
    This class enables performing Principal Component Analysis (PCA)
    of the original data set :math:`\mathbf{X}`.

    For more detailed information on PCA the user is referred to :cite:`Jolliffe2002`.

    Two options for performing PCA are implemented:

    +--------------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------+
    | | Eigendecomposition of the covariance matrix                                                    | | Singular Value Decomposition (SVD)                                             |
    | | Set ``use_eigendec=True`` (default)                                                            | | Set ``use_eigendec=False``                                                     |
    +==================================================================================================+==================================================================================+
    | | **Centering and scaling** (as per ``preprocess.center_scale`` function):                                                                                                          |
    | | If ``nocenter=False``: :math:`\mathbf{X_{cs}} = (\mathbf{X} - \mathbf{C}) \cdot \mathbf{D}^{-1}`                                                                                  |
    | | If ``nocenter=True``: :math:`\mathbf{X_{cs}} = \mathbf{X} \cdot \mathbf{D}^{-1}`                                                                                                  |
    +--------------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------+
    | Eigendecomposition of the covariance matrix :math:`\mathbf{S}`                                   | SVD: :math:`\mathbf{X_{cs}} = \mathbf{U} \mathbf{\Sigma} \mathbf{V}^{\mathbf{T}}`|
    +--------------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------+
    | | **Modes:**                                                                                     | | **Modes:**                                                                     |
    | | Eigenvectors :math:`\mathbf{A}`                                                                | | :math:`\mathbf{A} = \mathbf{V}`                                                |
    +--------------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------+
    | | **Amplitudes:**                                                                                | | **Amplitudes:**                                                                |
    | | Eigenvalues :math:`\mathbf{L}`                                                                 | | :math:`\mathbf{L} = diag(\mathbf{\Sigma})`                                     |
    +--------------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------+

    *Note:* For simplicity, we will from now on refer to :math:`\mathbf{A}` as the
    matrix of eigenvectors and to :math:`\mathbf{L}` as the vector of eigenvalues,
    irrespective of the method used to perform PCA.

    Covariance matrix is computed at the class initialization as:

    .. math::

        \mathbf{S} = \\frac{1}{N-1} \mathbf{X_{cs}}^{\mathbf{T}} \mathbf{X_{cs}}

    where :math:`N` is the number of observations in the data set :math:`\mathbf{X}`.

    Loadings matrix :math:`\mathbf{l}` is computed at the class initialization as well
    such that element :math:`\mathbf{l}_{ij}` is the corresponding scaled element
    of the eigenvectors matrix :math:`\mathbf{A}_{ij}`:

    .. math::

        \mathbf{l}_{ij} = \\frac{\mathbf{A}_{ij} \\sqrt{\mathbf{L}_j}}{\\sqrt{\mathbf{S}_{ii}}}

    where :math:`\mathbf{L}_j` is the :math:`j^{th}` eigenvalue and :math:`\mathbf{S}_{ii}`
    is the :math:`i^{th}` element on the diagonal of the covariance matrix :math:`\mathbf{S}`.

    **Example:**

    .. code:: python

        from PCAfold import PCA
        import numpy as np

        # Generate dummy data set:
        X = np.random.rand(100,20)

        # Instantiate PCA class object:
        pca_X = PCA(X, scaling='none', n_components=2, use_eigendec=True, nocenter=False)

    :param X:
        original data set :math:`\mathbf{X}`.
    :param scaling: (optional)
        string specifying the scaling methodology as per
        ``preprocess.center_scale`` function.
    :param n_components: (optional)
        number of retained Principal Components :math:`q`. If set to 0 all PCs are retained.
    :param use_eigendec: (optional)
        boolean specifying the method for obtaining eigenvalues and eigenvectors:

            * ``use_eigendec=True`` uses eigendecomposition of the covariance matrix (from ``numpy.linalg.eigh``)
            * ``use_eigendec=False`` uses Singular Value Decomposition (SVD) (from ``scipy.linalg.svd``)

    :raises ValueError:
        if the original data set :math:`\mathbf{X}` has more variables (columns)
        then observations (rows).

    :raises ValueError:
        if a constant column is detected in the original data set :math:`\mathbf{X}`.

    :raises ValueError:
        if ``scaling`` method is not a string or is not within the available scalings.

    :raises ValueError:
        if ``n_components`` is not an integer, is negative or larger than the number of variables in a data set.

    :raises ValueError:
        if ``use_eigendec`` is not a boolean.

    :raises ValueError:
        if ``nocenter`` is not a boolean.

    **Attributes:**

        - **n_components** - (can be re-set) number of retained Principal Components :math:`q`.
        - **n_components_init** - (read only) number of retained Principal Components :math:`q` with which ``PCA`` class object was initialized.
        - **scaling** - (read only) scaling criteria with which ``PCA`` class object was initialized.
        - **n_variables** - (read only) number of variables of the original data set on which ``PCA`` class object was initialized.
        - **X_cs** - (read only) centered and scaled data set :math:`\mathbf{X_{cs}}`.
        - **X_center** - (read only) vector of centers :math:`\mathbf{C}` applied on the original data set :math:`\mathbf{X}`.
        - **X_scale** - (read only) vector of scales :math:`\mathbf{D}` applied on the original data set :math:`\mathbf{X}`.
        - **S** - (read only) covariance matrix :math:`\mathbf{S}`.
        - **L** - (read only) vector of eigenvalues :math:`\mathbf{L}`.
        - **A** - (read only) matrix of eigenvectors :math:`\mathbf{A}` (vectors are stored in columns, rows correspond to weights).
        - **loadings** - (read only) loadings :math:`\mathbf{l}` (vectors are stored in columns, rows correspond to weights).
    """

    def __init__(self, X, scaling='std', n_components=0, use_eigendec=True, nocenter=False):

        # Check X:
        (n_observations, n_variables) = np.shape(X)
        if (n_observations < n_variables):
            raise ValueError('Variables should be in columns; observations in rows.\n'
                             'Also ensure that you have more than one observation\n')
        (_, idx_removed, _) = preprocess.remove_constant_vars(X)
        if len(idx_removed) != 0:
            raise ValueError('Constant variable detected. Must preprocess data for PCA.')

        # Check scaling:
        if not isinstance(scaling, str):
            raise ValueError("Parameter `scaling` has to be a string.")
        else:
            if scaling.lower() not in _scalings_list:
                raise ValueError("Unrecognized scaling method.")
            else:
                self.__scaling = scaling.upper()

        # Check n_components:
        if not isinstance(n_components, int) or isinstance(n_components, bool):
            raise ValueError("Parameter `n_components` has to be an integer.")
        else:
            if (n_components < 0) or (n_components > n_variables):
                raise ValueError("Parameter `n_components` cannot be negative or larger than number of variables in a data set.")
            else:
                if n_components > 0:
                    self.__n_components = n_components
                    self.__n_components_init = n_components
                else:
                    self.__n_components = n_variables
                    self.__n_components_init = n_variables

        # Check use_eigendec:
        if not isinstance(use_eigendec, bool):
            raise ValueError("Parameter `use_eigendec` has to be a boolean.")

        # Check nocenter:
        if not isinstance(nocenter, bool):
            raise ValueError("Parameter `nocenter` has to be a boolean.")

        # Center and scale the data set:
        self.__X_cs, self.__X_center, self.__X_scale = preprocess.center_scale(X, self.scaling, nocenter)

        # Compute covariance matrix:
        self.__S = np.dot(self.X_cs.transpose(), self.X_cs) / (n_observations-1)

        # Perform PCA with eigendecomposition of the covariance matrix:
        if use_eigendec:
            L, Q = np.linalg.eigh(self.S)
            L = L / np.sum(L)

        # Perform PCA with Singular Value Decomposition:
        else:
            U, s, vh = lg.svd(self.X_cs)
            Q = vh.transpose()
            L = s * s / np.sum(s * s)

        # Sort eigenvalues and eigenvectors in the descending order:
        isort = np.argsort(-np.diagonal(np.diag(L)))
        Lsort = L[isort]
        Qsort = Q[:, isort]
        self.__A = Qsort
        self.__L = Lsort

        # Set number of variables in a data set (equal to the number of eigenvalues):
        self.__n_variables = len(self.L)

        # Compute loadings:
        loadings_matrix = np.zeros((self.n_variables, self.n_components))

        for i in range(self.n_components):
            for j in range(self.n_variables):
                loadings_matrix[j, i] = (self.A[j, i] * np.sqrt(self.L[i])) / np.sqrt(self.S[j, j])

        self.__loadings = loadings_matrix

    @property
    def n_components(self):
        return self.__n_components

    @property
    def n_components_init(self):
        return self.__n_components_init

    @property
    def scaling(self):
        return self.__scaling

    @property
    def n_variables(self):
        return self.__n_variables

    @property
    def X_cs(self):
        return self.__X_cs

    @property
    def X_center(self):
        return self.__X_center

    @property
    def X_scale(self):
        return self.__X_scale

    @property
    def S(self):
        return self.__S

    @property
    def A(self):
        return self.__A

    @property
    def L(self):
        return self.__L

    @property
    def loadings(self):
        return self.__loadings

    @n_components.setter
    def n_components(self, new_n_components):
        if not isinstance(new_n_components, int) or isinstance(new_n_components, bool):
            raise ValueError("Parameter `n_components` has to be an integer.")
        else:
            if (new_n_components < 0) or (new_n_components > self.n_variables):
                raise ValueError("Parameter `n_components` cannot be negative or larger than number of variables in a data set.")
            else:
                if new_n_components > 0:
                    self.__n_components = new_n_components
                else:
                    self.__n_components = self.n_variables

    def transform(self, X, nocenter=False):
        """
        This function transforms any data set :math:`\mathbf{X}` to a new
        truncated basis :math:`\mathbf{A_q}` identified by PCA.
        It computes the :math:`q`-first Principal Components
        :math:`\mathbf{Z_q}` given the original data.

        If ``nocenter=False``:

        .. math::

            \mathbf{Z_q} = (\mathbf{X} - \mathbf{C}) \cdot \mathbf{D}^{-1} \cdot \mathbf{A_q}

        If ``nocenter=True``:

        .. math::

            \mathbf{Z_q} = \mathbf{X} \cdot \mathbf{D}^{-1} \cdot \mathbf{A_q}

        Here :math:`\mathbf{C}` and :math:`\mathbf{D}` are centers and scales
        computed during ``PCA`` class initialization
        and :math:`\mathbf{A_q}` is the matrix of :math:`q`-first eigenvectors
        extracted from :math:`\mathbf{A}`.

        .. warning::

            Set ``nocenter=True`` only if you know what you are doing.

            One example when ``nocenter`` should be set to ``True`` is
            when transforming chemical source terms :math:`\mathbf{S_X}` to Principal Components space
            (as per :cite:`Sutherland2009`)
            to obtain sources of Principal Components :math:`\mathbf{S_Z}`. In
            that case :math:`\mathbf{X} = \mathbf{S_X}` and the transformation
            should be performed *without* centering:

            .. math::

                \mathbf{S_{Z_q}} = \mathbf{S_X} \cdot \mathbf{D}^{-1} \cdot \mathbf{A_q}

        **Example:**

        .. code:: python

            from PCAfold import PCA
            import numpy as np

            # Generate dummy data set:
            X = np.random.rand(100,20)

            # Instantiate PCA class object:
            pca_X = PCA(X, scaling='none', n_components=2, use_eigendec=True, nocenter=False)

            # Calculate the Principal Components:
            principal_components = pca_X.transform(X)

        :param X:
            data set to transform. Note that it does not need to
            be the same data set that was used to construct the PCA object. It
            could for instance be a function of that data set. By default,
            this data set will be pre-processed with the centers and scales
            computed on the data set used when constructing the PCA object.
        :param nocenter: (optional)
            boolean specifying whether ``PCA.X_center`` centers should be applied to
            center the data set before transformation.
            If ``nocenter=True`` centers will not be applied on the
            data set.

        :raises ValueError:
            if ``nocenter`` is not a boolean.

        :raises ValueError:
            if the number of variables in a data set is inconsistent with number of eigenvectors.

        :return:
            - **principal_components** - the :math:`q`-first Principal Components :math:`\mathbf{Z_q}`.
        """

        if not isinstance(nocenter, bool):
            raise ValueError("Parameter `nocenter` has to be a boolean.")

        n_components = self.n_components

        (n_observations, n_variables) = np.shape(X)

        if n_variables != len(self.L):
            raise ValueError("Number of variables in a data set is inconsistent with number of eigenvectors.")

        A = self.A[:, 0:n_components]
        x = np.zeros_like(X, dtype=float)

        if nocenter:
            for i in range(0, n_variables):
                x[:, i] = X[:, i] / self.X_scale[i]
            principal_components = x.dot(A)
        else:
            for i in range(0, n_variables):
                x[:, i] = (X[:, i] - self.X_center[i]) / self.X_scale[i]
            principal_components = x.dot(A)

        return principal_components

    def reconstruct(self, principal_components, nocenter=False):
        """
        This function calculates rank-:math:`q` reconstruction of the
        data set from the :math:`q`-first  Principal Components
        :math:`\mathbf{Z_q}`.

        If ``nocenter=False``:

        .. math::

            \mathbf{X_{rec}} = \mathbf{Z_q} \mathbf{A_q}^{\mathbf{T}} \cdot \mathbf{D} + \mathbf{C}

        If ``nocenter=True``:

        .. math::

            \mathbf{X_{rec}} = \mathbf{Z_q} \mathbf{A_q}^{\mathbf{T}} \cdot \mathbf{D}

        Here :math:`\mathbf{C}` and :math:`\mathbf{D}` are centers and scales
        computed during ``PCA`` class initialization
        and :math:`\mathbf{A_q}` is the matrix of :math:`q`-first eigenvectors
        extracted from :math:`\mathbf{A}`.

        .. warning::

            Set ``nocenter=True`` only if you know what you are doing.

            One example when ``nocenter`` should be set to ``True`` is
            when reconstructing chemical source terms :math:`\mathbf{S_X}`
            (as per :cite:`Sutherland2009`)
            from the :math:`q`-first sources of Principal Components :math:`\mathbf{S_{Z_q}}`. In
            that case :math:`\mathbf{Z_q} = \mathbf{S_{Z_q}}` and the reconstruction
            should be performed *without* uncentering:

            .. math::

                \mathbf{S_{X, rec}} = \mathbf{S_{Z_q}} \mathbf{A_q}^{\mathbf{T}} \cdot \mathbf{D}

        **Example:**

        .. code:: python

            from PCAfold import PCA
            import numpy as np

            # Generate dummy data set:
            X = np.random.rand(100,20)

            # Instantiate PCA class object:
            pca_X = PCA(X, scaling='none', n_components=2, use_eigendec=True, nocenter=False)

            # Calculate the Principal Components:
            principal_components = pca_X.transform(X)

            # Calculate the reconstructed variables:
            X_rec = pca_X.reconstruct(principal_components)

        :param principal_components:
            matrix of :math:`q`-first Principal Components :math:`\mathbf{Z_q}`.
        :param nocenter: (optional)
            boolean specifying whether ``PCA.X_center`` centers should be applied to
            un-center the reconstructed data set.
            If ``nocenter=True`` centers will not be applied on the
            reconstructed data set.

        :raises ValueError:
            if ``nocenter`` is not a boolean.

        :raises ValueError:
            if the number of Principal Components supplied is larger than the number of eigenvectors computed by PCA.

        :return:
            - **X_rec** - rank-:math:`q` reconstruction of the original data set.
        """

        if not isinstance(nocenter, bool):
            raise ValueError("Parameter `nocenter` has to be a boolean.")

        (n_observations, n_components) = np.shape(principal_components)

        if n_components > self.n_variables:
            raise ValueError("Number of Principal Components supplied is larger than the number of eigenvectors computed by PCA.")

        # Select n_components first Principal Components:
        A = self.A[:, 0:n_components]

        # Calculate unscaled, uncentered approximation to the data:
        x = principal_components.dot(A.transpose())

        if nocenter:
            C_zeros = np.zeros_like(self.X_center)
            X_rec = preprocess.invert_center_scale(x, C_zeros, self.X_scale)
        else:
            X_rec = preprocess.invert_center_scale(x, self.X_center, self.X_scale)

        return(X_rec)

    def calculate_r2(self, X):
        """
        This function calculates coefficient of determination :math:`R^2` values
        for the rank-:math:`q` reconstruction :math:`\mathbf{X_{rec}}` of the original
        data set :math:`\mathbf{X}`:

        .. math::

            R^2 = 1 - \\frac{\\sum_{i=1}^N (\mathbf{X}_i - \mathbf{X_{rec}}_i)^2}{\\sum_{i=1}^N (\mathbf{X}_i - mean(\mathbf{X}_i))^2}

        where :math:`\mathbf{X}_i` is the :math:`i^{th}` column
        of :math:`\mathbf{X}`, :math:`\mathbf{X_{rec}}_i` is the :math:`i^{th}` column
        of :math:`\mathbf{X_{rec}}` and :math:`N` is the number of
        observations in :math:`\mathbf{X}`.

        If all of the eigenvalues are retained, :math:`R^2` will be equal to unity.

        **Example:**

        .. code:: python

            from PCAfold import PCA
            import numpy as np

            # Generate dummy data set:
            X = np.random.rand(100,20)

            # Instantiate PCA class object:
            pca_X = PCA(X, scaling='auto', n_components=10, use_eigendec=True, nocenter=False)

            # Calculate the R2 values:
            r2 = pca_X.calculate_r2(X)

        :param X:
            original data set :math:`\mathbf{X}`.

        :return:
            - **r2** - coefficient of determination values :math:`R^2` for the rank-:math:`q` reconstruction of the original data set.
        """

        self.data_consistency_check(X, errors_are_fatal=True)

        (n_observations, n_variables) = np.shape(X)

        assert (n_observations > n_variables), "Need more observations than variables."

        xapprox = self.reconstruct(self.transform(X))
        r2 = np.zeros(n_variables)

        for i in range(0, n_variables):
            r2[i] = 1 - np.sum((X[:, i] - xapprox[:, i]) * (X[:, i] - xapprox[:, i])) / np.sum(
                (X[:, i] - X[:, i].mean(axis=0)) * (X[:, i] - X[:, i].mean(axis=0)))

        return r2

    def data_consistency_check(self, X, errors_are_fatal=False):
        """
        This function checks if the supplied data matrix ``X`` is consistent
        with the current ``PCA`` class object.

        **Example:**

        .. code:: python

            from PCAfold import PCA
            import numpy as np

            # Generate dummy data set:
            X = np.random.rand(100,20)

            # Instantiate PCA class object:
            pca_X = PCA(X, scaling='auto', n_components=10)

            # This data set will be consistent:
            X_1 = np.random.rand(50,20)
            is_consistent = pca_X.data_consistency_check(X_1)

            # This data set will not be consistent but will not throw ValueError:
            X_2 = np.random.rand(100,10)
            is_consistent = pca_X.data_consistency_check(X_2)

            # This data set will not be consistent and will throw ValueError:
            X_3 = np.random.rand(100,10)
            is_consistent = pca_X.data_consistency_check(X_3, errors_are_fatal=True)

        :param X:
            data set to check.
        :param errors_are_fatal: (optional)
            boolean indicating if ValueError should be raised if an incompatibility
            is detected.

        :raises ValueError:
            if ``errors_are_fatal`` is not a boolean.

        :raises ValueError:
            when data set is not consistent with the ``PCA`` class object and ``errors_are_fatal=True`` flag has been set.

        :return:
            - **is_consistent** - boolean for whether or not supplied data matrix ``X``\
            is consistent with the ``PCA`` class object.
        """

        if not isinstance(errors_are_fatal, bool):
            raise ValueError("Parameter `errors_are_fatal` has to be a boolean.")

        (n_observations, n_variables) = np.shape(X)

        # Save the currently set n_components to re-set it back later:
        initial_n_components = self.n_components

        # Set n_components to the number of variables in a currently supplied data set:
        self.n_components = n_variables

        is_inconsistent = False

        try:
            X_rec = self.reconstruct(self.transform(X))
        except Exception:
            is_inconsistent = True

        # err = X - self.reconstruct(self.transform(X))
        #
        # isBad = (np.max(np.abs(err), axis=0) / np.max(np.abs(X), axis=0) > 1e-10).any() or (
        #     np.min(np.abs(err), axis=0) / np.min(np.abs(X), axis=0) > 1e-10).any()

        # Set n_components back to what it was at the start of this function:
        self.n_components = initial_n_components

        if is_inconsistent and errors_are_fatal:
            raise ValueError('It appears that the data set supplied is not consistent with the data used to construct the PCA object.')

        is_consistent = not is_inconsistent

        return is_consistent

    def r2_convergence(self, X, n_pcs, variable_names=[], print_width=10, verbose=False, save_filename=None):
        """
        This function returns and optionally prints and/or saves to ``.txt`` file
        :math:`R^2` values (as per ``PCA.calculate_r2``
        function) for reconstruction of the original data set :math:`\mathbf{X}`
        as a function of number of retained Principal Components (PCs).

        **Example:**

        .. code:: python

            from PCAfold import PCA
            import numpy as np

            # Generate dummy data set:
            X = np.random.rand(100,3)

            # Instantiate PCA class object:
            pca_X = PCA(X, scaling='auto', n_components=3)

            # Compute and print convergence of R2 values:
            r2 = pca_X.r2_convergence(X, n_pcs=3, variable_names=['X1', 'X2', 'X3'], print_width=10, verbose=True)

        The code above will print :math:`R^2` values retaining 1-3 PCs:

        .. code-block:: text

            | n PCs      | X1         | X2         | X3         | Mean       |
            | 1          | 0.17857365 | 0.53258736 | 0.49905763 | 0.40340621 |
            | 2          | 0.99220888 | 0.57167479 | 0.61150487 | 0.72512951 |
            | 3          | 1.0        | 1.0        | 1.0        | 1.0        |

        :param X:
            original dataset :math:`\mathbf{X}`.
        :param n_pcs:
            the maximum number of PCs to consider.
        :param variable_names: (optional)
            list of strings specifying variable names. If not specified variables will be numbered.
        :param print_width: (optional)
            width of columns printed out.
        :param verbose: (optional)
            boolean for printing out the table with :math:`R^2` values.
        :param save_filename: (optional)
            string specifying ``.txt`` save location/filename.

        :raises ValueError:
            if ``n_pcs`` is not a positive integer or is larger than the number of variables in a data set provided.
        :raises ValueError:
            if the number of variables in ``variable_names`` is not consistent with the number of variables in a data set provided.
        :raises ValueError:
            if ``save_filename`` is not a string.
        :raises ValueError:
            if ``verbose`` is not a boolean.

        :return:
            - **r2** - matrix of size ``(n_pcs, n_variables)`` containing the :math:`R^2` values\
            for each variable as a function of the number of retained PCs.
        """

        if not isinstance(n_pcs, int) or n_pcs < 1 or isinstance(n_pcs, bool):
            raise ValueError("Parameter `n_pcs` has to be a positive integer.")

        if not isinstance(verbose, bool):
            raise ValueError("Parameter `verbose` has to be a boolean.")

        if save_filename != None:
            if not isinstance(save_filename, str):
                raise ValueError("Parameter `save_filename` has to be a string.")

        n_observations, nvar = X.shape

        if n_pcs > nvar:
            raise ValueError("Parameter `n_pcs` cannot be larger than the number of variables in a data set.")

        # Save the currently set n_components to re-set it back later:
        initial_n_components = self.n_components

        r2 = np.zeros((n_pcs, nvar))
        r2vec = np.zeros((n_pcs, nvar + 1))
        self.data_consistency_check(X, errors_are_fatal=True)

        if len(variable_names) > 0:
            if len(variable_names) != nvar:
                raise ValueError("Number of variables in `variable_names` is not consistent with the number of variables in a data set.")
            rows_names = cp.deepcopy(variable_names)
        else:
            rows_names = []
            for i in range(nvar):
                rows_names.append(str(i + 1))

        neig = np.zeros((n_pcs), dtype=int)
        for i in range(n_pcs):
            self.n_components = i + 1
            neig[i] = self.n_components
            r2[i, :] = self.calculate_r2(X)

            r2vec[i, 0:-1] = np.round(r2[i, :], 8)
            r2vec[i, -1] = np.round(r2[i, :].mean(axis=0), 8)

        row_format = '|'
        for i in range(nvar + 2):
            row_format += ' {' + str(i) + ':<' + str(print_width) + '} |'
        rows_names.insert(0, 'n PCs')
        rows_names.append('Mean')

        if verbose:
            print(row_format.format(*rows_names))
            for i, row in zip(neig, r2vec):
                print(row_format.format(i, *row))

        if save_filename != None:

            fid = open(save_filename, 'w')
            fid.write("n PCs")

            for name in rows_names[1::]:
                fid.write(',%8s' % name)

            fid.write('\n')
            fid.close()

            for i in range(n_pcs):
                fid = open(save_filename, 'a')
                fid.write('%4i' % (i + 1))
                fid.close()

                with open(save_filename, 'ab') as fid:
                    np.savetxt(fid, np.array([r2vec[i, :]]), delimiter=' ', fmt=',%8.4f')
                fid.close()

        # Set n_components back to what it was at the start of this function:
        self.n_components = initial_n_components

        return r2

    def principal_variables(self, method='B2', x=[]):
        """
        This function extracts Principal Variables (PVs) from a PCA.

        The following methods are currently supported:

        * ``'B4'`` - selects Principal Variables based on the variables\
        contained in the eigenvectors corresponding to the largest\
        eigenvalues :cite:`Jolliffe1972`.

        * ``'B2'`` - selects Principal Variables based on the variables contained in the\
        smallest eigenvalues. These are discarded and the remaining\
        variables are used as the PVs :cite:`Jolliffe1972`.

        * ``'M2'`` - at each iteration, each remaining variable is analyzed\
        via PCA :cite:`Krzanowski1987`. *Note:* this is a very expensive method.

        For more detailed information on the options implemented here the user
        is referred to :cite:`Jolliffe2002`.

        **Example:**

        .. code:: python

            from PCAfold import PCA
            import numpy as np

            # Generate dummy data set:
            X = np.random.rand(100,20)

            # Instantiate PCA class object:
            pca_X = PCA(X, scaling='auto')

            # Select Principal Variables (PVs) using M2 method:
            principal_variables_indices = pca_X.principal_variables(method='M2', X)

        :param method: (optional)
            string specifying the method for determining the Principal Variables (PVs).

        :param x: (optional)
            data set to accompany ``'M2'`` method. Note that this is *only* required for the ``'M2'`` method.

        :raises ValueError:
            if the method selected is not ``'B4'``, ``'B2'`` or ``'M2'``.

        :raises ValueError:
            if the data set ``x`` is not supplied when ``method='M2'``.

        :return:
            - **principal_variables_indices** - a vector of indices of retained Principal Variables (PVs).
        """

        method = method.upper()

        if method == 'B2':  # B2 Method of Jolliffe (1972)
            nvar = self.n_variables
            neta = self.n_components
            eigVec = self.A  # eigenvectors

            # set indices for discarded variables by looking at eigenvectors
            # corresponding to the discarded eigenvalues

            idisc = -1 * np.ones(nvar - neta)

            for i in range(nvar - neta):
                j = nvar - i - 1
                ev = eigVec[:, j]
                isrt = np.argsort(-np.abs(ev))  # descending order
                evsrt = ev[isrt]

                # find the largest weight in this eigenvector
                # that has not yet been identified.
                for j in range(nvar):
                    ivar = isrt[j]
                    if np.all(idisc != ivar):
                        idisc[i] = ivar
                        break
            sd = np.setdiff1d(np.arange(nvar), idisc)
            principal_variables_indices = sd[np.argsort(sd)]

        elif method == 'B4':  # B4 Forward method
            nvar = self.n_variables
            neta = self.n_components
            eigVec = self.A  # eigenvectors

            # set indices for retained variables by looking at eigenvectors
            # corresponding to the retained eigenvalues
            principal_variables_indices = -1 * np.ones(neta)

            for i in range(neta):
                isrt = np.argsort(-np.abs(eigVec[:, i]))  # descending order

                # find the largest weight in this eigenvector
                # that has not yet been identified.
                for j in range(nvar):
                    ivar = isrt[j]
                    if np.all(principal_variables_indices != ivar):
                        principal_variables_indices[i] = ivar
                        break
            principal_variables_indices = principal_variables_indices[np.argsort(principal_variables_indices)]

        elif method == 'M2':  # Note: this is EXPENSIVE
            if len(x) == 0:
                raise ValueError('You must supply the data vector x when using the M2 method.')

            eta = self.transform(x)  # the PCs based on the full set of x.

            nvarTot = self.n_variables
            neta = self.n_components

            idiscard = []
            q = nvarTot
            while q > neta:

                n_observations, nvar = x.shape
                m2cut = 1e12

                for i in range(nvar):

                    # look at a PCA obtained from a subset of x.
                    xs = np.hstack((x[:, np.arange(i)], x[:, np.arange(i + 1, nvar)]))
                    pca2 = PCA(xs, self.scaling, neta)
                    etaSub = pca2.transform(xs)

                    cov = (etaSub.transpose()).dot(eta)  # covariance of the two sets of PCs

                    U, S, V = lg.svd(cov)  # svd of the covariance
                    m2 = np.trace((eta.transpose()).dot(eta) + (etaSub.transpose()).dot(etaSub) - 2 * S)

                    if m2 < m2cut:
                        m2cut = m2
                        idisc = i

                # discard the selected variable
                x = np.hstack((x[:, np.arange(idisc)], x[:, np.arange(idisc + 1, nvar)]))

                # determine the original index for this discarded variable.
                ii = np.setdiff1d(np.arange(nvarTot), idiscard)
                idisc = ii[idisc]
                idiscard.append(idisc)
                print('Discarding variable: %i\n' % (idisc + 1))

                q -= 1

            sd = np.setdiff1d(np.arange(nvarTot), idiscard)
            principal_variables_indices = sd[np.argsort(sd)]

        else:
            raise ValueError('Invalid method ' + method + ' for identifying principle variables')

        principal_variables_indices = principal_variables_indices.astype(int)

        return principal_variables_indices

    def save_to_txt(self, save_filename):
        """
        This function writes the eigenvector matrix :math:`\mathbf{A}`,
        loadings :math:`\mathbf{l}`, centering :math:`\mathbf{C}`
        and scaling :math:`\mathbf{D}` vectors to ``.txt`` file.

        **Example:**

        .. code:: python

            from PCAfold import PCA
            import numpy as np

            # Generate dummy data set:
            X = np.random.rand(100,5)

            # Instantiate PCA class object:
            pca_X = PCA(X, scaling='auto', n_components=5)

            # Save the PCA results to .txt:
            pca_X.save_to_txt('pca_X_Data.txt')

        :param save_filename:
            string specifying ``.txt`` save location/filename.
        """

        fid = open(save_filename, 'w')
        fid.write('%s\n' % "Eigenvectors:")
        fid.close()

        with open(save_filename, 'ab') as fid:
            np.savetxt(fid, self.A, delimiter=',', fmt='%6.12f')
        fid.close()

        fid = open(save_filename, 'a')
        fid.write('\n%s\n' % "Loadings:")
        fid.close()

        with open(save_filename, 'ab') as fid:
            np.savetxt(fid, self.loadings, delimiter=',', fmt='%6.12f')
        fid.close()

        fid = open(save_filename, 'a')
        fid.write('\n%s\n' % "Centering Factors:")
        fid.close()

        with open(save_filename, 'ab') as fid:
            np.savetxt(fid, np.array([self.X_center]), delimiter=',', fmt='%6.12f')
        fid.close()

        fid = open(save_filename, 'a')
        fid.write('\n%s\n' % "Scaling Factors:")
        fid.close()

        with open(save_filename, 'ab') as fid:
            np.savetxt(fid, np.array([self.X_scale]), delimiter=',', fmt='%6.12f')
        fid.close()

    def set_retained_eigenvalues(self, method='SCREE GRAPH', option=None):
        """
        This function helps determine how many Principal Components (PCs) should be retained.
        The following methods are available:

        * ``'TOTAL VARIANCE'`` - retain the PCs whose eigenvalues account for a\
        specific percentage of the total variance. The required\
        number of PCs is then the smallest value of :math:`q` for which this chosen\
        percentage is exceeded. Fraction of variance can be supplied using the\
        ``option`` parameter. For instance, set ``option=0.6`` if you want to\
        account for 60% variance. If variance is not supplied in the ``option``\
        paramter, the user will be asked for input during function execution.

        * ``'INDIVIDUAL VARIANCE'`` - retain the PCs whose eigenvalues are\
        greater than the average of the eigenvalues :cite:`Kaiser1960` or than 0.7\
        times the average of the eigenvalues :cite:`Jolliffe1972`. For a correlation\
        matrix this average equals 1. Fraction of variance can be supplied using the\
        ``option`` parameter. For instance, set ``option=0.6`` if you want to\
        account for 60% variance. If variance is not supplied in the ``option``\
        paramter, the user will be asked for input during function execution.

        * ``'BROKEN STICK'`` - retain the PCs according to the *Broken Stick Model* :cite:`Frontier1976`.

        * ``'SCREE GRAPH'`` - retain the PCs using the scree graph, a plot of the eigenvalues\
        agaist their indexes, and look for a natural break between the large\
        and small eigenvalues.

        For more detailed information on the options implemented here the user
        is referred to :cite:`Jolliffe2002`.

        **Example:**

        .. code:: python

            from PCAfold import PCA
            import numpy as np

            # Generate dummy data set:
            X = np.random.rand(100,20)

            # Instantiate PCA class object:
            pca_X = PCA(X, scaling='auto')

            # Compute a new ``PCA`` class object with the new number of retained components:
            pca_X_new = pca_X.set_retained_eigenvalues(method='TOTAL VARIANCE', option=0.6)

            # The new number of Principal Components that has been set:
            print(pca_X_new.n_components)

        This function provides a few methods to select the number of eigenvalues
        to be retained in the PCA reduction.

        :param method: (optional)
            string specifying the method to use in selecting retained eigenvalues.
        :param option: (optional)
            additional parameter used for the ``'TOTAL VARIANCE'`` and
            ``'INDIVIDUAL VARIANCE'`` methods. If not supplied, information
            will be obtained interactively.

        :raises ValueError:
            if the fraction of retained variance supplied by the ``option`` parameter
            is not a number between 0 and 1.

        :raises ValueError:
            if the method selected is not ``'TOTAL VARIANCE'``, ``'INDIVIDUAL VARIANCE'``,
            ``'BROKEN STICK'`` or ``'SCREE GRAPH'``.

        :return:
            - **pca** - the PCA object with the number of retained eigenvalues set on it.
        """
        pca = cp.copy(self)
        neig = len(pca.L)
        method = method.upper()

        if method == 'TOTAL VARIANCE':
            if option:
                frac = option
            else:
                frac = float(input('Select the fraction of variance to preserve: '))
            neta = 1
            if (frac > 1.) or (frac < 0.):
                raise ValueError('Fraction of variance must be between 0 and 1.')
            tot_var = np.sum(pca.L)
            neig = len(pca.L)
            fracVar = 0
            while (fracVar < frac) and (neta <= neig):
                fracVar += pca.L[neta - 1] / tot_var
                neta += 1
            pca.n_components = neta - 1
        elif method == 'INDIVIDUAL VARIANCE':
            if option:
                fac = option
            else:
                print('Choose threshold between 0 and 1\n(1->Kaiser, 0.7->Joliffe)\n')
                fac = float(input(''))
            if (fac > 1.) or (fac < 0.):
                raise ValueError('Fraction of variance must be between 0 and 1.')

            cutoff = fac * pca.L.mean(axis=0)
            neta = 1
            if np.any(pca.L > cutoff):
                neta = neig
            else:
                while (pca.L[neta - 1] > cutoff) and neta <= neig:
                    neta += 1
            pca.n_components = neta - 1

        elif method == 'BROKEN STICK':
            neta = 1
            stick_stop = 1
            while (stick_stop == 1) and (neta <= neig):
                broken_stick = 0
                for j in np.arange(neta, neig + 1):
                    broken_stick += 1 / j
                stick_stop = pca.L[neta - 1] > broken_stick
                neta += 1
            pca.n_components = neta - 1

        elif method == 'SCREE PLOT' or method == 'SCREE GRAPH':
            plt = plot_cumulative_variance(pca.L, n_components=0, title=None, save_filename=None)
            plt.show()
            pca.n_components = int(float(input('Select number of retained eigenvalues: ')))
            plt.close()

        else:
            raise ValueError('Unsupported method: ' + method)

        return pca

    def u_scores(self, X):
        """
        This function calculates the U-scores (Principal Components):

        .. math::

            \mathbf{U_{scores}} = \mathbf{X_{cs}} \mathbf{A_q}

        This function is equivalent to ``PCA.transform``.

        **Example:**

        .. code:: python

            from PCAfold import PCA
            import numpy as np

            # Generate dummy data set:
            X = np.random.rand(100,20)

            # Instantiate PCA class object:
            pca_X = PCA(X, scaling='auto', n_components=10, use_eigendec=True, nocenter=False)

            # Calculate the U-scores:
            u_scores = pca_X.u_scores(X)

        :param X:
            data set to transform. Note that it does not need to
            be the same data set that was used to construct the PCA object. It
            could for instance be a function of that data set. By default,
            this data set will be pre-processed with the centers and scales
            computed on the data set used when constructing the PCA object.

        :return:
            - **u_scores** - U-scores (Principal Components).
        """

        self.data_consistency_check(X, errors_are_fatal=True)

        u_scores = self.transform(X)

        return(u_scores)

    def w_scores(self, X):
        """
        This function calculates the W-scores which are the Principal Components
        scaled by the inverse square root of the corresponding eigenvalue:

        .. math::

            \mathbf{W_{scores}} = \\frac{\mathbf{Z_q}}{\\sqrt{\mathbf{L_q}}}

        where :math:`\mathbf{L_q}` are the :math:`q`-first eigenvalues extracted
        from :math:`\mathbf{L}`.
        The W-scores are still uncorrelated and have variances equal unity.

        **Example:**

        .. code:: python

            from PCAfold import PCA
            import numpy as np

            # Generate dummy data set:
            X = np.random.rand(100,20)

            # Instantiate PCA class object:
            pca_X = PCA(X, scaling='auto', n_components=10, use_eigendec=True, nocenter=False)

            # Calculate the W-scores:
            w_scores = pca_X.w_scores(X)

        :param X:
            data set to transform. Note that it does not need to
            be the same data set that was used to construct the PCA object. It
            could for instance be a function of that data set. By default,
            this data set will be pre-processed with the centers and scales
            computed on the data set used when constructing the PCA object.

        :return:
            - **w_scores** - W-scores (scaled Principal Components).
        """

        self.data_consistency_check(X, errors_are_fatal=True)

        eval = self.L[0:self.n_components]

        w_scores = self.transform(X).dot(np.diag(1 / np.sqrt(eval)))

        return(w_scores)

    def __eq__(a, b):
        """
        Compares two PCA objects for equality.

        :param a:
            first PCA object.
        :param b:
            second PCA object.

        :return:
            - **iseq** - boolean for ``(a == b)``.
        """
        iseq = False
        scalErr = np.abs(a.X_scale - b.X_scale) / np.max(np.abs(a.X_scale))
        centErr = np.abs(a.X_center - b.X_center) / np.max(np.abs(a.X_center))

        RErr = np.abs(a.S - b.S) / np.max(np.abs(a.S))
        LErr = np.abs(a.L - b.L) / np.max(np.abs(a.L))
        QErr = np.abs(a.A - b.A) / np.max(np.abs(a.A))

        tol = 10 * np.finfo(float).eps

        if a.X_scale.all() == b.X_scale.all() and a.n_components == b.n_components and np.all(scalErr < tol) and np.all(
                        centErr < tol) and np.all(RErr < tol) and np.all(QErr < tol) and np.all(LErr < tol):
            iseq = True

        return iseq

    def __ne__(a, b):
        """
        Tests two PCA objects for inequality.

        :param a:
            first PCA object.
        :param b:
            second PCA object.

        :return:
            - **result** - boolean for ``(a != b)``.
        """
        result = not (a == b)

        return result

################################################################################
#
# Principal Component Analysis on sampled data sets
#
################################################################################

def pca_on_sampled_data_set(X, idx_X_r, scaling, n_components, biasing_option, X_source=[]):
    """
    This function performs PCA on sampled data set :math:`\mathbf{X_r}` with one
    of four implemented options.

    Reach out to the
    `Biasing options <https://pcafold.readthedocs.io/en/latest/user/data-reduction.html#id12>`_
    section of the documentation for more information on the available options.

    **Example:**

    .. code::

        from PCAfold import pca_on_sampled_data_set, DataSampler
        import numpy as np

        # Generate dummy data set:
        X = np.random.rand(100,10)

        # Generate dummy sampling indices:
        idx = np.zeros((100,))
        idx[50:80] = 1
        selection = DataSampler(idx)
        (idx_X_r, _) = selection.number(20, test_selection_option=1)

        # Perform PCA on sampled data set:
        (eigenvalues, eigenvectors, pc_scores, pc_sources, C, D, C_r, D_r) = pca_on_sampled_data_set(X, idx_X_r, scaling='auto', n_components=2, biasing_option=1)

    :param X:
        original (full) data set :math:`\mathbf{X}`.
    :param idx_X_r:
        vector of indices that should be extracted from :math:`\mathbf{X}` to
        form :math:`\mathbf{X_r}`.
    :param scaling:
        data scaling criterion.
    :param n_components:
        number of first Principal Components that will be saved.
    :param biasing_option:
        integer specifying biasing option.
        Can only attain values 1, 2, 3 or 4.
    :param X_source: (optional)
        source terms corresponding to the state-space variables in :math:`\mathbf{X}`.

    :raises ValueError:
        if ``biasing_option`` is not 1, 2, 3 or 4.

    :return:
        - **eigenvalues** - collected eigenvalues from each iteration.
        - **eigenvectors** - collected eigenvectors from each iteration.\
        This is a 2D array of size ``(n_variables, n_components)``.
        - **pc_scores** - collected PC scores from each iteration.\
        This is a 2D array of size ``(n_observations, n_components)``.
        - **pc_sources** - collected PC sources from each iteration.\
        This is a 2D array of size ``(n_observations, n_components)``.
        - **C** - a vector of centers that were used to pre-process the\
        original full data set.
        - **D** - a vector of scales that were used to pre-process the\
        original full data set.
        - **C_r** - a vector of centers that were used to pre-process the\
        sampled data set.
        - **D_r** - a vector of scales that were used to pre-process the\
        sampled data set.
    """

    # Check that `biasing_option` parameter was passed correctly:
    _biasing_options = [1,2,3,4]
    if biasing_option not in _biasing_options:
        raise ValueError("Option can only be 1-4.")

    (n_observations, n_variables) = np.shape(X)

    # Pre-process the original full data set:
    (X_cs, C, D) = preprocess.center_scale(X, scaling)

    # Pre-process the original full sources:
    if len(X_source) != 0:

        # Scale sources with the global scalings:
        X_source_cs = np.divide(X_source, D)

    if biasing_option == 1:

        # Generate the reduced data set X_r:
        X_r = X[idx_X_r,:]

        # Perform PCA on X_r:
        pca = PCA(X_r, scaling, n_components, use_eigendec=True)
        C_r = pca.X_center
        D_r = pca.X_scale

        # Compute eigenvectors:
        eigenvectors = pca.A

        # Compute eigenvalues:
        eigenvalues = pca.L

        # Compute PC-scores:
        pc_scores = X_cs.dot(eigenvectors[:,0:n_components])

        if len(X_source) != 0:

            # Compute PC-sources:
            pc_sources = X_source_cs.dot(eigenvectors[:,0:n_components])

    elif biasing_option == 2:

        # Generate the reduced data set X_r:
        X_r = X_cs[idx_X_r,:]

        # Perform PCA on X_r:
        pca = PCA(X_r, 'none', n_components, use_eigendec=True, nocenter=True)
        C_r = pca.X_center
        D_r = pca.X_scale

        # Compute eigenvectors:
        eigenvectors = pca.A

        # Compute eigenvalues:
        eigenvalues = pca.L

        # Compute local PC-scores:
        pc_scores = X_cs.dot(eigenvectors[:,0:n_components])

        if len(X_source) != 0:

            # Compute PC-sources:
            pc_sources = X_source_cs.dot(eigenvectors[:,0:n_components])

    elif biasing_option == 3:

        # Generate the reduced data set X_r:
        X_r = X[idx_X_r,:]

        # Perform PCA on X_r:
        pca = PCA(X_r, scaling, n_components, use_eigendec=True)
        C_r = pca.X_center
        D_r = pca.X_scale

        # Compute eigenvectors:
        eigenvectors = pca.A

        # Compute eigenvalues:
        eigenvalues = pca.L

        # Compute local PC-scores (the original data set will be centered and scaled with C_r and D_r):
        pc_scores = pca.transform(X, nocenter=False)

        if len(X_source) != 0:

            # Compute PC-sources:
            pc_sources = pca.transform(X_source, nocenter=True)

    elif biasing_option == 4:

        # Generate the reduced data set X_r:
        X_r = X[idx_X_r,:]

        # Compute the current centers and scales of X_r:
        (_, C_r, D_r) = preprocess.center_scale(X_r, scaling)

        # Pre-process the global data set with the current C_r and D_r:
        X_cs = (X - C_r) / D_r

        # Perform PCA on the original data set X:
        pca = PCA(X_cs, 'none', n_components, use_eigendec=True, nocenter=True)

        # Compute eigenvectors:
        eigenvectors = pca.A

        # Compute eigenvalues:
        eigenvalues = pca.L

        # Compute local PC-scores:
        pc_scores = X_cs.dot(eigenvectors[:,0:n_components])

        if len(X_source) != 0:

            # Compute PC-sources:
            pc_sources = X_source_cs.dot(eigenvectors[:,0:n_components])

    if len(X_source) == 0:

        pc_sources = []

    return(eigenvalues, eigenvectors, pc_scores, pc_sources, C, D, C_r, D_r)

def analyze_centers_change(X, idx_X_r, variable_names=[], plot_variables=[], legend_label=[], title=None, save_filename=None):
    """
    This function analyzes the change in normalized centers computed on the
    sampled subset of the original data set :math:`\mathbf{X_r}` with respect
    to the full original data set :math:`\mathbf{X}`.

    The original data set :math:`\mathbf{X}` is first normalized so that each
    variable ranges from 0 to 1:

    .. math::

        ||\mathbf{X}|| = \\frac{\mathbf{X} - min(\mathbf{X})}{max(\mathbf{X} - min(\mathbf{X}))}

    This normalization is done so that centers can be compared across variables
    on one plot.
    Samples are then extracted from :math:`||\mathbf{X}||`, according to
    ``idx_X_r``, to form :math:`||\mathbf{X_r}||`.

    Normalized centers are computed as:

    .. math::

        ||\mathbf{C}|| = mean(||\mathbf{X}||)

    .. math::

        ||\mathbf{C_r}|| = mean(||\mathbf{X_r}||)

    Percentage measuring the relative change in normalized centers is
    computed as:

    .. math::

        p = \\frac{||\mathbf{C_r}|| - ||\mathbf{C}||}{||\mathbf{C}||} \cdot 100\%

    **Example:**

    .. code:: python

        from PCAfold import analyze_centers_change, DataSampler
        import numpy as np

        # Generate dummy data set:
        X = np.random.rand(100,10)

        # Generate dummy sampling indices:
        idx = np.zeros((100,))
        idx[50:80] = 1
        selection = DataSampler(idx)
        (idx_X_r, _) = selection.number(20, test_selection_option=1)

        # Analyze the change in normalized centers:
        (normalized_C, normalized_C_r, center_movement_percentage, plt) = analyze_centers_change(X, idx_X_r)

    :param X:
        original data set :math:`\mathbf{X}`.
    :param idx_X_r:
        vector of indices that should be extracted from :math:`\mathbf{X}` to
        form :math:`\mathbf{X_r}`.
    :param variable_names: (optional)
        list of strings specifying variable names.
    :param plot_variables: (optional)
        list of integers specifying indices of variables to be plotted.
        By default, all variables are plotted.
    :param legend_label: (optional)
        list of strings specifying labels for the legend. First entry will refer
        to :math:`||\mathbf{C}||` and second entry to :math:`||\mathbf{C_r}||`.
        If the list is empty, legend will not be plotted.
    :param title: (optional)
        string specifying plot title. If set to ``None``
        title will not be plotted.
    :param save_filename: (optional)
        string specifying plot save location/filename. If set to ``None``
        plot will not be saved.
        You can also set a desired file extension,
        for instance ``.pdf``. If the file extension is not specified, the default
        is ``.png``.

    :return:
        - **normalized_C** - normalized centers :math:`||\mathbf{C}||`.
        - **normalized_C_r** - normalized centers :math:`||\mathbf{C_r}||`.
        - **center_movement_percentage** - percentage :math:`p`\
        measuring the relative change in normalized centers.
        - **plt** - plot handle.
    """

    color_X = '#191b27'
    color_X_r = '#ff2f18'
    color_link = '#bbbbbb'

    (n_observations, n_variables) = np.shape(X)

    # Create default labels for variables:
    if len(variable_names) == 0:
        variable_names = ['$X_{' + str(i) + '}$' for i in range(0, n_variables)]

    if len(plot_variables) != 0:
        X = X[:,plot_variables]
        variable_names = [variable_names[i] for i in plot_variables]
        (_, n_variables) = np.shape(X)

    X_normalized = (X - np.min(X, axis=0))
    X_normalized = X_normalized / np.max(X_normalized, axis=0)

    # Extract X_r using the provided idx_X_r:
    X_r_normalized = X_normalized[idx_X_r,:]

    # Find centers:
    normalized_C = np.mean(X_normalized, axis=0)
    normalized_C_r = np.mean(X_r_normalized, axis=0)

    # Compute the relative percentage by how much the center has moved:
    center_movement_percentage = (normalized_C_r - normalized_C) / normalized_C * 100

    x_range = np.arange(1, n_variables+1)

    fig, ax = plt.subplots(figsize=(n_variables, 6))

    plt.scatter(x_range, normalized_C, c=color_X, marker='o', s=marker_size, edgecolor='none', alpha=1, zorder=2)
    plt.scatter(x_range, normalized_C_r, c=color_X_r, marker='>', s=marker_size, edgecolor='none', alpha=1, zorder=2)
    plt.xticks(x_range, variable_names, fontsize=font_axes, **csfont)
    plt.yticks(fontsize=font_axes, **csfont)
    plt.ylabel('Normalized center [-]', fontsize=font_labels, **csfont)
    plt.ylim(-0.05,1.05)
    plt.xlim(0, n_variables+1.5)
    plt.grid(alpha=0.3, zorder=0)

    for i in range(0, n_variables):

        dy = normalized_C_r[i] - normalized_C[i]
        plt.arrow(x_range[i], normalized_C[i], 0, dy, color=color_link, ls='-', lw=1, zorder=1)

    if title != None:
        plt.title(title, fontsize=font_title, **csfont)

    for i, value in enumerate(center_movement_percentage):
        plt.text(i+1.05, normalized_C_r[i]+0.01, str(int(value)) + ' %', fontsize=font_text, c=color_X_r, **csfont)

    ax.spines["top"].set_visible(True)
    ax.spines["bottom"].set_visible(True)
    ax.spines["right"].set_visible(True)
    ax.spines["left"].set_visible(True)

    if len(legend_label) != 0:
        lgnd = plt.legend(legend_label, fontsize=font_legend, markerscale=marker_scale_legend, loc="upper right")

    if save_filename != None:
        plt.savefig(save_filename, dpi = 500, bbox_inches='tight')

    return(normalized_C, normalized_C_r, center_movement_percentage, plt)

def analyze_eigenvector_weights_change(eigenvectors, variable_names=[], plot_variables=[], normalize=False, zero_norm=False, legend_label=[], title=None, save_filename=None):
    """
    This function analyzes the change of weights on an eigenvector obtained
    from a reduced data set as specified by the ``eigenvectors`` matrix.
    This matrix can contain many versions of eigenvectors, for instance coming
    from each iteration from the ``equilibrate_cluster_populations`` function.

    If the number of versions is larger than two, the weights are plot on a
    color scale that marks each version. If there is a consistent trend, the
    coloring should form a clear trajectory.

    In a special case, when there are only two versions within ``eigenvectors``
    matrix, it is understood that the first version corresponds to the original
    data set and the last version to the *equilibrated* data set.

    *Note:*
    This function plots absolute, (and optionally normalized) values of weights on each
    variable. Columns are normalized dividing by the maximum value. This is
    done in order to compare the movement of weights equally, with the highest,
    normalized one being equal to 1. You can additionally set the
    ``zero_norm=True`` in order to normalize weights such that they are between
    0 and 1 (this is not done by default).

    **Example:**

    .. code:: python

        from PCAfold import equilibrate_cluster_populations, analyze_eigenvector_weights_change
        import numpy as np

        # Generate dummy data set:
        X = np.random.rand(100,10)

        # Generate dummy sampling indices:
        idx = np.zeros((100,))
        idx[50:80] = 1

        # Run cluster equlibration:
        (eigenvalues, eigenvectors_matrix, pc_scores_matrix, pc_sources_matrix, idx_train, C_r, D_r) = equilibrate_cluster_populations(X, idx, 'auto', n_components=2, biasing_option=1, n_iterations=1, random_seed=100, verbose=True)

        # Analyze weights change on the first eigenvector:
        plt = analyze_eigenvector_weights_change(eigenvectors_matrix[:,0,:])

        # Analyze weights change on the second eigenvector:
        plt = analyze_eigenvector_weights_change(eigenvectors_matrix[:,1,:])

    :param eigenvectors:
        matrix of concatenated eigenvectors coming from different data sets or
        from different iterations. It should be size ``(n_variables, n_versions)``.
        This parameter can be directly extracted from ``eigenvectors_matrix``
        output from function ``equilibrate_cluster_populations``.
        For instance if the first and second eigenvector should be plotted:

        .. code:: python

            eigenvectors_1 = eigenvectors_matrix[:,0,:]
            eigenvectors_2 = eigenvectors_matrix[:,1,:]
    :param variable_names: (optional)
        list of strings specifying variable names.
    :param plot_variables: (optional)
        list of integers specifying indices of variables to be plotted.
        By default, all variables are plotted.
    :param normalize: (optional)
        boolean specifying whether weights should be normlized at all.
        If set to false, the absolute values are plotted.
    :param zero_norm: (optional)
        boolean specifying whether weights should be normalized between 0 and 1.
        By default they are not normalized to start at 0.
        Only has effect if ``normalize=True``.
    :param legend_label: (optional)
        list of strings specifying labels for the legend. If the list is empty,
        legend will not be plotted.
    :param title: (optional)
        boolean or string specifying plot title. If set to ``None``
        title will not be plotted.
    :param save_filename: (optional)
        plot save location/filename. If set to ``None`` plot will not be saved.
        You can also set a desired file extension,
        for instance ``.pdf``. If the file extension is not specified, the default
        is ``.png``.

    :raises ValueError:
        if the number of variables in ``variable_names`` list does not
        correspond to variables in the ``eigenvectors_matrix``.

    :return:
        - **plt** - plot handle.
    """

    (n_variables, n_versions) = np.shape(eigenvectors)

    # Create default labels for variables:
    if len(variable_names) == 0:
        variable_names = ['$X_{' + str(i) + '}$' for i in range(0, n_variables)]

    # Check that the number of columns in the eigenvector matrix is equal to the
    # number of elements in the `variable_names` vector:
    if len(variable_names) != n_variables:
        raise ValueError("The number of variables in the eigenvector matrix is not equal to the number of variable names.")

    if len(plot_variables) != 0:

        eigenvectors = eigenvectors[plot_variables,:]
        variable_names = [variable_names[i] for i in plot_variables]
        (n_variables, _) = np.shape(eigenvectors)

    # Normalize each column inside `eigenvector_weights`:
    if normalize == True:
        if zero_norm == True:
            eigenvectors = (np.abs(eigenvectors).T - np.min(np.abs(eigenvectors), 1)).T

        eigenvectors = np.divide(np.abs(eigenvectors).T, np.max(np.abs(eigenvectors), 1)).T
    else:
        eigenvectors = np.abs(eigenvectors)

    x_range = np.arange(0, n_variables)

    # When there are only two versions, plot a comparison of the original data
    # set X and an equilibrated data set X_r(e):
    if n_versions == 2:

        color_X = '#191b27'
        color_X_r = '#ff2f18'
        color_link = '#bbbbbb'

        fig, ax = plt.subplots(figsize=(n_variables, 6))

        plt.scatter(x_range, eigenvectors[:,0], c=color_X, marker='o', s=marker_size, edgecolor='none', alpha=1, zorder=2)
        plt.scatter(x_range, eigenvectors[:,-1], c=color_X_r, marker='>', s=marker_size, edgecolor='none', alpha=1, zorder=2)

        for i in range(0,n_variables):

            dy = eigenvectors[i,-1] - eigenvectors[i,0]
            plt.arrow(x_range[i], eigenvectors[i,0], 0, dy, color=color_link, ls='-', lw=1, zorder=1)

        plt.xticks(x_range, variable_names, fontsize=font_axes, **csfont)
        plt.yticks(fontsize=font_axes, **csfont)

        if normalize == True:
            plt.ylabel('Normalized weight [-]', fontsize=font_labels, **csfont)
        else:
            plt.ylabel('Absolute weight [-]', fontsize=font_labels, **csfont)

        plt.ylim(-0.05,1.05)
        plt.xlim(-1, n_variables)
        plt.grid(alpha=0.3, zorder=0)

        if title != None:
            plt.title(title, fontsize=font_title, **csfont)

        ax.spines["top"].set_visible(True)
        ax.spines["bottom"].set_visible(True)
        ax.spines["right"].set_visible(True)
        ax.spines["left"].set_visible(True)

        if len(legend_label) != 0:

            lgnd = plt.legend(legend_label, fontsize=font_legend, markerscale=marker_scale_legend, loc="upper right")

            lgnd.legendHandles[0]._sizes = [marker_size*1.5]
            lgnd.legendHandles[1]._sizes = [marker_size*1.5]
            plt.setp(lgnd.texts, **csfont)

    # When there are more than two versions plot the trends:
    else:

        color_range = np.arange(0, n_versions)

        # Plot the eigenvector weights movement:
        fig, ax = plt.subplots(figsize=(n_variables, 6))

        for idx, variable in enumerate(variable_names):
            scat = ax.scatter(np.repeat(idx, n_versions), eigenvectors[idx,:], c=color_range, cmap=plt.cm.Spectral)

        plt.xticks(x_range, variable_names, fontsize=font_axes, **csfont)
        plt.yticks(fontsize=font_axes, **csfont)

        if normalize == True:
            plt.ylabel('Normalized weight [-]', fontsize=font_labels, **csfont)
        else:
            plt.ylabel('Absolute weight [-]', fontsize=font_labels, **csfont)

        plt.ylim(-0.05,1.05)
        plt.xlim(-1, n_variables)
        plt.grid(alpha=0.3)

        if title != None:
            plt.title(title, fontsize=font_title, **csfont)

        cbar = plt.colorbar(scat, ticks=[0, round((n_versions-1)/2), n_versions-1])

    if save_filename != None:
        plt.savefig(save_filename, dpi = 500, bbox_inches='tight')

    return plt

def analyze_eigenvalue_distribution(X, idx_X_r, scaling, biasing_option, legend_label=[], title=None, save_filename=None):
    """
    This function analyzes the normalized eigenvalue distribution when PCA is
    performed on the original data set :math:`\mathbf{X}` and on the sampled
    data set :math:`\mathbf{X_r}`.

    Reach out to the
    `Biasing options <https://pcafold.readthedocs.io/en/latest/user/data-reduction.html#id12>`_
    section of the documentation for more information on the available options.

    **Example:**

    .. code:: python

        from PCAfold import analyze_eigenvalue_distribution, DataSampler
        import numpy as np

        # Generate dummy data set:
        X = np.random.rand(100,10)

        # Generate dummy sampling indices:
        idx = np.zeros((100,))
        idx[50:80] = 1
        selection = DataSampler(idx)
        (idx_X_r, _) = selection.number(20, test_selection_option=1)

        # Analyze the change in eigenvalue distribution:
        plt = analyze_eigenvalue_distribution(X, idx_X_r, 'auto', biasing_option=1)

    :param X:
        original (full) data set :math:`\mathbf{X}`.
    :param idx_X_r:
        vector of indices that should be extracted from :math:`\mathbf{X}` to
        form :math:`\mathbf{X_r}`.
    :param scaling:
        data scaling criterion.
    :param biasing_option:
        integer specifying biasing option.
        Can only attain values 1, 2, 3 or 4.
    :param legend_label: (optional)
        list of strings specifying labels for the legend. First entry will refer
        to :math:`\mathbf{X}` and second entry to :math:`\mathbf{X_r}`.
        If the list is empty, legend will not be plotted.
    :param title: (optional)
        boolean or string specifying plot title. If set to ``None``
        title will not be plotted.
    :param save_filename: (optional)
        plot save location/filename. If set to ``None`` plot will not be saved.
        You can also set a desired file extension,
        for instance ``.pdf``. If the file extension is not specified, the default
        is ``.png``.

    :return:
        - **plt** - plot handle.
    """

    color_X = '#191b27'
    color_X_r = '#ff2f18'
    color_link = '#bbbbbb'

    (n_observations, n_variables) = np.shape(X)
    n_components_range = np.arange(1, n_variables+1)
    n_components = 1

    # PCA on the original full data set X:
    pca_original = PCA(X, scaling, n_components, use_eigendec=True)

    # Compute eigenvalues:
    eigenvalues_original = pca_original.L

    # PCA on the sampled data set X_r:
    (eigenvalues_sampled, _, _, _, _, _, _, _) = pca_on_sampled_data_set(X, idx_X_r, scaling, n_components, biasing_option, X_source=[])

    # Normalize eigenvalues:
    eigenvalues_original = eigenvalues_original / np.max(eigenvalues_original)
    eigenvalues_sampled = eigenvalues_sampled / np.max(eigenvalues_sampled)

    fig, ax = plt.subplots(figsize=(n_variables, 6))

    # Plot the eigenvalue distribution from the full original data set X:
    original_distribution = plt.scatter(n_components_range, eigenvalues_original, c=color_X, marker='o', s=marker_size, edgecolor='none', alpha=1, zorder=2)

    # Plot the eigenvalue distribution from the sampled data set:
    sampled_distribution = plt.scatter(n_components_range, eigenvalues_sampled, c=color_X_r, marker='>', s=marker_size, edgecolor='none', alpha=1, zorder=2)

    if len(legend_label) != 0:
        lgnd = plt.legend(legend_label, fontsize=font_legend, markerscale=marker_scale_legend, loc="upper right")

    plt.plot(n_components_range, eigenvalues_original, '-', c=color_X, linewidth=line_width, alpha=1, zorder=1)
    plt.plot(n_components_range, eigenvalues_sampled, '-', c=color_X_r, linewidth=line_width, alpha=1, zorder=1)

    plt.xticks(n_components_range, fontsize=font_axes, **csfont)
    plt.xlabel('$q$ [-]', fontsize=font_labels, **csfont)
    plt.ylabel('Normalized eigenvalue [-]', fontsize=font_labels, **csfont)
    plt.ylim(-0.05,1.05)
    plt.xlim(0, n_variables+1)
    plt.grid(alpha=0.3, zorder=0)

    ax.spines["top"].set_visible(True)
    ax.spines["bottom"].set_visible(True)
    ax.spines["right"].set_visible(True)
    ax.spines["left"].set_visible(True)

    if title != False:
        plt.title(title, fontsize=font_title, **csfont)

    if save_filename != None:
        plt.savefig(save_filename, dpi = 500, bbox_inches='tight')

    return plt

def equilibrate_cluster_populations(X, idx, scaling, n_components, biasing_option, X_source=[], n_iterations=10, stop_iter=0, random_seed=None, verbose=False):
    """
    This function gradually (in ``n_iterations``) equilibrates cluster populations heading towards
    population of the smallest cluster, in each cluster.

    At each iteration it generates a reduced data set :math:`\mathbf{X_r}^{(i)}`
    made up from new populations, performs PCA on that data set to find the
    :math:`i^{th}` version of the eigenvectors. Depending on the option
    selected, it then does the projection of a data set (and optionally also
    its sources) onto the found eigenvectors.

    Reach out to the
    `Biasing options <https://pcafold.readthedocs.io/en/latest/user/data-reduction.html#id12>`_
    section of the documentation for more information on the available options.

    **Equilibration:**

    For the moment, there is only one way implemented for the equilibration.
    The smallest cluster is found and any larger :math:`j^{th}` cluster's
    observations are diminished at each iteration by:

    .. math::

        \\frac{N_j - N_s}{\\verb|n_iterations|}

    :math:`N_j` is the number of observations in that :math:`j^{th}` cluster and
    :math:`N_s` is the number of observations in the smallest
    cluster. This is further illustrated on synthetic data set below:

    .. image:: ../images/cluster-equilibration-scheme.png
        :width: 700
        :align: center

    Future implementation will include equilibration that slows down close to
    equilibrium.

    **Interpretation for the outputs:**

    This function returns 3D arrays ``eigenvectors``, ``pc_scores`` and
    ``pc_sources`` that have the following structure:

    .. image:: ../images/cbpca-equlibrate-outputs.png
        :width: 700
        :align: center

    **Example:**

    .. code::

        from PCAfold import equilibrate_cluster_populations
        import numpy as np

        # Generate dummy data set:
        X = np.random.rand(100,10)

        # Generate dummy sampling indices:
        idx = np.zeros((100,))
        idx[50:80] = 1

        # Run cluster equlibration:
        (eigenvalues, eigenvectors_matrix, pc_scores_matrix, pc_sources_matrix, idx_train, C_r, D_r) = equilibrate_cluster_populations(X, idx, 'auto', n_components=2, biasing_option=1, n_iterations=1, random_seed=100, verbose=True)

    :param X:
        original (full) data set :math:`\mathbf{X}`.
    :param idx:
        vector of cluster classifications.
        The first cluster has index 0.
    :param scaling:
        data scaling criterion.
    :param X_source:
        source terms corresponding to the state-space variables in ``X``.
    :param n_components:
        number of first Principal Components that will be saved.
    :param biasing_option:
        integer specifying biasing option.
        Can only attain values 1, 2, 3 or 4.
    :param n_iterations: (optional)
        number of iterations to loop over.
    :param stop_iter: (optional)
        index of iteration to stop.
    :param random_seed: (optional)
        integer specifying random seed for random sample selection.
    :param verbose: (optional)
        boolean for printing verbose details.

    :raises ValueError:
        if ``biasing_option`` is not 1, 2, 3 or 4.

    :raises ValueError:
        if ``random_seed`` is not an integer.

    :return:
        - **eigenvalues** - collected eigenvalues from each iteration.
        - **eigenvectors_matrix** - collected eigenvectors from each iteration.\
        This is a 3D array of size ``(n_variables, n_components, n_iterations+1)``.
        - **pc_scores_matrix** - collected PC scores from each iteration.\
        This is a 3D array of size ``(n_observations, n_components, n_iterations+1)``.
        - **pc_sources_matrix** - collected PC sources from each iteration.\
        This is a 3D array of size ``(n_observations, n_components, n_iterations+1)``.
        - **idx_train** - the final training indices from the equilibrated iteration.
        - **C_r** - a vector of final centers that were used to center\
        the data set at the last (equlibration) iteration.
        - **D_r** - a vector of final scales that were used to scale the\
        data set at the last (equlibration) iteration.
    """

    # Check that `biasing_option` parameter was passed correctly:
    _biasing_options = [1,2,3,4]
    if biasing_option not in _biasing_options:
        raise ValueError("Option can only be 1-4.")

    if random_seed != None:
        if not isinstance(random_seed, int):
            raise ValueError("Random seed has to be an integer.")

    (n_observations, n_variables) = np.shape(X)
    populations = preprocess.get_populations(idx)
    N_smallest_cluster = np.min(populations)
    k = len(populations)

    # Initialize matrices:
    eigenvectors_matrix = np.zeros((n_variables,n_components,n_iterations+1))
    pc_scores_matrix = np.zeros((n_observations,n_components,n_iterations+1))
    pc_sources_matrix = np.zeros((n_observations,n_components,n_iterations+1))
    eigenvalues = np.zeros((n_variables, 1))
    idx_train = []

    # Perform global PCA on the original data set X: ---------------------------
    pca_global = PCA(X, scaling, n_components, use_eigendec=True)

    # Get a centered and scaled data set:
    X_cs = pca_global.X_cs
    X_center = pca_global.X_center
    X_scale = pca_global.X_scale

    # Compute global eigenvectors:
    global_eigenvectors = pca_global.A

    # Compute global eigenvalues:
    global_eigenvalues = pca_global.L
    maximum_global_eigenvalue = np.max(global_eigenvalues)

    # Append the global eigenvectors:
    eigenvectors_matrix[:,:,0] = global_eigenvectors[:,0:n_components]

    # Append the global eigenvalues:
    eigenvalues = np.hstack((eigenvalues, np.reshape(global_eigenvalues, (n_variables, 1))/maximum_global_eigenvalue))

    # Compute global PC-scores:
    global_pc_scores = pca_global.transform(X, nocenter=False)

    # Append the global PC-scores:
    pc_scores_matrix[:,:,0] = global_pc_scores

    if len(X_source) != 0:

        # Scale sources with the global scalings:
        X_source_cs = np.divide(X_source, X_scale)

        # Compute global PC-sources:
        global_pc_sources = pca_global.transform(X_source, nocenter=True)

        # Append the global PC-sources:
        pc_sources_matrix[:,:,0] = global_pc_sources

    # Number of observations that should be taken from each cluster at each iteration:
    eat_ups = np.zeros((k,))
    for cluster in range(0,k):
        eat_ups[cluster] = (populations[cluster] - N_smallest_cluster)/n_iterations

    if verbose == True:
        print('Biasing is performed with option ' + str(biasing_option) + '.')

    # Perform PCA on the reduced data set X_r(i): ------------------------------
    for iter in range(0,n_iterations):

        if (stop_iter != 0) and (iter == stop_iter):
            break

        for cluster, population in enumerate(populations):
            if population != N_smallest_cluster:
                # Eat up the segment:
                populations[cluster] = population - int(eat_ups[cluster])
            else:
                populations[cluster] = N_smallest_cluster

        # Generate a dictionary for manual sampling:
        sampling_dictionary = {}

        for cluster in range(0,k):
            if iter == n_iterations-1:
                # At the last iteration reach equal number of samples:
                sampling_dictionary[cluster] = int(N_smallest_cluster)
            else:
                sampling_dictionary[cluster] = int(populations[cluster])

        if verbose == True:
            print("\nAt iteration " + str(iter+1) + " taking samples:")
            print(sampling_dictionary)

        # Sample manually according to current `sampling_dictionary`:
        sampling_object = DataSampler(idx, random_seed=random_seed)
        (idx_train, _) = sampling_object.manual(sampling_dictionary, sampling_type='number')

        # Biasing option 1 -----------------------------------------------------
        if biasing_option == 1:

            (local_eigenvalues, eigenvectors, pc_scores, pc_sources, _, _, C_r, D_r) = pca_on_sampled_data_set(X, idx_train, scaling, n_components, biasing_option, X_source=X_source)
            maximum_local_eigenvalue = np.max(local_eigenvalues)

            # Append the local PC-scores:
            pc_scores_matrix[:,:,iter+1] = pc_scores

            if len(X_source) != 0:

                # Append the global PC-sources:
                pc_sources_matrix[:,:,iter+1] = pc_sources

        # Biasing option 2 -----------------------------------------------------
        elif biasing_option == 2:

            (local_eigenvalues, eigenvectors, pc_scores, pc_sources, _, _, C_r, D_r) = pca_on_sampled_data_set(X, idx_train, scaling, n_components, biasing_option, X_source=X_source)
            maximum_local_eigenvalue = np.max(local_eigenvalues)

            # Append the local PC-scores:
            pc_scores_matrix[:,:,iter+1] = pc_scores

            if len(X_source) != 0:

                # Append the global PC-sources:
                pc_sources_matrix[:,:,iter+1] = pc_sources

        # Biasing option 3 -----------------------------------------------------
        elif biasing_option == 3:

            (local_eigenvalues, eigenvectors, pc_scores, pc_sources, _, _, C_r, D_r) = pca_on_sampled_data_set(X, idx_train, scaling, n_components, biasing_option, X_source=X_source)
            maximum_local_eigenvalue = np.max(local_eigenvalues)

            # Append the local PC-scores:
            pc_scores_matrix[:,:,iter+1] = pc_scores

            if len(X_source) != 0:

                # Append the global PC-sources:
                pc_sources_matrix[:,:,iter+1] = pc_sources

        # Biasing option 4 -----------------------------------------------------
        elif biasing_option == 4:

            (local_eigenvalues, eigenvectors, pc_scores, pc_sources, _, _, C_r, D_r) = pca_on_sampled_data_set(X, idx_train, scaling, n_components, biasing_option, X_source=X_source)
            maximum_local_eigenvalue = np.max(local_eigenvalues)

            # Append the local PC-scores:
            pc_scores_matrix[:,:,iter+1] = pc_scores

            if len(X_source) != 0:

                # Append the global PC-sources:
                pc_sources_matrix[:,:,iter+1] = pc_sources

        # Append the local eigenvectors:
        eigenvectors_matrix[:,:,iter+1] = eigenvectors[:,0:n_components]

        # Append the local eigenvalues:
        eigenvalues = np.hstack((eigenvalues, np.reshape(local_eigenvalues, (n_variables, 1))/maximum_local_eigenvalue))

    # Remove the first column of zeros:
    eigenvalues = eigenvalues[:,1::]

    return(eigenvalues, eigenvectors_matrix, pc_scores_matrix, pc_sources_matrix, idx_train, C_r, D_r)

################################################################################
#
# Plotting functions
#
################################################################################

def plot_2d_manifold(x, y, color_variable=[], x_label=None, y_label=None, colorbar_label=None, title=None, save_filename=None):
    """
    This function plots a 2-dimensional manifold given two vectors
    defining the manifold.

    :param x:
        variable on the :math:`x`-axis.
    :param y:
        variable on the :math:`y`-axis.
    :param color_variable: (optional)
        a 1D vector or string specifying color for the manifold. If it is a
        vector, it has to have length consistent with the number of observations
        in ``x`` and ``y`` vectors.
        It can also be set to a string specifying the color directly, for
        instance ``'r'`` or ``'#006778'``.
        If not specified, manifold will be plotted in black.
    :param x_label: (optional)
        string specifying :math:`x`-axis label annotation. If set to ``None``
        label will not be plotted.
    :param y_label: (optional)
        string specifying :math:`y`-axis label annotation. If set to ``None``
        label will not be plotted.
    :param colorbar_label: (optional)
        string specifying colorbar label annotation.
        If set to ``None``, colorbar label will not be plotted.
    :param title: (optional)
        string specifying plot title. If set to ``None`` title will not be
        plotted.
    :param save_filename: (optional)
        string specifying plot save location/filename. If set to ``None``
        plot will not be saved.
        You can also set a desired file extension,
        for instance ``.pdf``. If the file extension is not specified, the default
        is ``.png``.

    :return:
        - **plt** - plot handle.
    """

    fig, axs = plt.subplots(1, 1, figsize=(6,6))

    if len(color_variable) == 0:
        scat = plt.scatter(x.ravel(), y.ravel(), c='k', marker='o', s=scatter_point_size, edgecolor='none', alpha=1)
    else:
        scat = plt.scatter(x.ravel(), y.ravel(), c=color_variable, marker='o', s=scatter_point_size, edgecolor='none', alpha=1)

    plt.xticks(fontsize=font_axes, **csfont)
    plt.yticks(fontsize=font_axes, **csfont)
    if x_label != None: plt.xlabel(x_label, fontsize=font_labels, **csfont)
    if y_label != None: plt.ylabel(y_label, fontsize=font_labels, **csfont)
    plt.grid(alpha=0.3)

    if not isinstance(color_variable, str):
        if len(color_variable) != 0:
            cb = fig.colorbar(scat)
            cb.ax.tick_params(labelsize=font_colorbar_axes)
            if colorbar_label != None: cb.set_label(colorbar_label, fontsize=font_colorbar, rotation=0, horizontalalignment='left')

    if title != None: plt.title(title, fontsize=font_title, **csfont)
    if save_filename != None: plt.savefig(save_filename, dpi = 500, bbox_inches='tight')

    return plt

def plot_parity(variable, variable_rec, color_variable=[], x_label=None, y_label=None, colorbar_label=None, title=None, save_filename=None):
    """
    This function plots a parity plot between a variable and its reconstruction.

    :param variable:
        vector specifying the original variable.
    :param variable_rec:
        vector specifying the reconstruction of the original variable
    :param color_variable: (optional)
        a 1D vector or string specifying color for the manifold. If it is a
        vector, it has to have length consistent with the number of observations
        in ``variable`` vector.
        It can also be set to a string specifying the color directly, for
        instance ``'r'`` or ``'#006778'``.
        If not specified, manifold will be plotted in black.
    :param x_label: (optional)
        string specifying :math:`x`-axis label annotation. If set to ``None``
        label will not be plotted.
    :param y_label: (optional)
        string specifying :math:`y`-axis label annotation. If set to ``None``
        label will not be plotted.
    :param colorbar_label: (optional)
        string specifying colorbar label annotation.
        If set to ``None``, colorbar label will not be plotted.
    :param title: (optional)
        string specifying plot title. If set to ``None`` title will not be
        plotted.
    :param save_filename: (optional)
        string specifying plot save location/filename. If set to ``None``
        plot will not be saved.
        You can also set a desired file extension,
        for instance ``.pdf``. If the file extension is not specified, the default
        is ``.png``.

    :return:
        - **plt** - plot handle.
    """

    color_line = '#ff2f18'

    fig, axs = plt.subplots(1, 1, figsize=(6,6))

    if len(color_variable) == 0:
        scat = plt.scatter(variable, variable_rec, c='k', marker='o', s=scatter_point_size, edgecolor='none', alpha=1)
    else:
        scat = plt.scatter(variable, variable_rec, c=color_variable, marker='o', s=scatter_point_size, edgecolor='none', alpha=1)

    plt.plot(variable, variable, c=color_line)
    plt.axis('equal')
    plt.xticks(fontsize=font_axes, **csfont)
    plt.yticks(fontsize=font_axes, **csfont)
    if x_label != None: plt.xlabel(x_label, fontsize=font_labels, **csfont)
    if y_label != None: plt.ylabel(y_label, fontsize=font_labels, **csfont)
    plt.grid(alpha=0.3)

    if not isinstance(color_variable, str):
        if len(color_variable) != 0:
            cb = fig.colorbar(scat)
            cb.ax.tick_params(labelsize=font_colorbar_axes)
            if colorbar_label != None: cb.set_label(colorbar_label, fontsize=font_colorbar, rotation=0, horizontalalignment='left')

    if title != None: plt.title(title, fontsize=font_title, **csfont)
    if save_filename != None: plt.savefig(save_filename, dpi = 500, bbox_inches='tight')

    return plt

def plot_eigenvectors(eigenvectors, eigenvectors_indices=[], variable_names=[], plot_absolute=False, bar_color=None, title=None, save_filename=None):
    """
    This function plots weights on eigenvectors. It will generate as many
    plots as there are eigenvectors present in the ``eigenvectors`` matrix.

    :param eigenvectors:
        matrix of eigenvectors to plot. It can be supplied as an attribute of
        the ``PCA`` class: ``PCA.A``.
    :param eigenvectors_indices:
        list of integers specifying indexing of eigenvectors inside
        ``eigenvectors`` supplied. If it is not supplied, it is assumed that
        eigenvectors are numbered :math:`[0, 1, 2, \\dots, n]`, where :math:`n`
        is the number of eigenvectors provided.
    :param variable_names: (optional)
        list of strings specifying variable names.
    :param plot_absolute:
        boolean specifying whether absolute values of eigenvectors should be plotted.
    :param bar_color: (optional)
        string specifying color of bars.
    :param title: (optional)
        string specifying plot title. If set to ``None`` title will not be
        plotted.
    :param save_filename: (optional)
        string specifying plot save location/filename. If set to ``None``
        plot will not be saved.
        You can also set a desired file extension,
        for instance ``.pdf``. If the file extension is not specified, the default
        is ``.png``.

    :return:
        - **plot_handles** - list of plot handles.
    """

    if bar_color == None:
        bar_color = '#191b27'

    try:
        (n_variables, n_components) = np.shape(eigenvectors)
    except:
        eigenvectors = eigenvectors[:, np.newaxis]
        (n_variables, n_components) = np.shape(eigenvectors)

    if len(eigenvectors_indices) == 0:
        eigenvectors_indices = [i for i in range(0,n_components)]

    # Create default labels for variables:
    if len(variable_names) == 0:
        variable_names = ['$X_{' + str(i) + '}$' for i in range(0, n_variables)]

    x_range = np.arange(1, n_variables+1)

    plot_handles = []

    for n_pc in range(0,n_components):

        fig, ax = plt.subplots(figsize=(n_variables, 6))

        if plot_absolute:
            plt.bar(x_range, abs(eigenvectors[:,n_pc]), width=eigenvector_bar_width, color=bar_color, edgecolor=bar_color, align='center', zorder=2)
        else:
            plt.bar(x_range, eigenvectors[:,n_pc], width=eigenvector_bar_width, color=bar_color, edgecolor=bar_color, align='center', zorder=2)

        plt.xticks(x_range, variable_names, fontsize=font_axes, **csfont)
        if plot_absolute:
            plt.ylabel('PC-' + str(eigenvectors_indices[n_pc] + 1) + ' absolute weight [-]', fontsize=font_labels, **csfont)
        else:
            plt.ylabel('PC-' + str(eigenvectors_indices[n_pc] + 1) + ' weight [-]', fontsize=font_labels, **csfont)

        plt.grid(alpha=0.3, zorder=0)
        plt.xlim(0, n_variables+1)
        if plot_absolute == True:
            plt.ylim(-0.05,1.05)
        else:
            plt.ylim(-1.05,1.05)

        ax.spines["top"].set_visible(True)
        ax.spines["bottom"].set_visible(True)
        ax.spines["right"].set_visible(True)
        ax.spines["left"].set_visible(True)

        if title != None:
            plt.title(title, fontsize=font_title, **csfont)

        if save_filename != None:
            plt.savefig(save_filename + '-eigenvector-' + str(eigenvectors_indices[n_pc] + 1) + '.png', dpi = 500, bbox_inches='tight')

        plot_handles.append(plt)

    return plot_handles

def plot_eigenvectors_comparison(eigenvectors_tuple, legend_labels=[], variable_names=[], plot_absolute=False, color_map='coolwarm', title=None, save_filename=None):
    """
    This function plots a comparison of weights on eigenvectors.

    :param eigenvectors_tuple:
        a tuple of eigenvectors to plot. Each eigenvector inside a tuple should be a 0D array.
        It can be supplied as an attribute of the ``PCA`` class, for instance: ``(PCA.A[:,0], PCA.A[:,1])``.
    :param legend_labels:
        list of strings specifying labels for each element in the ``eigenvectors_tuple``.
    :param variable_names: (optional)
        list of strings specifying variable names.
    :param plot_absolute:
        boolean specifying whether absolute values of eigenvectors should be plotted.
    :param color_map: (optional)
        colormap to use as per ``matplotlib.cm``. Default is *coolwarm*.
    :param title: (optional)
        string specifying plot title. If set to ``None`` title will not be
        plotted.
    :param save_filename: (optional)
        string specifying plot save location/filename. If set to ``None``
        plot will not be saved.
        You can also set a desired file extension,
        for instance ``.pdf``. If the file extension is not specified, the default
        is ``.png``.

    :return:
        - **plt** - plot handle.
    """

    from matplotlib import cm

    for i in range(0, len(eigenvectors_tuple)):
        try:
            (n_variables, n_components) = np.shape(eigenvectors_tuple[i])
            raise ValueError("Eigenvectors inside eigenvectors_tuple has to be 0D arrays.")
        except:
            pass

    n_sets = len(eigenvectors_tuple)

    updated_bar_width = eigenvector_bar_width/n_sets

    (n_variables, ) = np.shape(eigenvectors_tuple[0])

    color_map_colors = cm.get_cmap(color_map, n_sets)
    sets_colors = color_map_colors(np.linspace(0, 1, n_sets))

    # Create default labels for variables:
    if len(variable_names) == 0:
        variable_names = ['$X_{' + str(i) + '}$' for i in range(0, n_variables)]

    # Create default labels for legend:
    if len(legend_labels) == 0:
        legend_labels = ['Set ' + str(i) + '' for i in range(1, n_sets+1)]

    x_range = np.arange(1, n_variables+1)

    plot_handles = []

    fig, ax = plt.subplots(figsize=(n_variables, 6))

    for n_set in range(0,n_sets):

        if plot_absolute:
            plt.bar(x_range + n_set*updated_bar_width, abs(eigenvectors_tuple[n_set]), width=updated_bar_width, color=sets_colors[n_set], edgecolor=sets_colors[n_set], align='center', zorder=2, label=legend_labels[n_set])
        else:
            plt.bar(x_range + n_set*updated_bar_width, eigenvectors_tuple[n_set], width=updated_bar_width, color=sets_colors[n_set], edgecolor=sets_colors[n_set], align='center', zorder=2, label=legend_labels[n_set])

    plt.xticks(x_range, variable_names, fontsize=font_axes, **csfont)
    if plot_absolute:
        plt.ylabel('Absolute weight [-]', fontsize=font_labels, **csfont)
    else:
        plt.ylabel('Weight [-]', fontsize=font_labels, **csfont)

    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), fancybox=True, shadow=True, ncol=n_sets, fontsize=font_legend, markerscale=marker_scale_legend)

    plt.grid(alpha=0.3, zorder=0)
    plt.xlim(0, n_variables+1)
    if plot_absolute == True:
        plt.ylim(-0.05,1.05)
    else:
        plt.ylim(-1.05,1.05)

    ax.spines["top"].set_visible(True)
    ax.spines["bottom"].set_visible(True)
    ax.spines["right"].set_visible(True)
    ax.spines["left"].set_visible(True)

    if title != None:
        plt.title(title, fontsize=font_title, **csfont)

    if save_filename != None:
        plt.savefig(save_filename + '.png', dpi = 500, bbox_inches='tight')

    return plt

def plot_eigenvalue_distribution(eigenvalues, normalized=False, title=None, save_filename=None):
    """
    This function plots eigenvalue distribution.

    :param eigenvalues:
        a 0D vector of eigenvalues to plot. It can be supplied as an attribute of the
        ``PCA`` class: ``PCA.L``.
    :param normalized: (optional)
        boolean specifying whether eigenvalues should be normalized to 1.
    :param title: (optional)
        boolean or string specifying plot title. If set to ``None``
        title will not be plotted.
    :param save_filename: (optional)
        plot save location/filename. If set to ``None`` plot will not be saved.
        You can also set a desired file extension,
        for instance ``.pdf``. If the file extension is not specified, the default
        is ``.png``.

    :return:
        - **plt** - plot handle.
    """

    color_plot = '#191b27'

    (n_eigenvalues, ) = np.shape(eigenvalues)
    x_range = np.arange(1, n_eigenvalues+1)

    fig, ax = plt.subplots(figsize=(n_eigenvalues, 6))

    if normalized:
        plt.scatter(x_range, eigenvalues/np.max(eigenvalues), c=color_plot, marker='o', s=marker_size, edgecolor='none', alpha=1, zorder=2)
        plt.plot(x_range, eigenvalues/np.max(eigenvalues), '-', c=color_plot, linewidth=line_width, alpha=1, zorder=1)
        plt.ylabel('Normalized eigenvalue [-]', fontsize=font_labels, **csfont)
        plt.ylim(-0.05,1.05)
    else:
        plt.scatter(x_range, eigenvalues, c=color_plot, marker='o', s=marker_size, edgecolor='none', alpha=1, zorder=2)
        plt.plot(x_range, eigenvalues, '-', c=color_plot, linewidth=line_width, alpha=1, zorder=1)
        plt.ylabel('Eigenvalue [-]', fontsize=font_labels, **csfont)

    plt.xticks(x_range, fontsize=font_axes, **csfont)
    plt.xlabel('$q$ [-]', fontsize=font_labels, **csfont)
    plt.xlim(0, n_eigenvalues+1)
    plt.grid(alpha=0.3, zorder=0)

    ax.spines["top"].set_visible(True)
    ax.spines["bottom"].set_visible(True)
    ax.spines["right"].set_visible(True)
    ax.spines["left"].set_visible(True)

    if title != False:
        plt.title(title, fontsize=font_title, **csfont)

    if save_filename != None:
        plt.savefig(save_filename, dpi = 500, bbox_inches='tight')

    return plt

def plot_eigenvalue_distribution_comparison(eigenvalues_tuple, legend_labels=[], normalized=False, color_map='coolwarm', title=None, save_filename=None):
    """
    This function plots eigenvalue distribution.

    :param eigenvalues_tuple:
        a tuple of eigenvalues to plot. Each vector of eigenvalues inside a tuple
        should be a 0D array. It can be supplied as an attribute of the
        ``PCA`` class, for instance: ``(PCA_1.L, PCA_2.L)``.
    :param legend_labels:
        list of strings specifying labels for each element in the ``eigenvalues_tuple``.
    :param normalized: (optional)
        boolean specifying whether eigenvalues should be normalized to 1.
    :param color_map: (optional)
        colormap to use as per ``matplotlib.cm``. Default is *coolwarm*.
    :param title: (optional)
        boolean or string specifying plot title. If set to ``None``
        title will not be plotted.
    :param save_filename: (optional)
        plot save location/filename. If set to ``None`` plot will not be saved.
        You can also set a desired file extension,
        for instance ``.pdf``. If the file extension is not specified, the default
        is ``.png``.

    :return:
        - **plt** - plot handle.
    """

    from matplotlib import cm

    n_sets = len(eigenvalues_tuple)

    (n_eigenvalues, ) = np.shape(eigenvalues_tuple[0])
    x_range = np.arange(1, n_eigenvalues+1)

    color_map_colors = cm.get_cmap(color_map, n_sets)
    sets_colors = color_map_colors(np.linspace(0, 1, n_sets))

    # Create default labels for legend:
    if len(legend_labels) == 0:
        legend_labels = ['Set ' + str(i) + '' for i in range(1, n_sets+1)]

    fig, ax = plt.subplots(figsize=(n_eigenvalues, 6))

    for n_set in range(0,n_sets):

        if normalized:
            plt.scatter(x_range, eigenvalues_tuple[n_set]/np.max(eigenvalues_tuple[n_set]), c=sets_colors[n_set].reshape(1,-1), marker='o', s=marker_size, edgecolor='none', alpha=1, zorder=2, label=legend_labels[n_set])
            plt.plot(x_range, eigenvalues_tuple[n_set]/np.max(eigenvalues_tuple[n_set]), '-', c=sets_colors[n_set], linewidth=line_width, alpha=1, zorder=1)
            plt.ylabel('Normalized eigenvalue [-]', fontsize=font_labels, **csfont)
            plt.ylim(-0.05,1.05)
        else:
            plt.scatter(x_range, eigenvalues_tuple[n_set], c=sets_colors[n_set].reshape(1,-1), marker='o', s=marker_size, edgecolor='none', alpha=1, zorder=2, label=legend_labels[n_set])
            plt.plot(x_range, eigenvalues_tuple[n_set], '-', c=sets_colors[n_set], linewidth=line_width, alpha=1, zorder=1)
            plt.ylabel('Eigenvalue [-]', fontsize=font_labels, **csfont)

    plt.xticks(x_range, fontsize=font_axes, **csfont)
    plt.xlabel('$q$ [-]', fontsize=font_labels, **csfont)
    plt.xlim(0, n_eigenvalues+1)
    plt.grid(alpha=0.3, zorder=0)

    ax.spines["top"].set_visible(True)
    ax.spines["bottom"].set_visible(True)
    ax.spines["right"].set_visible(True)
    ax.spines["left"].set_visible(True)

    plt.legend(loc='upper right', fancybox=True, shadow=True, fontsize=font_legend, markerscale=marker_scale_legend)

    if title != False:
        plt.title(title, fontsize=font_title, **csfont)

    if save_filename != None:
        plt.savefig(save_filename, dpi = 500, bbox_inches='tight')

    return plt

def plot_cumulative_variance(eigenvalues, n_components=0, title=None, save_filename=None):
    """
    This function plots the eigenvalues as bars and their cumulative sum to visualize
    the percent variance in the data explained by each Principal Component
    individually and by each Principal Component cumulatively.

    :param eigenvalues:
        a 0D vector of eigenvalues to analyze. It can be supplied as an attribute of the
        ``PCA`` class: ``PCA.L``.
    :param n_components: (optional)
        how many principal components you want to visualize (default is all).
    :param title: (optional)
        boolean or string specifying plot title. If set to ``None``
        title will not be plotted.
    :param save_filename: (optional)
        plot save location/filename. If set to ``None`` plot will not be saved.
        You can also set a desired file extension,
        for instance ``.pdf``. If the file extension is not specified, the default
        is ``.png``.

    :return:
        - **plt** - plot handle.
    """

    bar_color = '#191b27'
    line_color = '#ff2f18'

    (n_eigenvalues, ) = np.shape(eigenvalues)

    if n_components == 0:
        n_retained = n_eigenvalues
    else:
        n_retained = n_components

    x_range = np.arange(1, n_retained+1)

    fig, ax1 = plt.subplots(figsize=(n_retained, 6))

    ax1.bar(x_range, eigenvalues[0:n_retained], color=bar_color, edgecolor=bar_color, align='center', zorder=2, label='Eigenvalue')
    ax1.set_ylabel('Eigenvalue [-]', fontsize=font_labels, **csfont)
    ax1.set_ylim(0,1.05)
    ax1.grid(alpha=0.3, zorder=0)
    ax1.set_xlabel('$q$ [-]', fontsize=font_labels, **csfont)

    ax2 = ax1.twinx()
    ax2.plot(x_range, np.cumsum(eigenvalues[0:n_retained])*100, 'o-', color=line_color, zorder=2, label='Cumulative')
    ax2.set_ylabel('Variance explained [%]', color=line_color, fontsize=font_labels, **csfont)
    ax2.set_ylim(0,105)
    ax2.tick_params('y', colors=line_color)

    plt.xlim(0, n_retained+1)
    plt.xticks(x_range, fontsize=font_axes, **csfont)

    if title != False:
        plt.title(title, fontsize=font_title, **csfont)

    if save_filename != None:
        plt.savefig(save_filename, dpi = 500, bbox_inches='tight')

    return plt
