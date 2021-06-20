"""reduction.py: module for data reduction."""

__author__ = "Kamila Zdybal, Elizabeth Armstrong, Alessandro Parente and James C. Sutherland"
__copyright__ = "Copyright (c) 2020, 2021, Kamila Zdybal, Elizabeth Armstrong, Alessandro Parente and James C. Sutherland"
__credits__ = ["Department of Chemical Engineering, University of Utah, Salt Lake City, Utah, USA", "Universite Libre de Bruxelles, Aero-Thermo-Mechanics Laboratory, Brussels, Belgium"]
__license__ = "MIT"
__version__ = "1.0.0"
__maintainer__ = ["Kamila Zdybal", "Elizabeth Armstrong"]
__email__ = ["kamilazdybal@gmail.com", "Elizabeth.Armstrong@chemeng.utah.edu", "James.Sutherland@chemeng.utah.edu"]
__status__ = "Production"

import numpy as np
import copy as cp
from scipy import linalg as lg
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from PCAfold import preprocess
from PCAfold import DataSampler
from PCAfold.styles import *
from matplotlib.colors import ListedColormap
from PCAfold.preprocess import _scalings_list
import warnings

################################################################################
#
# Principal Component Analysis (PCA)
#
################################################################################

class PCA:
    """
    Enables performing Principal Component Analysis (PCA)
    of the original data set, :math:`\mathbf{X}`. For more detailed information
    on the theory of PCA the user is referred to :cite:`Jolliffe2002`.

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

    where :math:`N` is the number of observations in the original data set, :math:`\mathbf{X}`.

    Loadings matrix, :math:`\mathbf{l}`, is computed at the class initialization as well
    such that the element :math:`\mathbf{l}_{ij}` is the corresponding scaled element
    of the eigenvectors matrix, :math:`\mathbf{A}_{ij}`:

    .. math::

        \mathbf{l}_{ij} = \\frac{\mathbf{A}_{ij} \\sqrt{\mathbf{L}_j}}{\\sqrt{\mathbf{S}_{ii}}}

    where :math:`\mathbf{L}_j` is the :math:`j^{th}` eigenvalue and :math:`\mathbf{S}_{ii}`
    is the :math:`i^{th}` element on the diagonal of the covariance matrix, :math:`\mathbf{S}`.

    **Example:**

    .. code:: python

        from PCAfold import PCA
        import numpy as np

        # Generate dummy data set:
        X = np.random.rand(100,20)

        # Instantiate PCA class object:
        pca_X = PCA(X, scaling='none', n_components=2, use_eigendec=True, nocenter=False)

    :param X:
        ``numpy.ndarray`` specifying the original data set, :math:`\mathbf{X}`. It should be of size ``(n_observations,n_variables)``.
    :param scaling:
        ``str`` specifying the scaling methodology. It can be one of the following:
        ``'none'``, ``''``, ``'auto'``, ``'std'``, ``'pareto'``, ``'vast'``, ``'range'``, ``'0to1'``,
        ``'-1to1'``, ``'level'``, ``'max'``, ``'poisson'``, ``'vast_2'``, ``'vast_3'``, ``'vast_4'``.
    :param n_components: (optional)
        ``int`` specifying the number of retained principal components, :math:`q`. If set to 0 all PCs are retained. It should be a non-negative number.
    :param use_eigendec: (optional)
        ``bool`` specifying the method for obtaining eigenvalues and eigenvectors:

        * ``use_eigendec=True`` uses eigendecomposition of the covariance matrix (from ``numpy.linalg.eigh``)
        * ``use_eigendec=False`` uses Singular Value Decomposition (SVD) (from ``scipy.linalg.svd``)
    :param nocenter: (optional)
        ``bool`` specifying whether the data original data set should be centered by mean.

    **Attributes:**

    - **n_components** - (can be re-set) number of retained principal components :math:`q`.
    - **n_components_init** - (read only) number of retained principal components :math:`q` with which ``PCA`` class object was initialized.
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
        Transforms any original data set, :math:`\mathbf{X}`, to a new
        truncated basis, :math:`\mathbf{A}_q`, identified by PCA.
        It computes the :math:`q` first principal components,
        :math:`\mathbf{Z}_q`, given the original data.

        If ``nocenter=False``:

        .. math::

            \mathbf{Z}_q = (\mathbf{X} - \mathbf{C}) \cdot \mathbf{D}^{-1} \cdot \mathbf{A}_q

        If ``nocenter=True``:

        .. math::

            \mathbf{Z}_q = \mathbf{X} \cdot \mathbf{D}^{-1} \cdot \mathbf{A}_q

        Here :math:`\mathbf{C}` and :math:`\mathbf{D}` are centers and scales
        computed during ``PCA`` class initialization
        and :math:`\mathbf{A}_q` is the matrix of :math:`q` first eigenvectors
        extracted from :math:`\mathbf{A}`.

        .. warning::

            Set ``nocenter=True`` only if you know what you are doing.

            One example when ``nocenter`` should be set to ``True`` is
            when transforming chemical source terms, :math:`\mathbf{S_X}`, to
            principal components space
            (as per :cite:`Sutherland2009`)
            to obtain sources of principal components, :math:`\mathbf{S_Z}`. In
            that case :math:`\mathbf{X} = \mathbf{S_X}` and the transformation
            should be performed *without* centering:

            .. math::

                \mathbf{S}_{\mathbf{Z}, q} = \mathbf{S_X} \cdot \mathbf{D}^{-1} \cdot \mathbf{A}_q

        **Example:**

        .. code:: python

            from PCAfold import PCA
            import numpy as np

            # Generate dummy data set:
            X = np.random.rand(100,20)

            # Instantiate PCA class object:
            pca_X = PCA(X, scaling='none', n_components=2, use_eigendec=True, nocenter=False)

            # Calculate the principal components:
            principal_components = pca_X.transform(X)

        :param X:
            ``numpy.ndarray`` specifying the data set :math:`\mathbf{X}` to transform. It should be of size ``(n_observations,n_variables)``.
            Note that it does not need to be the same data set that was used to construct the ``PCA`` class object. It
            could for instance be a function of that data set. By default,
            this data set will be pre-processed with the centers and scales
            computed on the data set used when constructing the PCA object.
        :param nocenter: (optional)
            ``bool`` specifying whether ``PCA.X_center`` centers should be applied to
            center the data set before transformation.
            If ``nocenter=True`` centers will not be applied on the
            data set.

        :return:
            - **principal_components** - ``numpy.ndarray`` specifying the :math:`q` first principal components :math:`\mathbf{Z}_q`. It has size ``(n_observations,n_components)``.
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
        Calculates rank-:math:`q` reconstruction of the
        data set from the :math:`q` first principal components, :math:`\mathbf{Z}_q`.

        If ``nocenter=False``:

        .. math::

            \mathbf{X_{rec}} = \mathbf{Z}_q \mathbf{A}_q^{\mathbf{T}} \cdot \mathbf{D} + \mathbf{C}

        If ``nocenter=True``:

        .. math::

            \mathbf{X_{rec}} = \mathbf{Z}_q \mathbf{A}_q^{\mathbf{T}} \cdot \mathbf{D}

        Here :math:`\mathbf{C}` and :math:`\mathbf{D}` are centers and scales
        computed during ``PCA`` class initialization
        and :math:`\mathbf{A}_q` is the matrix of :math:`q` first eigenvectors
        extracted from :math:`\mathbf{A}`.

        .. warning::

            Set ``nocenter=True`` only if you know what you are doing.

            One example when ``nocenter`` should be set to ``True`` is
            when reconstructing chemical source terms, :math:`\mathbf{S_X}`,
            (as per :cite:`Sutherland2009`)
            from the :math:`q` first sources of principal components, :math:`\mathbf{S}_{\mathbf{Z}, q}`. In
            that case :math:`\mathbf{Z}_q = \mathbf{S}_{\mathbf{Z}, q}` and the reconstruction
            should be performed *without* uncentering:

            .. math::

                \mathbf{S_{X, rec}} = \mathbf{S}_{\mathbf{Z}, q} \mathbf{A}_q^{\mathbf{T}} \cdot \mathbf{D}

        **Example:**

        .. code:: python

            from PCAfold import PCA
            import numpy as np

            # Generate dummy data set:
            X = np.random.rand(100,20)

            # Instantiate PCA class object:
            pca_X = PCA(X, scaling='none', n_components=2, use_eigendec=True, nocenter=False)

            # Calculate the principal components:
            principal_components = pca_X.transform(X)

            # Calculate the reconstructed variables:
            X_rec = pca_X.reconstruct(principal_components)

        :param principal_components:
            ``numpy.ndarray`` of :math:`q` first principal components, :math:`\mathbf{Z}_q`. It should be of size ``(n_observations,n_variables)``.
        :param nocenter: (optional)
            ``bool`` specifying whether ``PCA.X_center`` centers should be applied to
            un-center the reconstructed data set.
            If ``nocenter=True`` centers will not be applied on the
            reconstructed data set.

        :return:
            - **X_rec** - rank-:math:`q` reconstruction of the original data set.
        """

        if not isinstance(nocenter, bool):
            raise ValueError("Parameter `nocenter` has to be a boolean.")

        (n_observations, n_components) = np.shape(principal_components)

        if n_components > self.n_variables:
            raise ValueError("Number of principal components supplied is larger than the number of eigenvectors computed by PCA.")

        # Select n_components first principal components:
        A = self.A[:, 0:n_components]

        # Calculate unscaled, uncentered approximation to the data:
        x = principal_components.dot(A.transpose())

        if nocenter:
            C_zeros = np.zeros_like(self.X_center)
            X_rec = preprocess.invert_center_scale(x, C_zeros, self.X_scale)
        else:
            X_rec = preprocess.invert_center_scale(x, self.X_center, self.X_scale)

        return(X_rec)

    def get_weights_dictionary(self, variable_names, pc_index, n_digits=10):
        """
        Creates a dictionary where keys are the names of the variables
        in the original data set :math:`\mathbf{X}` and values are the eigenvector weights
        corresponding to the principal component selected by ``pc_index``.
        This function helps in accessing weight value for a specific variable and for a specific PC.

        **Example:**

        .. code:: python

            from PCAfold import PCA
            import numpy as np

            # Generate dummy data set:
            X = np.random.rand(100,5)

            # Generate dummy variables names:
            variable_names = ['A1', 'A2', 'A3', 'A4', 'A5']

            # Instantiate PCA class object:
            pca_X = PCA(X, scaling='auto', n_components=0, use_eigendec=True, nocenter=False)

            # Create a dictionary for PC-1 weights:
            PC1_weights_dictionary = pca_X.get_weights_dictionary(variable_names, 0, n_digits=8)

        The code above will create a dictionary:

        .. code-block:: text

            {'A1': 0.63544443,
             'A2': -0.39500424,
             'A3': -0.28819465,
             'A4': 0.57000796,
             'A5': 0.17949037}

        Eigenvector weight for a specific variable can then be accessed by:

        .. code:: python

            PC1_weights_dictionary['A3']

        :param variable_names:
            ``list`` of ``str`` specifying names for all variables in the original data set, :math:`\mathbf{X}`.
        :param pc_index:
            non-negative ``int`` specifying the index of the PC to create the dictionary for. Set ``pc_index=0`` if you want to look at the first PC.
        :param n_digits: (optional)
            non-negative ``int`` specifying how many digits should be kept in rounding the eigenvector weights.

        :return:
            - **weights_dictionary** - ``dict`` of variable names as keys and selected eigenvector weights as values.
        """

        (n_variables, n_pcs) = np.shape(self.A)

        # Check that the number of variables in `variables_names` is consistent with the number of weights:
        if len(variable_names) != n_variables:
            raise ValueError("The number of variables in `variable_names` has to be equal to the number of variables in the original data set.")

        if not isinstance(pc_index, int) or pc_index < 0:
            raise ValueError("Parameter `pc_index` has to be a non-negative integer.")

        if pc_index > n_pcs - 1:
            raise ValueError("Index of the selected PC (`pc_index`) exceeds the number of eigenvectors found for this data set.")

        if not isinstance(n_digits, int) or n_digits < 0:
            raise ValueError("Parameter `n_digits` has to be a non-negative integer.")

        weights_dictionary = dict(zip(variable_names, [round(i,n_digits) for i in self.A[:,pc_index]]))

        return weights_dictionary

    def calculate_r2(self, X):
        """
        Calculates coefficient of determination, :math:`R^2`, values
        for the rank-:math:`q` reconstruction, :math:`\mathbf{X_{rec}}`, of the original
        data set, :math:`\mathbf{X}`:

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
            ``numpy.ndarray`` specifying the original data set, :math:`\mathbf{X}`. It should be of size ``(n_observations,n_variables)``.

        :return:
            - **r2** - ``numpy.ndarray`` specifying the coefficient of determination values :math:`R^2` for the rank-:math:`q` reconstruction of the original data set. It has size ``(n_variables,)``.
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
        Checks if the supplied data matrix ``X`` is consistent
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
            ``numpy.ndarray`` specifying the original data set, :math:`\mathbf{X}`. It should be of size ``(n_observations,n_variables)``.
        :param errors_are_fatal: (optional)
            ``bool`` indicating if ``ValueError`` should be raised if an incompatibility is detected.

        :return:
            - **is_consistent** - ``bool`` specifying whether or not the supplied data matrix :math:`\mathbf{X}`\
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
        Returns and optionally prints and/or saves to ``.txt`` file
        :math:`R^2` values (as per ``PCA.calculate_r2``
        function) for reconstruction of the original data set :math:`\mathbf{X}`
        as a function of number of retained principal components (PCs).

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
            ``numpy.ndarray`` specifying the original data set, :math:`\mathbf{X}`. It should be of size ``(n_observations,n_variables)``.
        :param n_pcs:
            the maximum number of PCs to consider.
        :param variable_names: (optional)
            ``list`` of strings specifying variable names. If not specified variables will be numbered.
        :param print_width: (optional)
            width of columns printed out.
        :param verbose: (optional)
            ``bool`` for printing out the table with :math:`R^2` values.
        :param save_filename: (optional)
            ``str`` specifying ``.txt`` save location/filename.

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
        Extracts Principal Variables (PVs) from a PCA.

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
            ``str`` specifying the method for determining the Principal Variables (PVs).
        :param x: (optional)
            data set to accompany ``'M2'`` method. Note that this is *only* required for the ``'M2'`` method.

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
        Writes the eigenvector matrix, :math:`\mathbf{A}`,
        loadings, :math:`\mathbf{l}`, centering, :math:`\mathbf{C}`,
        and scaling ,:math:`\mathbf{D}`, to ``.txt`` file.

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
            ``str`` specifying ``.txt`` save location/filename.
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
        Helps determine how many principal components (PCs) should be retained.
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

            # The new number of principal components that has been set:
            print(pca_X_new.n_components)

        This function provides a few methods to select the number of eigenvalues
        to be retained in the PCA reduction.

        :param method: (optional)
            ``str`` specifying the method to use in selecting retained eigenvalues.
        :param option: (optional)
            additional parameter used for the ``'TOTAL VARIANCE'`` and
            ``'INDIVIDUAL VARIANCE'`` methods. If not supplied, information
            will be obtained interactively.

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
        Calculates the U-scores (principal components):

        .. math::

            \mathbf{U_{scores}} = \mathbf{X_{cs}} \mathbf{A}_q

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
            - **u_scores** - U-scores (principal components).
        """

        self.data_consistency_check(X, errors_are_fatal=True)

        u_scores = self.transform(X)

        return(u_scores)

    def w_scores(self, X):
        """
        Calculates the W-scores which are the principal components
        scaled by the inverse square root of the corresponding eigenvalue:

        .. math::

            \mathbf{W_{scores}} = \\frac{\mathbf{Z}_q}{\\sqrt{\mathbf{L_q}}}

        where :math:`\mathbf{L_q}` are the :math:`q` first eigenvalues extracted
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
            - **w_scores** - W-scores (scaled principal components).
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
            - **iseq** - ``bool`` for ``(a == b)``.
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
            - **result** - ``bool`` for ``(a != b)``.
        """
        result = not (a == b)

        return result

################################################################################
#
# Local Principal Component Analysis
#
################################################################################

class LPCA:
    """
    Enables performing local Principal Component Analysis (LPCA)
    of the original data set, :math:`\mathbf{X}`, partitioned into clusters.

    **Example:**

    .. code:: python

        from PCAfold import LPCA
        import numpy as np

        # Generate dummy data set:
        X = np.random.rand(100,10)

        # Generate dummy vector of cluster classifications:
        idx = np.zeros((100,))
        idx[50:80] = 1
        idx = idx.astype(int)

        # Instantiate LPCA class object:
        lpca_X = LPCA(X, idx, scaling='none', n_components=2)

        # Access the local eigenvectors in the first cluster:
        A_k1 = lpca_X.A[0]

        # Access the local eigenvalues in the first cluster:
        L_k1 = lpca_X.L[0]

        # Access the local principal components in the first cluster:
        Z_k1 = lpca_X.principal_components[0]

    :param X:
        ``numpy.ndarray`` specifying the original data set, :math:`\mathbf{X}`. It should be of size ``(n_observations,n_variables)``.
    :param idx:
        ``numpy.ndarray`` of cluster classifications. It should be of size ``(n_observations,)`` or ``(n_observations,1)``.
    :param scaling: (optional)
        ``str`` specifying the scaling methodology. It can be one of the following:
        ``'none'``, ``''``, ``'auto'``, ``'std'``, ``'pareto'``, ``'vast'``, ``'range'``, ``'0to1'``,
        ``'-1to1'``, ``'level'``, ``'max'``, ``'poisson'``, ``'vast_2'``, ``'vast_3'``, ``'vast_4'``.
    :param n_components: (optional)
        ``int`` specifying the number of returned eigenvectors, eigenvalues and principal components, :math:`q`. If set to 0 all are returned.
    :param use_eigendec: (optional)
        ``bool`` specifying the method for obtaining eigenvalues and eigenvectors:

        * ``use_eigendec=True`` uses eigendecomposition of the covariance matrix (from ``numpy.linalg.eigh``)
        * ``use_eigendec=False`` uses Singular Value Decomposition (SVD) (from ``scipy.linalg.svd``)
    :param nocenter: (optional)
        ``bool`` specifying whether data should be centered by mean.

    **Attributes:**

    - **A** - (read only) ``list`` of ``numpy.ndarray`` specifying the local eigenvectors, :math:`\mathbf{A}`. Each list element corresponds to eigenvectors in a single cluster.
    - **L** - (read only) ``list`` of ``numpy.ndarray`` specifying the local eigenvalues, :math:`\mathbf{L}`. Each list element corresponds to eigenvalues in a single cluster.
    - **principal_components** - (read only) ``list`` of ``numpy.ndarray`` specifying the local principal components, :math:`\mathbf{Z}`. Each list element corresponds to principal components in a single cluster.
    """

    def __init__(self, X, idx, scaling='std', n_components=0, use_eigendec=True, nocenter=False):

        if not isinstance(X, np.ndarray):
            raise ValueError("Parameter `X` has to be of type `numpy.ndarray`.")

        try:
            (n_observations, n_variables) = np.shape(X)
        except:
            raise ValueError("Parameter `X` has to have size `(n_observations,n_variables)`.")

        try:
            (n_observations_idx, ) = np.shape(idx)
            n_idx = 1
        except:
            (n_observations_idx, n_idx) = np.shape(idx)

        if n_idx != 1:
            raise ValueError("Parameter `idx` has to have size `(n_observations,)` or `(n_observations,1)`.")

        if isinstance(idx, np.ndarray):
            if not all(isinstance(i, np.integer) for i in idx.ravel()):
                raise ValueError("Parameter `idx` can only contain integers.")
        else:
            raise ValueError("Parameter `idx` has to be of type `numpy.ndarray`.")

        if n_observations_idx != n_observations:
            raise ValueError('Vector of cluster classifications `idx` has different number of observations than the original data set `X`.')

        if (len(np.unique(idx)) != (np.max(idx)+1)) or (np.min(idx) != 0):
            (idx, _) = preprocess.degrade_clusters(idx, verbose)

        self.__idx = idx.ravel()

        if not isinstance(scaling, str):
            raise ValueError("Parameter `scaling` has to be a string.")
        else:
            if scaling.lower() not in _scalings_list:
                raise ValueError("Unrecognized scaling method.")
            else:
                self.__scaling = scaling.upper()

        if not isinstance(n_components, int) or isinstance(n_components, bool):
            raise ValueError("Parameter `n_components` has to be an integer.")
        else:
            if (n_components < 0) or (n_components > n_variables):
                raise ValueError("Parameter `n_components` cannot be negative or larger than number of variables in a data set.")
            else:
                if n_components > 0:
                    self.__n_components = n_components
                else:
                    self.__n_components = n_variables

        if not isinstance(use_eigendec, bool):
            raise ValueError("Parameter `use_eigendec` has to be a boolean.")

        if not isinstance(nocenter, bool):
            raise ValueError("Parameter `nocenter` has to be a boolean.")

        n_clusters = len(np.unique(idx))

        # Initialize the outputs:
        eigenvectors = []
        eigenvalues = []
        PCs = []

        for k in range(0, n_clusters):

            # Extract local cluster:
            X_k = X[self.__idx==k,:]

            # Remove constant variables from local cluster:
            (X_removed, idx_removed, idx_retained) = preprocess.remove_constant_vars(X_k, maxtol=1e-12, rangetol=0.0001)

            # Perform PCA in local cluster:
            pca = PCA(X_removed, scaling=scaling, n_components=self.__n_components, use_eigendec=use_eigendec, nocenter=nocenter)
            Z = pca.transform(X_removed, nocenter=False)

            # Append the local eigenvectors, eigenvalues and PCs:
            eigenvectors.append(pca.A[:,0:self.__n_components])
            eigenvalues.append(pca.L[0:self.__n_components])
            PCs.append(Z)

        self.__A = eigenvectors
        self.__L = eigenvalues
        self.__principal_components = PCs

    @property
    def A(self):
        return self.__A

    @property
    def L(self):
        return self.__L

    @property
    def principal_components(self):
        return self.__principal_components

    def local_correlation(self, variable, index=0, metric='pearson', verbose=False):
        """
        Computes a correlation in each cluster and a globally-averaged correlation between the local
        principal component, PC, and some specified variable, :math:`\\phi`.
        The average is taken from each of the :math:`k` clusters.
        Correlation in the :math:`n^{th}` cluster is referred to as :math:`r_n(\\mathrm{PC}, \\phi)`.

        Available correlation functions are:

        - Pearson correlation coefficient (PCC), set ``metric='pearson'``:

        .. math::

            r_n(\\mathrm{PC}, \\phi) = \\mathrm{abs} \\Bigg( \\frac{\\sum_{i=1}^{N_n} (\\mathrm{PC}_i - \\overline{\\mathrm{PC}}) (\\phi_i - \\bar{\\phi})}{\\sqrt{\\sum_{i=1}^{N_n} (\\mathrm{PC}_i - \\overline{\\mathrm{PC}})^2} \\sqrt{\\sum_{i=1}^{N_n} (\\phi_i - \\bar{\\phi})^2}} \\Bigg)

        where :math:`N_n` is the number of observations in the :math:`n^{th}` cluster.

        - Distance correlation (dCor), set ``metric='dcor'``:

        .. math::

            r_n(\\mathrm{PC}, \\phi) = \\sqrt{ \\frac{\\mathrm{dCov}(\\mathrm{PC}_n, \\phi_n)}{\\mathrm{dCov}(\\mathrm{PC}_n, \\mathrm{PC}_n) \\mathrm{dCov}(\\phi_n, \\phi_n)} }

        where :math:`\\mathrm{dCov}` is the distance covariance computed for any two variables, :math:`X` and :math:`Y`, as:

        .. math::

            \\mathrm{dCov}(X,Y) = \\sqrt{ \\frac{1}{N^2} \\sum_{i,j=1}^N x_{i,j} y_{i,j}}

        where :math:`x_{i,j}` and :math:`y_{i,j}` are the elements of the
        double-centered Euclidean distances matrices for :math:`X` and :math:`Y`
        observations respectively. :math:`N` is the total number of observations.
        Note, that the distance correlation computation allows :math:`X` and :math:`Y`
        to have different dimensions.

        .. note::

            The distance correlation computation requires the ``dcor`` module. You can install it through:

            ``pip install dcor``

        Globally-averaged correlation metric is computed in two variants:

        - Weighted, where each local correlation is weighted by the size of each cluster:

        .. math::

            \\bar{r} = \\frac{1}{N} \\sum_{n=1}^k N_n r_n(\\mathrm{PC}, \\phi)

        - Unweighted, which computes an arithmetic average of local correlations from all clusters:

        .. math::

            r = \\frac{1}{k} \\sum_{n=1}^k r_n(\\mathrm{PC}, \\phi)

        **Example:**

        .. code::

            from PCAfold import predefined_variable_bins, LPCA
            import numpy as np

            # Generate dummy data set:
            x = np.linspace(-1,1,1000)
            y = -x**2 + 1
            X = np.hstack((x[:,None], y[:,None]))

            # Generate dummy vector of cluster classifications:
            (idx, _) = predefined_variable_bins(x, [-0.9, 0, 0.6])

            # Instantiate LPCA class object:
            lpca = LPCA(X, idx, scaling='none')

            # Compute local Pearson correlation coefficient between PC-1 and y:
            (local_correlations, weighted, unweighted) = lpca.local_correlation(y, index=0, metric='pearson', verbose=True)

        With ``verbose=True`` we will see some detailed information:

        .. code-block:: text

            PCC in cluster 1:	0.999996
            PCC in cluster 2:	-0.990817
            PCC in cluster 3:	0.983221
            PCC in cluster 4:	0.999838

            Globally-averaged weighted correlation: 0.990801
            Globally-averaged unweighted correlation: 0.993468

        :param variable:
            ``numpy.ndarray`` specifying the variable, :math:`\\phi`, for correlation computation.
            It should be of size ``(n_observations,)`` or ``(n_observations,1)`` or ``(n_observations,n_variables)`` when ``metric='dcor'``.
        :param index:
            ``int`` specifying the index of the local principal component for correlation computation.
            Set ``index=0`` if you want to look at the first PC.
        :param metric:
            ``str`` specifying the correlation metric to use. It can be ``'pearson'`` or ``'dcor'``.
        :param verbose: (optional)
            ``bool`` for printing verbose details.

        :return:
            - **local_correlations** - ``numpy.ndarray`` specifying the computed correlation in each cluster. It has size ``(k,)``.
            - **weighted** - ``float`` specifying the globally-averaged weighted correlation.
            - **unweighted** - ``float`` specifying the globally-averaged unweighted correlation.
        """

        __metrics = ['pearson', 'dcor']

        if not isinstance(variable, np.ndarray):
            raise ValueError("Parameter `variable` has to be of type `numpy.ndarray`.")

        if metric != 'dcor':
            try:
                (n_observations,) = np.shape(variable)
                n_dim = 1
            except:
                (n_observations,n_dim) = np.shape(variable)

            if n_dim != 1:
                raise ValueError("Parameter `variable` has to have size `(n_observations,)` or `(n_observations,1)`.")
        else:
            try:
                (n_observations,) = np.shape(variable)
            except:
                (n_observations,_) = np.shape(variable)

        (n_observations_idx,) = np.shape(self.__idx)

        if n_observations != n_observations_idx:
            raise ValueError("Parameter `variable` has different number of observations than parameter `idx`.")

        if not isinstance(index, int):
            raise ValueError("Parameter `index` has to be of type `int`.")

        if index < 0:
            raise ValueError("Parameter `index` has to be a positive `int`.")

        if index > self.__n_components-1:
            raise ValueError("Parameter `index` is larger than the number of principal components `n_components` at `LPCA` class initialization.")

        if metric not in __metrics:
            raise ValueError("Parameter `metric` can be `'pearson'` or `'dcor'`.")

        if metric == 'dcor':
            try:
                from dcor import distance_correlation
            except:
                raise ValueError("Distance correlation requires the `dcor` module: `pip install dcor`.")

        if not isinstance(verbose, bool):
            raise ValueError("Parameter `verbose` has to be of type `bool`.")

        n_clusters = len(np.unique(self.__idx))

        weighted_collected = []
        unweighted_collected = []

        local_correlations = np.zeros((n_clusters,))

        for k in range(0, n_clusters):

            indices = list(np.where(self.__idx==k)[0])

            # Extract the variable in the current cluster:
            try:
                local_variable = variable[indices]
            except:
                local_variable = variable[indices,:]

            # Extract the principal component in the current cluster:
            local_pc = self.principal_components[k][:,index]

            if metric == 'pearson':

                (local_correlation, _) = pearsonr(local_pc, local_variable.ravel())
                local_correlations[k] = local_correlation

                if verbose:
                    print('PCC in cluster ' + str(k+1) + ':\t' + str(round(local_correlation,6)))

            elif metric == 'dcor':

                local_correlation = distance_correlation(local_pc, local_variable)

                if verbose:
                    print('dCor in cluster ' + str(k+1) + ':\t' + str(round(local_correlation,6)))

            # Compute the weighted correlation:
            weighted_collected.append(abs(local_correlation) * len(indices))

            # Compute the unweighted correlation:
            unweighted_collected.append(abs(local_correlation))

        # Compute the globally averaged weighted correlation:
        weighted = np.sum(weighted_collected) / n_observations

        # Compute the globally averaged unweighted correlation:
        unweighted = np.sum(unweighted_collected) / n_clusters

        if verbose:
            print('\nGlobally-averaged weighted correlation: ' + str(round(weighted,6)))
            print('Globally-averaged unweighted correlation: ' + str(round(unweighted,6)))

        return (local_correlations, weighted, unweighted)

################################################################################
#
# Subset Principal Component Analysis
#
################################################################################

class SubsetPCA:
    """
    Enables performing Principal Component Analysis (PCA) of a subset of the
    original data set, :math:`\mathbf{X}`.

    **Example:**

    .. code:: python

        from PCAfold import SubsetPCA
        import numpy as np

        # Generate dummy data set:
        X = np.random.rand(100,10)

        # Instantiate SubsetPCA class object:
        subset_pca_X = SubsetPCA(X, full_sequence=True, scaling='std', n_components=2)

    :param X:
        ``numpy.ndarray`` specifying the original data set, :math:`\mathbf{X}`. It should be of size ``(n_observations,n_variables)``.
    :param X_source: (optional)
        ``numpy.ndarray`` specifying the source terms, :math:`\mathbf{S_X}`, corresponding to the state-space
        variables in :math:`\mathbf{X}`. This parameter is applicable to data sets
        representing reactive flows. More information can be found in :cite:`Sutherland2009`.
    :param full_sequence: (optional)
        ``bool`` specifying if a full sequence of subset PCAs should be performed. If set to ``True``, it is assumed that variables in :math:`\mathbf{X}` have been ordered
        according to some criterion. A sequence of subset PCAs will then be performed starting from the first ``n_components+1`` variables
        and gradually adding the next variable in :math:`\mathbf{X}`. When ``full_sequence=True``, parameter ``subset_indices`` will be ignored
        and the class attributes will be of type ``list`` of ``numpy.ndarray``. Each element in those lists corresponds to one subset PCA in a sequence.
    :param subset_indices: (optional)
        ``list`` specifying the indices of columns to be taken from the original data set to form a subset of a data set.
    :param variable_names: (optional)
        ``list`` of ``str`` specifying the names of variables in :math:`\mathbf{X}`. It should have length ``n_variables`` and each element should correspond to a column in :math:`\mathbf{X}`.
    :param scaling: (optional)
        ``str`` specifying the scaling methodology. It can be one of the following:
        ``'none'``, ``''``, ``'auto'``, ``'std'``, ``'pareto'``, ``'vast'``, ``'range'``, ``'0to1'``,
        ``'-1to1'``, ``'level'``, ``'max'``, ``'poisson'``, ``'vast_2'``, ``'vast_3'``, ``'vast_4'``.
    :param n_components: (optional)
        ``int`` specifying the number of retained principal components, :math:`q`. If set to 0 all PCs are retained. It should be a non-negative number.
    :param use_eigendec: (optional)
        ``bool`` specifying the method for obtaining eigenvalues and eigenvectors:

        * ``use_eigendec=True`` uses eigendecomposition of the covariance matrix (from ``numpy.linalg.eigh``)
        * ``use_eigendec=False`` uses Singular Value Decomposition (SVD) (from ``scipy.linalg.svd``)
    :param nocenter: (optional)
        ``bool`` specifying whether the data original data set should be centered by mean.

    **Attributes:**

    - **S** - (read only) ``numpy.ndarray`` or ``list`` of ``numpy.ndarray`` specifying the covariance matrix, :math:`\mathbf{S}`.
    - **L** - (read only) ``numpy.ndarray`` or ``list`` of ``numpy.ndarray`` specifying the vector of eigenvalues, :math:`\mathbf{L}`.
    - **A** - (read only) ``numpy.ndarray`` or ``list`` of ``numpy.ndarray`` specifying the matrix of eigenvectors, :math:`\mathbf{A}`.
    - **principal_components** - (read only) ``list`` of ``numpy.ndarray`` specifying the local principal components, :math:`\mathbf{Z}`.
    """

    def __init__(self, X, X_source=None, full_sequence=True, subset_indices=None, variable_names=None, scaling='std', n_components=2, use_eigendec=True, nocenter=False, verbose=False):

        if not isinstance(X, np.ndarray):
            raise ValueError("Parameter `X` has to be of type `numpy.ndarray`.")

        try:
            (n_observations, n_variables) = np.shape(X)
        except:
            raise ValueError("Parameter `X` has to have size `(n_observations,n_variables)`.")

        if subset_indices is not None:
            if not isinstance(subset_indices, list):
                raise ValueError("Parameter `subset_indices` has to be of type `list`.")

        if variable_names is not None:
            if not isinstance(variable_names, list):
                raise ValueError("Parameter `variable_names` has to be of type `list`.")
            else:
                n_names = len(variable_names)
        else:
            variable_names = []
            for i in range(0,n_variables):
                variable_names.append('X' + str(i))
            n_names = len(variable_names)

        if n_variables != n_names:
            raise ValueError("Parameters `X` and `variables_names` have different number of variables.")

        if not isinstance(scaling, str):
            raise ValueError("Parameter `scaling` has to be a string.")
        else:
            if scaling.lower() not in _scalings_list:
                raise ValueError("Unrecognized scaling method.")
            else:
                self.__scaling = scaling.upper()

        if not isinstance(n_components, int) or isinstance(n_components, bool):
            raise ValueError("Parameter `n_components` has to be an integer.")
        else:
            if (n_components < 0) or (n_components > n_variables):
                raise ValueError("Parameter `n_components` cannot be negative or larger than number of variables in a data set.")

        if not isinstance(use_eigendec, bool):
            raise ValueError("Parameter `use_eigendec` has to be a boolean.")

        if not isinstance(nocenter, bool):
            raise ValueError("Parameter `nocenter` has to be a boolean.")

        if not isinstance(verbose, bool):
            raise ValueError("Parameter `verbose` has to be a boolean.")

        if full_sequence:

            covariance_matrix = []
            eigenvectors = []
            eigenvalues = []
            PCs = []
            PC_source_terms = []
            variable_sequence = []

            if verbose:
                print('Full sequence of subset PCAs will be performed.')

            if subset_indices is not None:
                if len(subset_indices) != 0:
                    warnings.warn('Parameter `subset_indices` will be ignored.')

            for i_subset in range(n_components+1, n_variables+1):

                # Perform global PCA on the current subset:
                global_pca = PCA(X[:,0:i_subset], scaling=scaling, n_components=n_components, use_eigendec=use_eigendec, nocenter=nocenter)
                global_PCs = global_pca.transform(X[:,0:i_subset], nocenter=False)
                if X_source is not None: global_PC_sources = global_pca.transform(X_source[:,0:i_subset], nocenter=True)

                # Append the current subset PCA solution:
                covariance_matrix.append(global_pca.S)
                eigenvectors.append(global_pca.A[:,0:n_components])
                eigenvalues.append(global_pca.L[0:n_components])
                PCs.append(global_PCs)
                if X_source is not None: PC_source_terms.append(global_PC_sources)
                variable_sequence.append(variable_names[0:i_subset])

        else:

            # Perform global PCA on the current subset:
            global_pca = PCA(X[:,subset_indices], scaling=scaling, n_components=n_components, use_eigendec=use_eigendec, nocenter=nocenter)

            # Append the current subset PCA solution:
            covariance_matrix = global_pca.S
            eigenvectors = global_pca.A[:,0:n_components]
            eigenvalues = global_pca.L[0:n_components]
            PCs = global_pca.transform(X[:,subset_indices], nocenter=False)
            if X_source is not None: PC_source_terms = global_pca.transform(X_source[:,subset_indices], nocenter=True)
            variable_sequence = list(variable_names[i] for i in subset_indices)

        self.__S = covariance_matrix
        self.__A = eigenvectors
        self.__L = eigenvalues
        self.__principal_components = PCs
        self.__PC_source_terms = PC_source_terms
        self.__variable_sequence = variable_sequence

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
    def principal_components(self):
        return self.__principal_components

    @property
    def PC_source_terms(self):
        return self.__PC_source_terms

    @property
    def variable_sequence(self):
        return self.__variable_sequence

################################################################################
#
# Principal Component Analysis on sampled data sets
#
################################################################################

def pca_on_sampled_data_set(X, idx_X_r, scaling, n_components, biasing_option, X_source=[]):
    """
    Performs PCA on sampled data set, :math:`\mathbf{X_r}`, with one
    of the four implemented options.

    Reach out to the
    `Biasing options <https://pcafold.readthedocs.io/en/latest/user/data-reduction.html#biasing-option-1>`_
    section of the documentation for more information on the available options.

    **Example:**

    .. code::

        from PCAfold import pca_on_sampled_data_set, DataSampler
        import numpy as np

        # Generate dummy data set:
        X = np.random.rand(100,10)

        # Generate dummy sampling indices:
        idx = np.zeros((100,)).astype(int)
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
        ``str`` specifying the scaling methodology. It can be one of the following:
        ``'none'``, ``''``, ``'auto'``, ``'std'``, ``'pareto'``, ``'vast'``, ``'range'``, ``'0to1'``,
        ``'-1to1'``, ``'level'``, ``'max'``, ``'poisson'``, ``'vast_2'``, ``'vast_3'``, ``'vast_4'``.
    :param n_components:
        number of :math:`q` first principal components that will be saved.
    :param biasing_option:
        integer specifying biasing option. Can only attain values 1, 2, 3 or 4.
    :param X_source: (optional)
        source terms :math:`\mathbf{S_X}` corresponding to the state-space
        variables in :math:`\mathbf{X}`. This parameter is applicable to data sets
        representing reactive flows. More information can be found in :cite:`Sutherland2009`.

    :return:
        - **eigenvalues** - biased eigenvalues :math:`\mathbf{L_r}`.
        - **eigenvectors** - biased eigenvectors :math:`\mathbf{A_r}`.
        - **pc_scores** - :math:`q` first biased principal components :math:`\mathbf{Z_r}`.
        - **pc_sources** - :math:`q` first biased sources of principal components :math:`\mathbf{S_{Z_r}}`. More information can be found in :cite:`Sutherland2009`.\
        This parameter is only computed if ``X_source`` input parameter was specified.
        - **C** - a vector of centers :math:`\mathbf{C}` that were used to pre-process the\
        original full data set :math:`\mathbf{X}`.
        - **D** - a vector of scales :math:`\mathbf{D}` that were used to pre-process the\
        original full data set :math:`\mathbf{X}`.
        - **C_r** - a vector of centers :math:`\mathbf{C_r}` that were used to pre-process the\
        sampled data set :math:`\mathbf{X_r}`.
        - **D_r** - a vector of scales :math:`\mathbf{D_r}` that were used to pre-process the\
        sampled data set :math:`\mathbf{X_r}`.
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

            # Scale sources with the current scalings D_r:
            X_source_cs = np.divide(X_source, D_r)

            # Compute PC-sources:
            pc_sources = X_source_cs.dot(eigenvectors[:,0:n_components])

    if len(X_source) == 0:

        pc_sources = []

    return(eigenvalues, eigenvectors, pc_scores, pc_sources, C, D, C_r, D_r)

def analyze_centers_change(X, idx_X_r, variable_names=[], plot_variables=[], legend_label=[], title=None, save_filename=None):
    """
    Analyzes the change in normalized centers computed on the
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
        idx = np.zeros((100,)).astype(int)
        idx[50:80] = 1
        selection = DataSampler(idx)
        (idx_X_r, _) = selection.number(20, test_selection_option=1)

        # Analyze the change in normalized centers:
        (normalized_C, normalized_C_r, center_movement_percentage, plt) = analyze_centers_change(X, idx_X_r)

    :param X:
        ``numpy.ndarray`` specifying the original data set, :math:`\mathbf{X}`. It should be of size ``(n_observations,n_variables)``.
    :param idx_X_r:
        vector of indices that should be extracted from :math:`\mathbf{X}` to
        form :math:`\mathbf{X_r}`.
    :param variable_names: (optional)
        ``list`` of ``str`` specifying variable names.
    :param plot_variables: (optional)
        ``list`` of ``int`` specifying indices of variables to be plotted.
        By default, all variables are plotted.
    :param legend_label: (optional)
        ``list`` of ``str`` specifying labels for the legend. First entry will refer
        to :math:`||\mathbf{C}||` and second entry to :math:`||\mathbf{C_r}||`.
        If the list is empty, legend will not be plotted.
    :param title: (optional)
        ``str`` specifying plot title. If set to ``None`` title will not be
        plotted.
    :param save_filename: (optional)
        ``str`` specifying plot save location/filename. If set to ``None``
        plot will not be saved. You can also set a desired file extension,
        for instance ``.pdf``. If the file extension is not specified, the default
        is ``.png``.

    :return:
        - **normalized_C** - normalized centers :math:`||\mathbf{C}||`.
        - **normalized_C_r** - normalized centers :math:`||\mathbf{C_r}||`.
        - **center_movement_percentage** - percentage :math:`p`\
        measuring the relative change in normalized centers.
        - **plt** - ``matplotlib.pyplot`` plot handle.
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

    fig, ax = plt.subplots(figsize=(n_variables, 4))

    plt.scatter(x_range, normalized_C, c=color_X, marker='o', s=marker_size, edgecolor='none', alpha=1, zorder=2)
    plt.scatter(x_range, normalized_C_r, c=color_X_r, marker='>', s=marker_size, edgecolor='none', alpha=1, zorder=2)
    plt.xticks(x_range, variable_names, fontsize=font_axes, **csfont)
    plt.yticks(fontsize=font_axes, **csfont)
    plt.ylabel('Normalized center [-]', fontsize=font_labels, **csfont)
    plt.ylim(-0.05,1.05)
    plt.xlim(0, n_variables+1.5)
    plt.grid(alpha=grid_opacity, zorder=0)

    if len(legend_label) != 0:
        lgnd = plt.legend(legend_label, fontsize=font_legend, markerscale=marker_scale_legend, loc="upper right")

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

    if save_filename != None:
        plt.savefig(save_filename, dpi=save_dpi, bbox_inches='tight')

    return(normalized_C, normalized_C_r, center_movement_percentage, plt)

def analyze_eigenvector_weights_change(eigenvectors, variable_names=[], plot_variables=[], normalize=False, zero_norm=False, legend_label=[], title=None, save_filename=None):
    """
    Analyzes the change of weights on an eigenvector obtained
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
        ``list`` of ``str`` specifying variable names.
    :param plot_variables: (optional)
        list of integers specifying indices of variables to be plotted.
        By default, all variables are plotted.
    :param normalize: (optional)
        ``bool`` specifying whether weights should be normlized at all.
        If set to false, the absolute values are plotted.
    :param zero_norm: (optional)
        ``bool`` specifying whether weights should be normalized between 0 and 1.
        By default they are not normalized to start at 0.
        Only has effect if ``normalize=True``.
    :param legend_label: (optional)
        ``list`` of ``str`` specifying labels for the legend. If the list is empty,
        legend will not be plotted.
    :param title: (optional)
        ``str`` specifying plot title. If set to ``None`` title will not be
        plotted.
    :param save_filename: (optional)
        ``str`` specifying plot save location/filename. If set to ``None``
        plot will not be saved. You can also set a desired file extension,
        for instance ``.pdf``. If the file extension is not specified, the default
        is ``.png``.

    :return:
        - **plt** - ``matplotlib.pyplot`` plot handle.
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

        fig, ax = plt.subplots(figsize=(n_variables, 4))

        plt.scatter(x_range, eigenvectors[:,0], c=color_X, marker='o', s=marker_size, edgecolor='none', alpha=1, zorder=2)
        plt.scatter(x_range, eigenvectors[:,-1], c=color_X_r, marker='>', s=marker_size, edgecolor='none', alpha=1, zorder=2)

        if len(legend_label) != 0:

            lgnd = plt.legend(legend_label, fontsize=font_legend, markerscale=marker_scale_legend, loc="upper right")

            lgnd.legendHandles[0]._sizes = [marker_size*1.5]
            lgnd.legendHandles[1]._sizes = [marker_size*1.5]
            plt.setp(lgnd.texts, **csfont)

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
        plt.grid(alpha=grid_opacity, zorder=0)

        if title != None:
            plt.title(title, fontsize=font_title, **csfont)

        ax.spines["top"].set_visible(True)
        ax.spines["bottom"].set_visible(True)
        ax.spines["right"].set_visible(True)
        ax.spines["left"].set_visible(True)

    # When there are more than two versions plot the trends:
    else:

        color_range = np.arange(0, n_versions)

        # Plot the eigenvector weights movement:
        fig, ax = plt.subplots(figsize=(n_variables, 4))

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
        plt.grid(alpha=grid_opacity)

        if title != None:
            plt.title(title, fontsize=font_title, **csfont)

        cbar = plt.colorbar(scat, ticks=[0, round((n_versions-1)/2), n_versions-1])

    if save_filename != None:
        plt.savefig(save_filename, dpi=save_dpi, bbox_inches='tight')

    return plt

def analyze_eigenvalue_distribution(X, idx_X_r, scaling, biasing_option, legend_label=[], title=None, save_filename=None):
    """
    Analyzes the normalized eigenvalue distribution when PCA is
    performed on the original data set :math:`\mathbf{X}` and on the sampled
    data set :math:`\mathbf{X_r}`.

    Reach out to the
    `Biasing options <https://pcafold.readthedocs.io/en/latest/user/data-reduction.html#biasing-option-1>`_
    section of the documentation for more information on the available options.

    **Example:**

    .. code:: python

        from PCAfold import analyze_eigenvalue_distribution, DataSampler
        import numpy as np

        # Generate dummy data set:
        X = np.random.rand(100,10)

        # Generate dummy sampling indices:
        idx = np.zeros((100,)).astype(int)
        idx[50:80] = 1
        selection = DataSampler(idx)
        (idx_X_r, _) = selection.number(20, test_selection_option=1)

        # Analyze the change in eigenvalue distribution:
        plt = analyze_eigenvalue_distribution(X, idx_X_r, 'auto', biasing_option=1)

    :param X:
        ``numpy.ndarray`` specifying the original data set, :math:`\mathbf{X}`. It should be of size ``(n_observations,n_variables)``.
    :param idx_X_r:
        vector of indices that should be extracted from :math:`\mathbf{X}` to
        form :math:`\mathbf{X_r}`.
    :param scaling:
        ``str`` specifying the scaling methodology. It can be one of the following:
        ``'none'``, ``''``, ``'auto'``, ``'std'``, ``'pareto'``, ``'vast'``, ``'range'``, ``'0to1'``,
        ``'-1to1'``, ``'level'``, ``'max'``, ``'poisson'``, ``'vast_2'``, ``'vast_3'``, ``'vast_4'``.
    :param biasing_option:
        ``int`` specifying biasing option.
        Can only attain values 1, 2, 3 or 4.
    :param legend_label: (optional)
        ``list`` of ``str`` specifying labels for the legend. First entry will refer
        to :math:`\mathbf{X}` and second entry to :math:`\mathbf{X_r}`.
        If the list is empty, legend will not be plotted.
    :param title: (optional)
        ``str`` specifying plot title. If set to ``None`` title will not be
        plotted.
    :param save_filename: (optional)
        ``str`` specifying plot save location/filename. If set to ``None``
        plot will not be saved. You can also set a desired file extension,
        for instance ``.pdf``. If the file extension is not specified, the default
        is ``.png``.

    :return:
        - **plt** - ``matplotlib.pyplot`` plot handle.
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

    fig, ax = plt.subplots(figsize=(n_variables, 4))

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
    plt.grid(alpha=grid_opacity, zorder=0)

    ax.spines["top"].set_visible(True)
    ax.spines["bottom"].set_visible(True)
    ax.spines["right"].set_visible(True)
    ax.spines["left"].set_visible(True)

    if title != False:
        plt.title(title, fontsize=font_title, **csfont)

    if save_filename != None:
        plt.savefig(save_filename, dpi=save_dpi, bbox_inches='tight')

    return plt

def equilibrate_cluster_populations(X, idx, scaling, n_components, biasing_option, X_source=[], n_iterations=10, stop_iter=0, random_seed=None, verbose=False):
    """
    Gradually (in ``n_iterations``) equilibrates cluster populations heading towards
    population of the smallest cluster, in each cluster.

    At each iteration it generates a reduced data set :math:`\mathbf{X_r}^{(i)}`
    made up from new populations, performs PCA on that data set to find the
    :math:`i^{th}` version of the eigenvectors. Depending on the option
    selected, it then does the projection of a data set (and optionally also
    its sources) onto the found eigenvectors.

    Reach out to the
    `Biasing options <https://pcafold.readthedocs.io/en/latest/user/data-reduction.html#biasing-option-1>`_
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

    .. image:: ../images/cluster-equilibration-scheme.svg
        :width: 700
        :align: center

    Future implementation will include equilibration that slows down close to
    equilibrium.

    **Interpretation for the outputs:**

    This function returns 3D arrays ``eigenvectors``, ``pc_scores`` and
    ``pc_sources`` that have the following structure:

    .. image:: ../images/cbpca-equilibrate-outputs.svg
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
        ``numpy.ndarray`` specifying the original data set, :math:`\mathbf{X}`. It should be of size ``(n_observations,n_variables)``.
    :param idx:
        vector of cluster classifications.
        The first cluster has index 0.
    :param scaling:
        ``str`` specifying the scaling methodology. It can be one of the following:
        ``'none'``, ``''``, ``'auto'``, ``'std'``, ``'pareto'``, ``'vast'``, ``'range'``, ``'0to1'``,
        ``'-1to1'``, ``'level'``, ``'max'``, ``'poisson'``, ``'vast_2'``, ``'vast_3'``, ``'vast_4'``.
    :param X_source:
        source terms :math:`\mathbf{S_X}` corresponding to the state-space
        variables in :math:`\mathbf{X}`. This parameter is applicable to data sets
        representing reactive flows. More information can be found in :cite:`Sutherland2009`.
    :param n_components:
        number of :math:`q` first principal components that will be saved.
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
        ``bool`` for printing verbose details.

    :return:
        - **eigenvalues** - collected eigenvalues from each iteration.
        - **eigenvectors_matrix** - collected eigenvectors from each iteration.\
        This is a 3D array of size ``(n_variables, n_components, n_iterations+1)``.
        - **pc_scores_matrix** - collected principal components from each iteration.\
        This is a 3D array of size ``(n_observations, n_components, n_iterations+1)``.
        - **pc_sources_matrix** - collected sources of principal components from each iteration.\
        This is a 3D array of size ``(n_observations, n_components, n_iterations+1)``.
        - **idx_train** - the final training indices from the equilibrated iteration.
        - **C_r** - a vector of final centers that were used to center\
        the data set at the last (equilibration) iteration.
        - **D_r** - a vector of final scales that were used to scale the\
        data set at the last (equilibration) iteration.
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

def plot_2d_manifold(x, y, color=None, x_label=None, y_label=None, colorbar_label=None, color_map='viridis', figure_size=(7,7), title=None, save_filename=None):
    """
    Plots a two-dimensional manifold given two vectors defining the manifold.

    **Example:**

    .. code:: python

        from PCAfold import PCA, plot_2d_manifold
        import numpy as np

        # Generate dummy data set:
        X = np.random.rand(100,10)

        # Obtain 2-dimensional manifold from PCA:
        pca_X = PCA(X)
        principal_components = pca_X.transform(X)

        # Plot the manifold:
        plt = plot_2d_manifold(principal_components[:,0], principal_components[:,1], color=X[:,0], x_label='PC-1', y_label='PC-2', colorbar_label='$X_1$', figure_size=(5,5), title='2D manifold', save_filename='2d-manifold.pdf')
        plt.close()

    :param x:
        ``numpy.ndarray`` specifying the variable on the :math:`x`-axis.
        It should be of size ``(n_observations,)`` or ``(n_observations,1)``.
    :param y:
        ``numpy.ndarray`` specifying the variable on the :math:`y`-axis.
        It should be of size ``(n_observations,)`` or ``(n_observations,1)``.
    :param color: (optional)
        vector or string specifying color for the manifold. If it is a
        vector, it has to have length consistent with the number of observations
        in ``x`` and ``y`` vectors. It should be of type ``numpy.ndarray`` and size
        ``(n_observations,)`` or ``(n_observations,1)``.
        It can also be set to a string specifying the color directly, for
        instance ``'r'`` or ``'#006778'``.
        If not specified, manifold will be plotted in black.
    :param x_label: (optional)
        ``str`` specifying :math:`x`-axis label annotation. If set to ``None``
        label will not be plotted.
    :param y_label: (optional)
        ``str`` specifying :math:`y`-axis label annotation. If set to ``None``
        label will not be plotted.
    :param colorbar_label: (optional)
        ``str`` specifying colorbar label annotation.
        If set to ``None``, colorbar label will not be plotted.
    :param color_map: (optional)
        ``str`` or ``matplotlib.colors.ListedColormap`` specifying the colormap to use as per ``matplotlib.cm``. Default is ``'viridis'``.
    :param figure_size: (optional)
        tuple specifying figure size.
    :param title: (optional)
        ``str`` specifying plot title. If set to ``None`` title will not be
        plotted.
    :param save_filename: (optional)
        ``str`` specifying plot save location/filename. If set to ``None``
        plot will not be saved. You can also set a desired file extension,
        for instance ``.pdf``. If the file extension is not specified, the default
        is ``.png``.

    :return:
        - **plt** - ``matplotlib.pyplot`` plot handle.
    """

    if not isinstance(x, np.ndarray):
        raise ValueError("Parameter `x` has to be of type `numpy.ndarray`.")

    try:
        (n_x,) = np.shape(x)
        n_var_x = 1
    except:
        (n_x, n_var_x) = np.shape(x)

    if n_var_x != 1:
        raise ValueError("Parameter `x` has to be a 0D or 1D vector.")

    if not isinstance(y, np.ndarray):
        raise ValueError("Parameter `y` has to be of type `numpy.ndarray`.")

    try:
        (n_y,) = np.shape(y)
        n_var_y = 1
    except:
        (n_y, n_var_y) = np.shape(y)

    if n_var_y != 1:
        raise ValueError("Parameter `y` has to be a 0D or 1D vector.")

    if n_x != n_y:
        raise ValueError("Parameter `x` has different number of elements than `y`.")

    if color is not None:
        if not isinstance(color, str):
            if not isinstance(color, np.ndarray):
                raise ValueError("Parameter `color` has to be `None`, or of type `str` or `numpy.ndarray`.")

    if isinstance(color, np.ndarray):

        try:
            (n_color,) = np.shape(color)
            n_var_color = 1
        except:
            (n_color, n_var_color) = np.shape(color)

        if n_var_color != 1:
            raise ValueError("Parameter `color` has to be a 0D or 1D vector.")

        if n_color != n_x:
            raise ValueError("Parameter `color` has different number of elements than `x` and `y`.")

    if x_label is not None:
        if not isinstance(x_label, str):
            raise ValueError("Parameter `x_label` has to be of type `str`.")

    if y_label is not None:
        if not isinstance(y_label, str):
            raise ValueError("Parameter `y_label` has to be of type `str`.")

    if not isinstance(color_map, str):
        if not isinstance(color_map, ListedColormap):
            raise ValueError("Parameter `color_map` has to be of type `str` or `matplotlib.colors.ListedColormap`.")

    if not isinstance(figure_size, tuple):
        raise ValueError("Parameter `figure_size` has to be of type `tuple`.")

    if title is not None:
        if not isinstance(title, str):
            raise ValueError("Parameter `title` has to be of type `str`.")

    if save_filename is not None:
        if not isinstance(save_filename, str):
            raise ValueError("Parameter `save_filename` has to be of type `str`.")

    fig, axs = plt.subplots(1, 1, figsize=figure_size)

    if color is None:
        scat = plt.scatter(x.ravel(), y.ravel(), c='k', marker='o', s=scatter_point_size, edgecolor='none', alpha=1)
    elif isinstance(color, str):
        scat = plt.scatter(x.ravel(), y.ravel(), c=color, cmap=color_map, marker='o', s=scatter_point_size, edgecolor='none', alpha=1)
    elif isinstance(color, np.ndarray):
        scat = plt.scatter(x.ravel(), y.ravel(), c=color.ravel(), cmap=color_map, marker='o', s=scatter_point_size, edgecolor='none', alpha=1)

    plt.xticks(fontsize=font_axes, **csfont)
    plt.yticks(fontsize=font_axes, **csfont)
    if x_label != None: plt.xlabel(x_label, fontsize=font_labels, **csfont)
    if y_label != None: plt.ylabel(y_label, fontsize=font_labels, **csfont)
    plt.grid(alpha=grid_opacity)

    if isinstance(color, np.ndarray):
        if color is not None:
            cb = fig.colorbar(scat)
            cb.ax.tick_params(labelsize=font_colorbar_axes)
            if colorbar_label != None: cb.set_label(colorbar_label, fontsize=font_colorbar, rotation=0, horizontalalignment='left')

    if title != None: plt.title(title, fontsize=font_title, **csfont)
    if save_filename != None: plt.savefig(save_filename, dpi=save_dpi, bbox_inches='tight')

    return plt

# ------------------------------------------------------------------------------

def plot_3d_manifold(x, y, z, color=None, elev=45, azim=-45, x_label=None, y_label=None, z_label=None, colorbar_label=None, color_map='viridis', figure_size=(7,7), title=None, save_filename=None):
    """
    Plots a three-dimensional manifold given three vectors defining the manifold.

    **Example:**

    .. code:: python

        from PCAfold import PCA, plot_3d_manifold
        import numpy as np

        # Generate dummy data set:
        X = np.random.rand(100,10)

        # Obtain 3-dimensional manifold from PCA:
        pca_X = PCA(X)
        PCs = pca_X.transform(X)

        # Plot the manifold:
        plt = plot_3d_manifold(PCs[:,0], PCs[:,1], PCs[:,2], color=X[:,0], elev=30, azim=-60, x_label='PC-1', y_label='PC-2', z_label='PC-3', colorbar_label='$X_1$', figure_size=(15,7), title='3D manifold', save_filename='3d-manifold.png')
        plt.close()

    :param x:
        variable on the :math:`x`-axis. It should be of type ``numpy.ndarray`` and size
        ``(n_observations,)`` or ``(n_observations,1)``.
    :param y:
        variable on the :math:`y`-axis. It should be of type ``numpy.ndarray`` and size
        ``(n_observations,)`` or ``(n_observations,1)``.
    :param z:
        variable on the :math:`z`-axis. It should be of type ``numpy.ndarray`` and size
        ``(n_observations,)`` or ``(n_observations,1)``.
    :param color: (optional)
        vector or string specifying color for the manifold. If it is a
        vector, it has to have length consistent with the number of observations
        in ``x``, ``y`` and ``z`` vectors. It should be of type ``numpy.ndarray`` and size
        ``(n_observations,)`` or ``(n_observations,1)``.
        It can also be set to a string specifying the color directly, for
        instance ``'r'`` or ``'#006778'``.
        If not specified, manifold will be plotted in black.
    :param elev: (optional)
        elevation angle.
    :param azim: (optional)
        azimuth angle.
    :param x_label: (optional)
        ``str`` specifying :math:`x`-axis label annotation. If set to ``None``
        label will not be plotted.
    :param y_label: (optional)
        ``str`` specifying :math:`y`-axis label annotation. If set to ``None``
        label will not be plotted.
    :param z_label: (optional)
        ``str`` specifying :math:`z`-axis label annotation. If set to ``None``
        label will not be plotted.
    :param colorbar_label: (optional)
        ``str`` specifying colorbar label annotation.
        If set to ``None``, colorbar label will not be plotted.
    :param color_map: (optional)
        ``str`` or ``matplotlib.colors.ListedColormap`` specifying the colormap to use as per ``matplotlib.cm``. Default is ``'viridis'``.
    :param figure_size: (optional)
        ``tuple`` specifying figure size.
    :param title: (optional)
        ``str`` specifying plot title. If set to ``None`` title will not be
        plotted.
    :param save_filename: (optional)
        ``str`` specifying plot save location/filename. If set to ``None``
        plot will not be saved. You can also set a desired file extension,
        for instance ``.pdf``. If the file extension is not specified, the default
        is ``.png``.

    :return:
        - **plt** - ``matplotlib.pyplot`` plot handle.
    """

    from mpl_toolkits.mplot3d import Axes3D

    if not isinstance(x, np.ndarray):
        raise ValueError("Parameter `x` has to be of type `numpy.ndarray`.")

    try:
        (n_x,) = np.shape(x)
        n_var_x = 1
    except:
        (n_x, n_var_x) = np.shape(x)

    if n_var_x != 1:
        raise ValueError("Parameter `x` has to be a 0D or 1D vector.")

    if not isinstance(y, np.ndarray):
        raise ValueError("Parameter `y` has to be of type `numpy.ndarray`.")

    try:
        (n_y,) = np.shape(y)
        n_var_y = 1
    except:
        (n_y, n_var_y) = np.shape(y)

    if n_var_y != 1:
        raise ValueError("Parameter `y` has to be a 0D or 1D vector.")

    if not isinstance(z, np.ndarray):
        raise ValueError("Parameter `z` has to be of type `numpy.ndarray`.")

    try:
        (n_z,) = np.shape(z)
        n_var_z = 1
    except:
        (n_z, n_var_z) = np.shape(z)

    if n_var_z != 1:
        raise ValueError("Parameter `z` has to be a 0D or 1D vector.")

    if n_x != n_y or n_y != n_z:
        raise ValueError("Parameters `x`, `y` and `z` have to have the same number of elements.")

    if color is not None:
        if not isinstance(color, str):
            if not isinstance(color, np.ndarray):
                raise ValueError("Parameter `color` has to be `None`, or of type `str` or `numpy.ndarray`.")

    if isinstance(color, np.ndarray):

        try:
            (n_color,) = np.shape(color)
            n_var_color = 1
        except:
            (n_color, n_var_color) = np.shape(color)

        if n_var_color != 1:
            raise ValueError("Parameter `color` has to be a 0D or 1D vector.")

        if n_color != n_x:
            raise ValueError("Parameter `color` has different number of elements than `x` and `y`.")

    if x_label is not None:
        if not isinstance(x_label, str):
            raise ValueError("Parameter `x_label` has to be of type `str`.")

    if y_label is not None:
        if not isinstance(y_label, str):
            raise ValueError("Parameter `y_label` has to be of type `str`.")

    if z_label is not None:
        if not isinstance(z_label, str):
            raise ValueError("Parameter `z_label` has to be of type `str`.")

    if not isinstance(color_map, str):
        if not isinstance(color_map, ListedColormap):
            raise ValueError("Parameter `color_map` has to be of type `str` or `matplotlib.colors.ListedColormap`.")

    if not isinstance(figure_size, tuple):
        raise ValueError("Parameter `figure_size` has to be of type `tuple`.")

    if title is not None:
        if not isinstance(title, str):
            raise ValueError("Parameter `title` has to be of type `str`.")

    if save_filename is not None:
        if not isinstance(save_filename, str):
            raise ValueError("Parameter `save_filename` has to be of type `str`.")

    fig = plt.figure(figsize=figure_size)
    ax = fig.add_subplot(111, projection='3d')

    if color is None:
        scat = ax.scatter(x.ravel(), y.ravel(), z.ravel(), c='k', marker='o', s=scatter_point_size, alpha=1)
    elif isinstance(color, str):
        scat = ax.scatter(x.ravel(), y.ravel(), z.ravel(), c=color, cmap=color_map, marker='o', s=scatter_point_size, alpha=1)
    elif isinstance(color, np.ndarray):
        scat = ax.scatter(x.ravel(), y.ravel(), z.ravel(), c=color.ravel(), cmap=color_map, marker='o', s=scatter_point_size, alpha=1)

    if x_label != None: ax.set_xlabel(x_label, **csfont, fontsize=font_labels, rotation=0, labelpad=20)
    if y_label != None: ax.set_ylabel(y_label, **csfont, fontsize=font_labels, rotation=0, labelpad=20)
    if z_label != None: ax.set_zlabel(z_label, **csfont, fontsize=font_labels, rotation=0, labelpad=20)

    ax.tick_params(pad=5)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('w')
    ax.yaxis.pane.set_edgecolor('w')
    ax.zaxis.pane.set_edgecolor('w')
    ax.view_init(elev=elev, azim=azim)
    ax.grid(alpha=grid_opacity)

    for label in (ax.get_xticklabels()):
        label.set_fontsize(font_axes)
    for label in (ax.get_yticklabels()):
        label.set_fontsize(font_axes)
    for label in (ax.get_zticklabels()):
        label.set_fontsize(font_axes)

    if isinstance(color, np.ndarray):
        if color is not None:
            cb = fig.colorbar(scat, shrink=0.6)
            cb.ax.tick_params(labelsize=font_colorbar_axes)
            if colorbar_label != None: cb.set_label(colorbar_label, fontsize=font_colorbar, rotation=0, horizontalalignment='left')

    if title != None: ax.set_title(title, **csfont, fontsize=font_title)
    if save_filename != None: plt.savefig(save_filename, dpi=save_dpi, bbox_inches='tight')

    return plt

# ------------------------------------------------------------------------------

def plot_2d_manifold_sequence(xy, color=None, x_label=None, y_label=None, cbar=False, colorbar_label=None, color_map='viridis', figure_size=(7,3), title=None, save_filename=None):
    """
    Plots a sequence of two-dimensional manifolds given a list of two vectors defining the manifold.

    **Example:**

    .. code:: python

        from PCAfold import SubsetPCA, plot_2d_manifold_sequence
        import numpy as np

        # Generate dummy data set:
        X = np.random.rand(100,5)

        # Obtain two-dimensional manifolds from subset PCA:
        subset_PCA = SubsetPCA(X)
        principal_components = subset_PCA.principal_components

        # Plot the manifold:
        plt = plot_2d_manifold_sequence(principal_components, color=X[:,0], x_label='PC-1', y_label='PC-2', colorbar_label='$X_1$', figure_size=(7,3), title=['First', 'Second', 'Third'], save_filename='2d-manifold-sequence.pdf')
        plt.close()

    :param xy:
        ``list`` of ``numpy.ndarray`` specifying the manifold (variables on the :math:`x` and :math:`y` -axis).
        Each element of the list should be of size ``(n_observations,2)``.
    :param color: (optional)
        ``numpy.ndarray`` or ``str``, or ``list`` of ``numpy.ndarray`` or ``str`` specifying colors for the manifolds. If it is a
        vector, it has to have length consistent with the number of observations
        in ``x`` and ``y`` vectors. Each ``numpy.ndarray`` should be of size
        ``(n_observations,)`` or ``(n_observations,1)``.
        It can also be set to a string specifying the color directly, for
        instance ``'r'`` or ``'#006778'``.
        If not specified, manifolds will be plotted in black.
    :param x_label: (optional)
        ``str`` specifying :math:`x`-axis label annotation. If set to ``None``
        label will not be plotted.
    :param y_label: (optional)
        ``str`` specifying :math:`y`-axis label annotation. If set to ``None``
        label will not be plotted.
    :param cbar: (optional)
        ``bool`` specifying if the colorbar should be plotted.
    :param colorbar_label: (optional)
        ``str`` specifying colorbar label annotation.
        If set to ``None``, colorbar label will not be plotted.
    :param color_map: (optional)
        ``str`` or ``matplotlib.colors.ListedColormap`` specifying the colormap to use as per ``matplotlib.cm``. Default is ``'viridis'``.
    :param figure_size: (optional)
        tuple specifying figure size.
    :param title: (optional)
        ``list`` of ``str`` specifying title for each subplot. If set to ``None`` titles will not be
        plotted.
    :param save_filename: (optional)
        ``str`` specifying plot save location/filename. If set to ``None``
        plot will not be saved. You can also set a desired file extension,
        for instance ``.pdf``. If the file extension is not specified, the default
        is ``.png``.

    :return:
        - **plt** - ``matplotlib.pyplot`` plot handle.
    """

    if not isinstance(xy, list):
        raise ValueError("Parameter `xy` has to be of type `list`.")

    n_subplots = len(xy)

    for i, element in enumerate(xy):
        if not isinstance(element, np.ndarray):
            raise ValueError("Parameter `xy` has to have elements of type `numpy.ndarray`.")

        (n_observations, n_variables) = np.shape(element)

        if n_variables != 2:
            raise ValueError("Parameter `xy` has to have elements of shape `(n_observations,2)`.")

        if i > 0:
            if n_observations != prev_n_observations or n_variables != prev_n_variables:
                raise ValueError("Parameter `xy` has to have elements of the same shapes, `(n_observations,2)`.")

        prev_n_observations = n_observations
        prev_n_variables = n_variables

    if color is not None:
        if (not isinstance(color, str)) and (not isinstance(color, np.ndarray)) and (not isinstance(color, list)):
                raise ValueError("Parameter `color` has to be `None`, or of type `str` or `numpy.ndarray` or `list`.")

    if isinstance(color, np.ndarray):

        try:
            (n_color,) = np.shape(color)
            n_var_color = 1
        except:
            (n_color, n_var_color) = np.shape(color)

        if n_var_color != 1:
            raise ValueError("Parameter `color` has to be a 0D or 1D vector.")

        if n_color != n_observations:
            raise ValueError("Parameter `color` has different number of observations than `xy`.")

    if isinstance(color, list):
        if len(color) != n_subplots:
            raise ValueError("Parameter `color` has different number of elements than `xy`.")

    if not isinstance(cbar, bool):
        raise ValueError("Parameter `cbar` has to be of type `bool`.")

    if colorbar_label is not None:
        if not isinstance(colorbar_label, str):
            raise ValueError("Parameter `colorbar_label` has to be of type `str`.")

    if x_label is not None:
        if not isinstance(x_label, str):
            raise ValueError("Parameter `x_label` has to be of type `str`.")

    if y_label is not None:
        if not isinstance(y_label, str):
            raise ValueError("Parameter `y_label` has to be of type `str`.")

    if not isinstance(color_map, str):
        if not isinstance(color_map, ListedColormap):
            raise ValueError("Parameter `color_map` has to be of type `str` or `matplotlib.colors.ListedColormap`.")

    if not isinstance(figure_size, tuple):
        raise ValueError("Parameter `figure_size` has to be of type `tuple`.")

    if title is not None:
        if not isinstance(title, list):
            raise ValueError("Parameter `title` has to be of type `list`.")
        if len(title) != n_subplots:
            raise ValueError("Parameter `title` has to have the same number of elements as parameter `xy`.")

    if save_filename is not None:
        if not isinstance(save_filename, str):
            raise ValueError("Parameter `save_filename` has to be of type `str`.")

    fig = plt.figure(figsize=figure_size)
    spec = fig.add_gridspec(ncols=n_subplots, nrows=1)

    for i, element in enumerate(xy):

        ax = fig.add_subplot(spec[0,i])

        if color is None:
            scat = ax.scatter(element[:,0], element[:,1], c='k', marker='o', s=scatter_point_size, edgecolor='none', alpha=1)
        elif isinstance(color, str):
            scat = ax.scatter(element[:,0], element[:,1], c=color, cmap=color_map, marker='o', s=scatter_point_size, edgecolor='none', alpha=1)
        elif isinstance(color, np.ndarray):
            scat = ax.scatter(element[:,0], element[:,1], c=color.ravel(), cmap=color_map, marker='o', s=scatter_point_size, edgecolor='none', alpha=1)
        elif isinstance(color, list):
            if isinstance(color[i], np.ndarray):
                scat = ax.scatter(element[:,0], element[:,1], c=color[i].ravel(), cmap=color_map, marker='o', s=scatter_point_size, edgecolor='none', alpha=1)
            elif isinstance(color[i], str):
                scat = ax.scatter(element[:,0], element[:,1], c=color[i], cmap=color_map, marker='o', s=scatter_point_size, edgecolor='none', alpha=1)
            else:
                raise ValueError("Parameter `color` should have elements of type `numpy.ndarray` or `str`.")

        if title != None: ax.set_title(title[i], fontsize=font_title, **csfont)
        ax.set_xticklabels([], fontdict=None, minor=False)
        ax.set_yticklabels([], fontdict=None, minor=False)
        ax.set_xticks([])
        ax.set_yticks([])

        if x_label != None: plt.xlabel(x_label, fontsize=font_labels, **csfont)

        if i == 0:
            if y_label != None: plt.ylabel(y_label, fontsize=font_labels, **csfont)

    if cbar:
        if isinstance(color, np.ndarray):
            if color is not None:
                cb = fig.colorbar(scat)
                cb.ax.tick_params(labelsize=font_colorbar_axes)
                if colorbar_label != None: cb.set_label(colorbar_label, fontsize=font_colorbar, rotation=0, horizontalalignment='left')

    fig.tight_layout(pad=0)

    if save_filename != None: plt.savefig(save_filename, dpi=save_dpi, bbox_inches='tight')

    return plt

# ------------------------------------------------------------------------------

def plot_parity(variable, variable_rec, color=None, x_label=None, y_label=None, colorbar_label=None, color_map='viridis', figure_size=(7,7), title=None, save_filename=None):
    """
    Plots a parity plot between a variable and its reconstruction.

    **Example:**

    .. code:: python

        from PCAfold import PCA, plot_parity
        import numpy as np

        # Generate dummy data set:
        X = np.random.rand(100,10)

        # Obtain PCA reconstruction of the data set:
        pca_X = PCA(X, n_components=8)
        principal_components = pca_X.transform(X)
        X_rec = pca_X.reconstruct(principal_components)

        # Parity plot for the reconstruction of the first variable:
        plt = plot_parity(X[:,0], X_rec[:,0], color=X[:,0], x_label='Observed $X_1$', y_label='Reconstructed $X_1$', colorbar_label='X_1', color_map='inferno', figure_size=(5,5), title='Parity plot', save_filename='parity-plot.pdf')
        plt.close()

    :param variable:
        vector specifying the original variable. It should be of type ``numpy.ndarray`` and size
        ``(n_observations,)`` or ``(n_observations,1)``.
    :param variable_rec:
        vector specifying the reconstruction of the original variable. It should be of type ``numpy.ndarray`` and size
        ``(n_observations,)`` or ``(n_observations,1)``.
    :param color: (optional)
        vector or string specifying color for the parity plot. If it is a
        vector, it has to have length consistent with the number of observations
        in ``x`` and ``y`` vectors. It should be of type ``numpy.ndarray`` and size
        ``(n_observations,)`` or ``(n_observations,1)``.
        It can also be set to a string specifying the color directly, for
        instance ``'r'`` or ``'#006778'``.
        If not specified, parity plot will be plotted in black.
    :param x_label: (optional)
        ``str`` specifying :math:`x`-axis label annotation. If set to ``None``
        label will not be plotted.
    :param y_label: (optional)
        ``str`` specifying :math:`y`-axis label annotation. If set to ``None``
        label will not be plotted.
    :param colorbar_label: (optional)
        ``str`` specifying colorbar label annotation.
        If set to ``None``, colorbar label will not be plotted.
    :param color_map: (optional)
        ``str`` or ``matplotlib.colors.ListedColormap`` specifying the colormap to use as per ``matplotlib.cm``. Default is ``'viridis'``.
    :param figure_size: (optional)
        tuple specifying figure size.
    :param title: (optional)
        ``str`` specifying plot title. If set to ``None`` title will not be
        plotted.
    :param save_filename: (optional)
        ``str`` specifying plot save location/filename. If set to ``None``
        plot will not be saved. You can also set a desired file extension,
        for instance ``.pdf``. If the file extension is not specified, the default
        is ``.png``.

    :return:
        - **plt** - ``matplotlib.pyplot`` plot handle.
    """

    if not isinstance(variable, np.ndarray):
        raise ValueError("Parameter `variable` has to be of type `numpy.ndarray`.")

    try:
        (n_x,) = np.shape(variable)
        n_var_x = 1
    except:
        (n_x, n_var_x) = np.shape(variable)

    if n_var_x != 1:
        raise ValueError("Parameter `variable` has to be a 0D or 1D vector.")

    if not isinstance(variable_rec, np.ndarray):
        raise ValueError("Parameter `variable_rec` has to be of type `numpy.ndarray`.")

    try:
        (n_y,) = np.shape(variable_rec)
        n_var_y = 1
    except:
        (n_y, n_var_y) = np.shape(variable_rec)

    if n_var_y != 1:
        raise ValueError("Parameter `variable_rec` has to be a 0D or 1D vector.")

    if n_x != n_y:
        raise ValueError("Parameter `variable` has different number of elements than `variable_rec`.")

    if color is not None:
        if not isinstance(color, str):
            if not isinstance(color, np.ndarray):
                raise ValueError("Parameter `color` has to be `None`, or of type `str` or `numpy.ndarray`.")

    if isinstance(color, np.ndarray):

        try:
            (n_color,) = np.shape(color)
            n_var_color = 1
        except:
            (n_color, n_var_color) = np.shape(color)

        if n_var_color != 1:
            raise ValueError("Parameter `color` has to be a 0D or 1D vector.")

        if n_color != n_x:
            raise ValueError("Parameter `color` has different number of elements than `variable` and `variable_rec`.")

    if x_label is not None:
        if not isinstance(x_label, str):
            raise ValueError("Parameter `x_label` has to be of type `str`.")

    if y_label is not None:
        if not isinstance(y_label, str):
            raise ValueError("Parameter `y_label` has to be of type `str`.")

    if not isinstance(color_map, str):
        if not isinstance(color_map, ListedColormap):
            raise ValueError("Parameter `color_map` has to be of type `str` or `matplotlib.colors.ListedColormap`.")

    if not isinstance(figure_size, tuple):
        raise ValueError("Parameter `figure_size` has to be of type `tuple`.")

    if title is not None:
        if not isinstance(title, str):
            raise ValueError("Parameter `title` has to be of type `str`.")

    if save_filename is not None:
        if not isinstance(save_filename, str):
            raise ValueError("Parameter `save_filename` has to be of type `str`.")

    color_line = '#ff2f18'

    fig, axs = plt.subplots(1, 1, figsize=figure_size)

    if color is None:
        scat = plt.scatter(variable.ravel(), variable_rec.ravel(), c='k', marker='o', s=scatter_point_size, edgecolor='none', alpha=1)
    elif isinstance(color, str):
        scat = plt.scatter(variable.ravel(), variable_rec.ravel(), c=color, cmap=color_map, marker='o', s=scatter_point_size, edgecolor='none', alpha=1)
    elif isinstance(color, np.ndarray):
        scat = plt.scatter(variable.ravel(), variable_rec.ravel(), c=color.ravel(), cmap=color_map, marker='o', s=scatter_point_size, edgecolor='none', alpha=1)


    plot_offset = 0.05 * (np.max(variable) - np.min(variable))

    line = plt.plot([np.min(variable)-plot_offset, np.max(variable)+plot_offset], [np.min(variable)-plot_offset, np.max(variable)+plot_offset], c=color_line)
    plt.xlim([np.min(variable)-plot_offset, np.max(variable)+plot_offset])
    plt.ylim([np.min(variable)-plot_offset, np.max(variable)+plot_offset])
    plt.xticks(fontsize=font_axes, **csfont)
    plt.yticks(fontsize=font_axes, **csfont)
    if x_label != None: plt.xlabel(x_label, fontsize=font_labels, **csfont)
    if y_label != None: plt.ylabel(y_label, fontsize=font_labels, **csfont)
    plt.grid(alpha=grid_opacity)

    if not isinstance(color, str):
        if color is not None:
            cb = fig.colorbar(scat)
            cb.ax.tick_params(labelsize=font_colorbar_axes)
            if colorbar_label != None: cb.set_label(colorbar_label, fontsize=font_colorbar, rotation=0, horizontalalignment='left')

    if title != None: plt.title(title, fontsize=font_title, **csfont)
    if save_filename != None: plt.savefig(save_filename, dpi=save_dpi, bbox_inches='tight')

    return plt

# ------------------------------------------------------------------------------

def plot_eigenvectors(eigenvectors, eigenvectors_indices=[], variable_names=[], plot_absolute=False, bar_color=None, figure_size=None, title=None, save_filename=None):
    """
    Plots weights on eigenvectors. It will generate as many
    plots as there are eigenvectors present in the ``eigenvectors`` matrix.

    **Example:**

    .. code:: python

        from PCAfold import PCA, plot_eigenvectors
        import numpy as np

        # Generate dummy data set:
        X = np.random.rand(100,3)

        # Perform PCA and obtain eigenvectors:
        pca_X = PCA(X, n_components=2)
        eigenvectors = pca_X.A

        # Plot second and third eigenvector:
        plts = plot_eigenvectors(eigenvectors[:,[1,2]], eigenvectors_indices=[1,2], variable_names=['$a_1$', '$a_2$', '$a_3$'], plot_absolute=False, bar_color=None, title='PCA on X', save_filename='PCA-X.pdf')
        plts[0].close()
        plts[1].close()

    :param eigenvectors:
        matrix of eigenvectors to plot. It can be supplied as an attribute of
        the ``PCA`` class: ``PCA.A``.
    :param eigenvectors_indices:
        ``list`` of ``int`` specifying indexing of eigenvectors inside
        ``eigenvectors`` supplied. If it is not supplied, it is assumed that
        eigenvectors are numbered :math:`[0, 1, 2, \\dots, n]`, where :math:`n`
        is the number of eigenvectors provided.
    :param variable_names: (optional)
        ``list`` of ``str`` specifying variable names.
    :param plot_absolute:
        ``bool`` specifying whether absolute values of eigenvectors should be plotted.
    :param bar_color: (optional)
        ``str`` specifying color of bars.
    :param figure_size: (optional)
        tuple specifying figure size.
    :param title: (optional)
        ``str`` specifying plot title. If set to ``None`` title will not be
        plotted.
    :param save_filename: (optional)
        ``str`` specifying plot save location/filename. If set to ``None``
        plot will not be saved. You can also set a desired file extension,
        for instance ``.pdf``. If the file extension is not specified, the default
        is ``.png``. Note that a prefix ``eigenvector-#`` will be added out front
        the filename, where ``#`` is the number of the currently plotted eigenvector.

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

        if figure_size is None:
            fig, ax = plt.subplots(figsize=(n_variables, 4))
        else:
            fig, ax = plt.subplots(figsize=figure_size)

        if plot_absolute:
            plt.bar(x_range, abs(eigenvectors[:,n_pc]), width=eigenvector_bar_width, color=bar_color, edgecolor=bar_color, align='center', zorder=2)
        else:
            plt.bar(x_range, eigenvectors[:,n_pc], width=eigenvector_bar_width, color=bar_color, edgecolor=bar_color, align='center', zorder=2)

        plt.xticks(x_range, variable_names, fontsize=font_axes, **csfont)
        if plot_absolute:
            plt.ylabel('PC-' + str(eigenvectors_indices[n_pc] + 1) + ' absolute weight [-]', fontsize=font_labels, **csfont)
        else:
            plt.ylabel('PC-' + str(eigenvectors_indices[n_pc] + 1) + ' weight [-]', fontsize=font_labels, **csfont)

        plt.grid(alpha=grid_opacity, zorder=0)
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
            save_filename_list = save_filename.split('/')
            if len(save_filename_list) == 1:
                plt.savefig('eigenvector-' + str(eigenvectors_indices[n_pc] + 1) + '-' + save_filename_list[-1], dpi=save_dpi, bbox_inches='tight')
            elif len(save_filename_list) > 1:
                plt.savefig('/'.join(save_filename_list[0:-1]) + '/eigenvector-' + str(eigenvectors_indices[n_pc] + 1) + '-' + save_filename_list[-1], dpi=save_dpi, bbox_inches='tight')

        plot_handles.append(plt)

    return plot_handles

# ------------------------------------------------------------------------------

def plot_eigenvectors_comparison(eigenvectors_tuple, legend_labels=[], variable_names=[], plot_absolute=False, color_map='coolwarm', figure_size=None, title=None, save_filename=None):
    """
    Plots a comparison of weights on eigenvectors.

    **Example:**

    .. code:: python

        from PCAfold import PCA, plot_eigenvectors_comparison
        import numpy as np

        # Generate dummy data set:
        X = np.random.rand(100,3)

        # Perform PCA and obtain eigenvectors:
        pca_X = PCA(X, n_components=2)
        eigenvectors = pca_X.A

        # Plot comparaison of first and second eigenvector:
        plt = plot_eigenvectors_comparison((eigenvectors[:,0], eigenvectors[:,1]), legend_labels=['PC-1', 'PC-2'], variable_names=['$a_1$', '$a_2$', '$a_3$'], plot_absolute=False, color_map='coolwarm', title='PCA on X', save_filename='PCA-X.pdf')
        plt.close()

    :param eigenvectors_tuple:
        ``tuple`` specifying the eigenvectors to plot. Each eigenvector inside a tuple should be a 0D array.
        It can be supplied as an attribute of the ``PCA`` class, for instance: ``(PCA.A[:,0], PCA.A[:,1])``.
    :param legend_labels:
        ``list`` of ``str`` specifying labels for each element in the ``eigenvectors_tuple``.
    :param variable_names: (optional)
        ``list`` of ``str`` specifying variable names.
    :param plot_absolute:
        ``bool`` specifying whether absolute values of eigenvectors should be plotted.
    :param color_map: (optional)
        ``str`` or ``matplotlib.colors.ListedColormap`` specifying the colormap to use as per ``matplotlib.cm``. Default is ``'coolwarm'``.
    :param figure_size: (optional)
        tuple specifying figure size.
    :param title: (optional)
        ``str`` specifying plot title. If set to ``None`` title will not be
        plotted.
    :param save_filename: (optional)
        ``str`` specifying plot save location/filename. If set to ``None``
        plot will not be saved. You can also set a desired file extension,
        for instance ``.pdf``. If the file extension is not specified, the default
        is ``.png``.

    :return:
        - **plt** - ``matplotlib.pyplot`` plot handle.
    """

    from matplotlib import cm

    for i in range(0, len(eigenvectors_tuple)):
        try:
            (n_variables, n_components) = np.shape(eigenvectors_tuple[i])
            raise ValueError("Eigenvectors inside eigenvectors_tuple has to be 0D arrays.")
        except:
            pass

    if not isinstance(color_map, str):
        if not isinstance(color_map, ListedColormap):
            raise ValueError("Parameter `color_map` has to be of type `str` or `matplotlib.colors.ListedColormap`.")

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

    if figure_size is None:
        fig, ax = plt.subplots(figsize=(n_variables, 4))
    else:
        fig, ax = plt.subplots(figsize=figure_size)

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

    plt.grid(alpha=grid_opacity, zorder=0)
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
        plt.savefig(save_filename, dpi=save_dpi, bbox_inches='tight')

    return plt

# ------------------------------------------------------------------------------

def plot_eigenvalue_distribution(eigenvalues, normalized=False, figure_size=None, title=None, save_filename=None):
    """
    Plots eigenvalue distribution.

    **Example:**

    .. code:: python

        from PCAfold import PCA, plot_eigenvalue_distribution
        import numpy as np

        # Generate dummy data set:
        X = np.random.rand(100,5)

        # Perform PCA and obtain eigenvalues:
        pca_X = PCA(X)
        eigenvalues = pca_X.L

        # Plot eigenvalue distribution:
        plt = plot_eigenvalue_distribution(eigenvalues, normalized=True, title='PCA on X', save_filename='PCA-X.pdf')
        plt.close()

    :param eigenvalues:
        a 0D vector of eigenvalues to plot. It can be supplied as an attribute of the
        ``PCA`` class: ``PCA.L``.
    :param normalized: (optional)
        ``bool`` specifying whether eigenvalues should be normalized to 1.
    :param figure_size: (optional)
        tuple specifying figure size.
    :param title: (optional)
        ``str`` specifying plot title. If set to ``None`` title will not be
        plotted.
    :param save_filename: (optional)
        ``str`` specifying plot save location/filename. If set to ``None``
        plot will not be saved. You can also set a desired file extension,
        for instance ``.pdf``. If the file extension is not specified, the default
        is ``.png``.

    :return:
        - **plt** - ``matplotlib.pyplot`` plot handle.
    """

    color_plot = '#191b27'

    (n_eigenvalues, ) = np.shape(eigenvalues)
    x_range = np.arange(1, n_eigenvalues+1)

    if figure_size is None:
        fig, ax = plt.subplots(figsize=(n_eigenvalues, 4))
    else:
        fig, ax = plt.subplots(figsize=figure_size)

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
    plt.grid(alpha=grid_opacity, zorder=0)

    ax.spines["top"].set_visible(True)
    ax.spines["bottom"].set_visible(True)
    ax.spines["right"].set_visible(True)
    ax.spines["left"].set_visible(True)

    if title != None:
        plt.title(title, fontsize=font_title, **csfont)

    if save_filename != None:
        plt.savefig(save_filename, dpi=save_dpi, bbox_inches='tight')

    return plt

# ------------------------------------------------------------------------------

def plot_eigenvalue_distribution_comparison(eigenvalues_tuple, legend_labels=[], normalized=False, color_map='coolwarm', figure_size=None, title=None, save_filename=None):
    """
    Plots a comparison of eigenvalue distributions.

    **Example:**

    .. code:: python

        from PCAfold import PCA, plot_eigenvalue_distribution_comparison
        import numpy as np

        # Generate dummy data sets:
        X = np.random.rand(100,10)
        Y = np.random.rand(100,10)

        # Perform PCA and obtain eigenvalues:
        pca_X = PCA(X)
        eigenvalues_X = pca_X.L
        pca_Y = PCA(Y)
        eigenvalues_Y = pca_Y.L

        # Plot eigenvalue distribution comparison:
        plt = plot_eigenvalue_distribution_comparison((eigenvalues_X, eigenvalues_Y), legend_labels=['PCA on X', 'PCA on Y'], normalized=True, title='PCA on X and Y', save_filename='PCA-X-Y.pdf')
        plt.close()

    :param eigenvalues_tuple:
        ``tuple`` specifying the eigenvalues to plot. Each vector of eigenvalues inside a tuple
        should be a 0D array. It can be supplied as an attribute of the
        ``PCA`` class, for instance: ``(PCA_1.L, PCA_2.L)``.
    :param legend_labels:
        ``list`` of ``str`` specifying the labels for each element in the ``eigenvalues_tuple``.
    :param normalized: (optional)
        ``bool`` specifying whether eigenvalues should be normalized to 1.
    :param color_map: (optional)
        ``str`` or ``matplotlib.colors.ListedColormap`` specifying the colormap to use as per ``matplotlib.cm``. Default is ``'coolwarm'``.
    :param figure_size: (optional)
        tuple specifying figure size.
    :param title: (optional)
        ``str`` specifying plot title. If set to ``None`` title will not be
        plotted.
    :param save_filename: (optional)
        ``str`` specifying plot save location/filename. If set to ``None``
        plot will not be saved. You can also set a desired file extension,
        for instance ``.pdf``. If the file extension is not specified, the default
        is ``.png``.

    :return:
        - **plt** - ``matplotlib.pyplot`` plot handle.
    """

    from matplotlib import cm

    if not isinstance(color_map, str):
        if not isinstance(color_map, ListedColormap):
            raise ValueError("Parameter `color_map` has to be of type `str` or `matplotlib.colors.ListedColormap`.")

    n_sets = len(eigenvalues_tuple)

    (n_eigenvalues, ) = np.shape(eigenvalues_tuple[0])
    x_range = np.arange(1, n_eigenvalues+1)

    color_map_colors = cm.get_cmap(color_map, n_sets)
    sets_colors = color_map_colors(np.linspace(0, 1, n_sets))

    # Create default labels for legend:
    if len(legend_labels) == 0:
        legend_labels = ['Set ' + str(i) + '' for i in range(1, n_sets+1)]

    if figure_size is None:
        fig, ax = plt.subplots(figsize=(n_eigenvalues, 4))
    else:
        fig, ax = plt.subplots(figsize=figure_size)

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
    plt.grid(alpha=grid_opacity, zorder=0)

    ax.spines["top"].set_visible(True)
    ax.spines["bottom"].set_visible(True)
    ax.spines["right"].set_visible(True)
    ax.spines["left"].set_visible(True)

    plt.legend(loc='upper right', fancybox=True, shadow=True, fontsize=font_legend, markerscale=marker_scale_legend)

    if title != None:
        plt.title(title, fontsize=font_title, **csfont)

    if save_filename != None:
        plt.savefig(save_filename, dpi=save_dpi, bbox_inches='tight')

    return plt

# ------------------------------------------------------------------------------

def plot_cumulative_variance(eigenvalues, n_components=0, figure_size=None, title=None, save_filename=None):
    """
    Plots the eigenvalues as bars and their cumulative sum to visualize
    the percent variance in the data explained by each principal component
    individually and by each principal component cumulatively.

    **Example:**

    .. code:: python

        from PCAfold import PCA, plot_cumulative_variance
        import numpy as np

        # Generate dummy data set:
        X = np.random.rand(100,5)

        # Perform PCA and obtain eigenvalues:
        pca_X = PCA(X)
        eigenvalues = pca_X.L

        # Plot the cumulative variance from eigenvalues:
        plt = plot_cumulative_variance(eigenvalues, n_components=0, title='PCA on X', save_filename='PCA-X.pdf')
        plt.close()

    :param eigenvalues:
        a 0D vector of eigenvalues to analyze. It can be supplied as an attribute of the
        ``PCA`` class: ``PCA.L``.
    :param n_components: (optional)
        how many principal components you want to visualize (default is all).
    :param figure_size: (optional)
        tuple specifying figure size.
    :param title: (optional)
        ``str`` specifying plot title. If set to ``None`` title will not be
        plotted.
    :param save_filename: (optional)
        ``str`` specifying plot save location/filename. If set to ``None``
        plot will not be saved. You can also set a desired file extension,
        for instance ``.pdf``. If the file extension is not specified, the default
        is ``.png``.

    :return:
        - **plt** - ``matplotlib.pyplot`` plot handle.
    """

    bar_color = '#191b27'
    line_color = '#ff2f18'

    (n_eigenvalues, ) = np.shape(eigenvalues)

    if n_components == 0:
        n_retained = n_eigenvalues
    else:
        n_retained = n_components

    x_range = np.arange(1, n_retained+1)

    if figure_size is None:
        fig, ax1 = plt.subplots(figsize=(n_retained, 4))
    else:
        fig, ax = plt.subplots(figsize=figure_size)

    ax1.bar(x_range, eigenvalues[0:n_retained], color=bar_color, edgecolor=bar_color, align='center', zorder=2, label='Eigenvalue')
    ax1.set_ylabel('Eigenvalue [-]', fontsize=font_labels, **csfont)
    ax1.set_ylim(0,1.05)
    ax1.grid(alpha=grid_opacity, zorder=0)
    ax1.set_xlabel('$q$ [-]', fontsize=font_labels, **csfont)

    ax2 = ax1.twinx()
    ax2.plot(x_range, np.cumsum(eigenvalues[0:n_retained])*100, 'o-', color=line_color, zorder=2, label='Cumulative')
    ax2.set_ylabel('Variance explained [%]', color=line_color, fontsize=font_labels, **csfont)
    ax2.set_ylim(0,105)
    ax2.tick_params('y', colors=line_color)

    plt.xlim(0, n_retained+1)
    plt.xticks(x_range, fontsize=font_axes, **csfont)

    if title != None:
        plt.title(title, fontsize=font_title, **csfont)

    if save_filename != None:
        plt.savefig(save_filename, dpi=save_dpi, bbox_inches='tight')

    return plt

# ------------------------------------------------------------------------------

def plot_heatmap(M, annotate=False, text_color='w', format_displayed='%.2f', x_ticks=False, y_ticks=False, color_map='viridis', cbar=False, colorbar_label=None, figure_size=(5,5), title=None, save_filename=None):
    """
    Plots a heatmap for any matrix :math:`\\mathbf{M}`.

    **Example:**

    .. code:: python

        from PCAfold import PCA, plot_heatmap
        import numpy as np

        # Generate dummy data set:
        X = np.random.rand(100,5)

        # Perform PCA and obtain the covariance matrix:
        pca_X = PCA(X)
        covariance_matrix = pca_X.S

        # Plot a heatmap of the covariance matrix:
        plt = plot_heatmap(covariance_matrix, annotate=True, title='Covariance', save_filename='covariance.pdf')
        plt.close()

    :param M:
        ``numpy.ndarray`` specifying the matrix :math:`\\mathbf{M}`.
    :param annotate: (optional)
        ``bool`` specifying whether numerical values of matrix elements should be plot on top of the heatmap.
    :param text_color: (optional)
        ``str`` specifying the color of the annotation text.
    :param format_displayed: (optional)
        ``str`` specifying the display format for the numerical entries inside the
        table. By default it is set to ``'%.2f'``.
    :param x_ticks: (optional)
        ``bool`` specifying whether ticks on the :math:`x` -axis should be plotted.
    :param y_ticks: (optional)
        ``bool`` specifying whether ticks on the :math:`y` -axis should be plotted.
    :param color_map: (optional)
        ``str`` or ``matplotlib.colors.ListedColormap`` specifying the colormap to use as per ``matplotlib.cm``. Default is ``'viridis'``.
    :param cbar: (optional)
        ``bool`` specifying whether colorbar should be plotted.
    :param colorbar_label: (optional)
        ``str`` specifying colorbar label annotation.
        If set to ``None``, colorbar label will not be plotted.
    :param figure_size: (optional)
        ``tuple`` specifying figure size.
    :param title: (optional)
        ``str`` specifying plot title. If set to ``None`` title will not be
        plotted.
    :param save_filename: (optional)
        ``str`` specifying plot save location/filename. If set to ``None``
        plot will not be saved. You can also set a desired file extension,
        for instance ``.pdf``. If the file extension is not specified, the default
        is ``.png``.

    :return:
        - **plt** - ``matplotlib.pyplot`` plot handle.
    """

    if not isinstance(M, np.ndarray):
        raise ValueError("Parameter `M` has to be of type `numpy.ndarray`.")

    try:
        (n_x, n_y) = np.shape(M)
    except:
        raise ValueError("Parameter `M` has to be a matrix.")

    if not isinstance(annotate, bool):
        raise ValueError("Parameter `annotate` has to be of type `bool`.")

    if not isinstance(text_color, str):
        raise ValueError("Parameter `text_color` has to be of type `str`.")

    if not isinstance(format_displayed, str):
        raise ValueError("Parameter `format_displayed` has to be of type `str`.")

    if not isinstance(x_ticks, bool):
        raise ValueError("Parameter `x_ticks` has to be of type `bool`.")

    if not isinstance(y_ticks, bool):
        raise ValueError("Parameter `y_ticks` has to be of type `bool`.")

    if not isinstance(color_map, str):
        if not isinstance(color_map, ListedColormap):
            raise ValueError("Parameter `color_map` has to be of type `str` or `matplotlib.colors.ListedColormap`.")

    if not isinstance(cbar, bool):
        raise ValueError("Parameter `cbar` has to be of type `bool`.")

    if colorbar_label is not None:
        if not isinstance(colorbar_label, str):
            raise ValueError("Parameter `colorbar_label` has to be of type `str`.")

    if not isinstance(figure_size, tuple):
        raise ValueError("Parameter `figure_size` has to be of type `tuple`.")

    if title is not None:
        if not isinstance(title, str):
            raise ValueError("Parameter `title` has to be of type `str`.")

    if save_filename is not None:
        if not isinstance(save_filename, str):
            raise ValueError("Parameter `save_filename` has to be of type `str`.")

    fig = plt.figure(figsize=figure_size)
    ims = plt.imshow(M, cmap=color_map)

    if x_ticks:
        plt.xticks(np.arange(0,n_x))
    else:
        plt.xticks([])

    if y_ticks:
        plt.yticks(np.arange(0,n_y))
    else:
        plt.yticks([])

    if annotate:
        for i in range(n_x):
            for j in range(n_y):
                text = plt.text(j, i, str(format_displayed % M[i,j]), fontsize=font_text, ha="center", va="center", color=text_color)

    if cbar:
        cb = fig.colorbar(ims)
        cb.ax.tick_params(labelsize=font_colorbar_axes)
        if colorbar_label != None: cb.set_label(colorbar_label, fontsize=font_colorbar, rotation=0, horizontalalignment='left')

    if title != None:
        plt.title(title, fontsize=font_title, **csfont)

    if save_filename != None:
        plt.savefig(save_filename, dpi=save_dpi, bbox_inches='tight')

    return plt

# ------------------------------------------------------------------------------

def plot_heatmap_sequence(M, annotate=False, text_color='w', format_displayed='%.2f', x_ticks=False, y_ticks=False, color_map='viridis', cbar=False, colorbar_label=None, figure_size=(5,5), title=None, save_filename=None):
    """
    Plots a sequence of heatmaps for matrices :math:`\\mathbf{M}` stored in a list.

    **Example:**

    .. code:: python

        from PCAfold import PCA, plot_heatmap_sequence
        import numpy as np

        # Generate dummy data set:
        X = np.random.rand(100,5)

        # Perform PCA and obtain the covariance matrices:
        pca_X_auto = PCA(X, scaling='auto')
        pca_X_range = PCA(X, scaling='range')
        pca_X_vast = PCA(X, scaling='vast')
        covariance_matrices = [pca_X_auto.S, pca_X_range.S, pca_X_vast.S]
        titles = ['Auto', 'Range', 'VAST']

        # Plot a sequence of heatmaps of the covariance matrices:
        plt = plot_heatmap_sequence(covariance_matrices, annotate=True, text_color='w', format_displayed='%.1f', color_map='viridis', cbar=True, title=titles, figure_size=(12,3), save_filename='covariance-matrices.pdf')
        plt.close()

    :param M:
        ``list`` of ``numpy.ndarray`` specifying the matrices :math:`\\mathbf{M}`.
    :param annotate: (optional)
        ``bool`` specifying whether numerical values of matrix elements should be plot on top of the heatmap.
    :param text_color: (optional)
        ``str`` specifying the color of the annotation text.
    :param format_displayed: (optional)
        ``str`` specifying the display format for the numerical entries inside the
        table. By default it is set to ``'%.2f'``.
    :param x_ticks: (optional)
        ``bool`` specifying whether ticks on the :math:`x` -axis should be plotted.
    :param y_ticks: (optional)
        ``bool`` specifying whether ticks on the :math:`y` -axis should be plotted.
    :param color_map: (optional)
        ``str`` or ``matplotlib.colors.ListedColormap`` specifying the colormap to use as per ``matplotlib.cm``. Default is ``'viridis'``.
    :param cbar: (optional)
        ``bool`` specifying whether colorbar should be plotted.
    :param colorbar_label: (optional)
        ``str`` specifying colorbar label annotation.
        If set to ``None``, colorbar label will not be plotted.
    :param figure_size: (optional)
        ``tuple`` specifying figure size.
    :param title: (optional)
        ``str`` specifying plot title. If set to ``None`` title will not be
        plotted.
    :param save_filename: (optional)
        ``str`` specifying plot save location/filename. If set to ``None``
        plot will not be saved. You can also set a desired file extension,
        for instance ``.pdf``. If the file extension is not specified, the default
        is ``.png``.

    :return:
        - **plt** - ``matplotlib.pyplot`` plot handle.
    """

    if not isinstance(M, list):
        raise ValueError("Parameter `M` has to be of type `list`.")

    for i, element in enumerate(M):
        if not isinstance(element, np.ndarray):
            raise ValueError("Parameter `M` has to have elements of type `numpy.ndarray`.")

    n_subplots = len(M)

    if not isinstance(annotate, bool):
        raise ValueError("Parameter `annotate` has to be of type `bool`.")

    if not isinstance(text_color, str):
        raise ValueError("Parameter `text_color` has to be of type `str`.")

    if not isinstance(format_displayed, str):
        raise ValueError("Parameter `format_displayed` has to be of type `str`.")

    if not isinstance(x_ticks, bool):
        raise ValueError("Parameter `x_ticks` has to be of type `bool`.")

    if not isinstance(y_ticks, bool):
        raise ValueError("Parameter `y_ticks` has to be of type `bool`.")

    if not isinstance(color_map, str):
        if not isinstance(color_map, ListedColormap):
            raise ValueError("Parameter `color_map` has to be of type `str` or `matplotlib.colors.ListedColormap`.")

    if not isinstance(cbar, bool):
        raise ValueError("Parameter `cbar` has to be of type `bool`.")

    if colorbar_label is not None:
        if not isinstance(colorbar_label, str):
            raise ValueError("Parameter `colorbar_label` has to be of type `str`.")

    if not isinstance(figure_size, tuple):
        raise ValueError("Parameter `figure_size` has to be of type `tuple`.")

    if title is not None:
        if not isinstance(title, list):
            raise ValueError("Parameter `title` has to be of type `list`.")
        if len(title) != n_subplots:
            raise ValueError("Parameter `title` has to have the same number of elements as parameter `xy`.")

    if save_filename is not None:
        if not isinstance(save_filename, str):
            raise ValueError("Parameter `save_filename` has to be of type `str`.")


    fig = plt.figure(figsize=figure_size)
    spec = fig.add_gridspec(ncols=n_subplots, nrows=1)

    for index, element in enumerate(M):

        try:
            (n_x, n_y) = np.shape(element)
        except:
            raise ValueError("Parameter `M` has to have elements that are matrices.")

        ax = fig.add_subplot(spec[0,index])
        ims = plt.imshow(element, cmap=color_map)

        if x_ticks:
            plt.xticks(np.arange(0,n_x))
        else:
            plt.xticks([])

        if y_ticks:
            plt.yticks(np.arange(0,n_y))
        else:
            plt.yticks([])

        if annotate:
            for i in range(n_x):
                for j in range(n_y):
                    text = plt.text(j, i, str(format_displayed % element[i,j]), fontsize=font_text, ha="center", va="center", color=text_color)

        if cbar:
            cb = fig.colorbar(ims)
            cb.ax.tick_params(labelsize=font_colorbar_axes)
            if colorbar_label != None: cb.set_label(colorbar_label, fontsize=font_colorbar, rotation=0, horizontalalignment='left')

        if title != None:
            plt.title(title[index], fontsize=font_title, **csfont)

    fig.tight_layout(pad=0)

    if save_filename != None: plt.savefig(save_filename, dpi=save_dpi, bbox_inches='tight')

    return plt
