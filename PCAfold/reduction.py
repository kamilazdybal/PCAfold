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

    **Example:**

    .. code:: python

        from PCAfold import PCA
        import numpy as np

        X = np.random.rand(100,20)
        pca_X = PCA(X, scaling='none', neta=2, useXTXeig=True, nocenter=False)

    :param X:
        original data set :math:`\mathbf{X}`.
    :param scaling: (optional)
        string specifying the scaling methodology as per
        ``preprocess.center_scale`` function.
    :param neta: (optional)
        number of retained Principal Components :math:`q`. If set to ``0`` all eigenvalues are retained.
    :param useXTXeig: (optional)
        method for obtaining the eigenvalues ``L`` and eigenvectors ``Q``:

            * ``useXTXeig=False`` uses singular-value decomposition (from ``scipy.linalg.svd``)
            * ``useXTXeig=True`` (default) uses ``numpy.linalg.eigh`` on the covariance matrix ``R``

    :raises ValueError:
        if the original data set :math:`\mathbf{X}` has more variables (columns)
        then observations (rows).

    :raises ValueError:
        if a constant column is detected in the original data set :math:`\mathbf{X}`.

    :raises ValueError:
        if ``scaling`` method is not a string or is not within the available scalings.

    :raises ValueError:
        if ``neta`` is not an integer or is negative.

    :raises ValueError:
        if ``useXTXeig`` is not a boolean.

    :raises ValueError:
        if ``nocenter`` is not a boolean.

    **Attributes:** (read only)

        - **X_cs** - centered and scaled data set :math:`\mathbf{X_{cs}}`.
        - **X_center** - vector of centers :math:`\mathbf{C}` applied on the original data set :math:`\mathbf{X}`.
        - **X_scale** - vector of scales :math:`\mathbf{D}` applied on the original data set :math:`\mathbf{X}`.
        - **R** - covariance matrix.
        - **L** - eigenvalues.
        - **Q** - eigenvectors (vectors are stored in columns, rows correspond to weights).
        - **loadings** - loadings (vectors are stored in columns, rows correspond to weights).
    """

    def __init__(self, X, scaling='std', neta=0, useXTXeig=True, nocenter=False):

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

        # Check neta:
        if not isinstance(neta, int):
            raise ValueError("Parameter `neta` has to be an integer.")
        else:
            if (neta < 0) or (neta > n_variables):
                raise ValueError("Parameter `neta` cannot be negative or larger than number of variables in a data set.")
            else:
                if isinstance(neta, bool):
                    raise ValueError("Parameter `neta` has to be an integer.")

        # Check useXTXeig:
        if not isinstance(useXTXeig, bool):
            raise ValueError("Parameter `useXTXeig` has to be a boolean.")

        # Check nocenter:
        if not isinstance(nocenter, bool):
            raise ValueError("Parameter `nocenter` has to be a boolean.")

        self.__scaling = scaling.upper()

        if neta > 0:
            self.__neta = neta
        else:
            self.__neta = n_variables

        # Center and scale the data set:
        self.__X_cs, self.__X_center, self.__X_scale = preprocess.center_scale(X, self.scaling, nocenter)

        # Compute covariance matrix:
        self.__R = np.dot(self.X_cs.transpose(), self.X_cs) / (n_observations-1)

        # Perform PCA with eigendecomposition of the covariance matrix:
        if useXTXeig:
            L, Q = np.linalg.eigh(self.R)
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
        self.__Q = Qsort
        self.__L = Lsort

        self.__nvar = len(self.L)
        loadings_matrix = np.zeros((self.__nvar, self.neta))

        # Compute loadings:
        for i in range(self.neta):
            for j in range(self.__nvar):
                loadings_matrix[j, i] = (self.Q[j, i] * np.sqrt(self.L[i])) / np.sqrt(self.R[j, j])

        self.__loadings = loadings_matrix

    @property
    def scaling(self):
        return self.__scaling

    @property
    def neta(self):
        return self.__neta

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
    def R(self):
        return self.__R

    @property
    def Q(self):
        return self.__Q

    @property
    def L(self):
        return self.__L

    @property
    def loadings(self):
        return self.__loadings

    def transform(self, X, nocenter=False):
        """
        This function transforms any data set :math:`\mathbf{X}` to the new
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

        **Example:**

        .. code:: python

            from PCAfold import PCA
            import numpy as np

            X = np.random.rand(100,20)
            pca_X = PCA(X, scaling='none', neta=2, useXTXeig=True, nocenter=False)

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

            .. warning::

                Set ``nocenter=True`` only if you know what you are doing.

        :raises ValueError:
            if ``nocenter`` is not a boolean.

        :raises ValueError:
            if the number of variables in a data set is inconsistent with number of eigenvectors.

        :return:
            - **principal_components** - the :math:`q`-first Principal Components :math:`\mathbf{Z_q}`.
        """

        if not isinstance(nocenter, bool):
            raise ValueError("Parameter `nocenter` has to be a boolean.")

        neta = self.neta

        (n_observations, n_variables) = np.shape(X)

        if n_variables != len(self.L):
            raise ValueError("Number of variables in a data set is inconsistent with number of eigenvectors.")

        A = self.Q[:, 0:neta]
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

        **Example:**

        .. code:: python

            from PCAfold import PCA
            import numpy as np

            X = np.random.rand(100,20)
            pca_X = PCA(X, scaling='none', neta=2, useXTXeig=True, nocenter=False)

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

            .. warning::

                Set ``nocenter=True`` only if you know what you are doing.

        :raises ValueError:
            if ``nocenter`` is not a boolean.

        :return:
            - **X_rec** - rank-:math:`q` reconstruction of the original data set.
        """

        if not isinstance(nocenter, bool):
            raise ValueError("Parameter `nocenter` has to be a boolean.")

        (n_observations, n_components) = np.shape(principal_components)

        # Select n_components first Principal Components:
        A = self.Q[:, 0:n_components]

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
        for the rank-:math:`q` reconstruction of the original
        data set :math:`\mathbf{X_{rec}}`:

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

            X = np.random.rand(100,20)
            pca_X = PCA(X, scaling='auto', neta=10, useXTXeig=True, nocenter=False)

            # Calculate the R2 values:
            r2 = pca_X.calculate_r2(X)

        :param X:
            original data set :math:`\mathbf{X}`.

        :return:
            - **r2** - coefficient of determination values :math:`R^2` for the rank-:math:`q` reconstruction of the original data set.
        """

        (n_observations, n_variables) = np.shape(X)

        assert (n_observations > n_variables), "Need more observations than variables."

        xapprox = self.reconstruct(self.transform(X))
        r2 = np.zeros(n_variables)

        for i in range(0, n_variables):
            r2[i] = 1 - np.sum((X[:, i] - xapprox[:, i]) * (X[:, i] - xapprox[:, i])) / np.sum(
                (X[:, i] - X[:, i].mean(axis=0)) * (X[:, i] - X[:, i].mean(axis=0)))

        return r2

    def data_consistency_check(self, X, errorsAreFatal=True):
        """
        Checks if the supplied data matrix ``X`` is consistent with the PCA object.

        **Example:**

        .. code:: python

            pca.data_consistency_check( X, errorsAreFatal )

        :param X:
            the independent variables.
        :param errorsAreFatal: (optional)
            flag indicating if an error should be raised if an incompatibility
            is detected - default is True.

        :return:
            - **okay** - boolean for whether or not supplied data matrix ``X``\
            is consistent with the PCA object.
        """
        n_observations, nvar = X.shape
        self.neta = nvar
        err = X - self.reconstruct(self.transform(X))
        isBad = (np.max(np.abs(err), axis=0) / np.max(np.abs(X), axis=0) > 1e-10).any() or (
            np.min(np.abs(err), axis=0) / np.min(np.abs(X), axis=0) > 1e-10).any()
        if isBad and errorsAreFatal:
            raise ValueError('it appears that the data is not consistent with the data used to construct the PCA')
        okay = not isBad
        return okay

    def convergence(self, X, nmax, names=[], printwidth=10):
        """
        Print :math:`R^2` values as a function of number of retained eigenvalues.

        **Example:**

        .. code:: python

            pca.convergence( X, nmax )
            pca.convergence( X, nmax, names )

        Print :math:`R^2` values retaining 1-5 eigenvalues:

        .. code:: python

            pca.convergence(X,5)

        :param X:
            the original dataset.
        :param nmax:
            the maximum number of PCs to consider.
        :param names: (optional)
            the names of the variables - otherwise variables are numbered.
        :param printwidth: (optional)
            width of columns printed out.

        :return:
            - **r2** matrix ``(nmax,nvar)`` containing the :math:`R^2` values\
            for each variable as a function of the number of retained eigenvalues.
        """
        n_observations, nvar = X.shape
        r2 = np.zeros((nmax, nvar))
        r2vec = np.zeros((nmax, nvar + 1))
        self.data_consistency_check(X)

        if len(names) > 0:
            assert len(names) == nvar, "Number of names given is not consistent with number of variables."
        else:
            for i in range(nvar):
                names.append(str(i + 1))

        neig = np.zeros((nmax), dtype=int)
        for i in range(nmax):
            self.neta = i + 1
            neig[i] = self.neta
            r2[i, :] = self.calculate_r2(X)

            r2vec[i, 0:-1] = np.round(r2[i, :], 8)
            r2vec[i, -1] = np.round(r2[i, :].mean(axis=0), 8)

        row_format = '|'
        for i in range(nvar + 2):
            row_format += ' {' + str(i) + ':<' + str(printwidth) + '} |'
        rownames = names
        rownames.insert(0, 'n Eig')
        rownames.append('Mean')

        print(row_format.format(*rownames))
        for i, row in zip(neig, r2vec):
            print(row_format.format(i, *row))

        return r2

    def eig_bar_plot_maker(self, neig, DataName, barWidth=0.3, plotABS=False):
        """
        Produces a bar plot of the weight of each state variable in the eigenvectors

        **Example:**

        .. code:: python

            pca.eig_bar_plot_maker( neig )

        :param neig:
            number of eigenvectors that you want to keep in the plot
        :param DataName:
            list containing the names of the variables
        :param barWidth: (optional)
            width of each bar in the plot
        :param plotABS: (optional)
            default False - plots the eigenvectors keeping their sign
            if True - plots the absolute value of the eigenvectors

        :return: (plot)
        """
        assert (neig <= self.__nvar), "Number of eigenvectors specified is greater than the number of variables"
        xtick = np.arange(len(self.Q))
        for i in range(neig):
            if i == 0:
                lab = '1st EigVec'
            elif i == 1:
                lab = '2nd EigVec'
            elif i == 2:
                lab = '3rd EigVec'
            else:
                lab = str(i + 1) + 'th EigVec'
            if plotABS:
                plt.bar(xtick + (i - 1) * barWidth, np.abs(self.Q[:, i]), label=lab, width=barWidth, align='center')
            else:
                plt.bar(xtick + (i - 1) * barWidth, self.Q[:, i], label=lab, width=barWidth, align='center')
        plt.xticks(xtick, DataName)
        plt.grid()
        plt.ylabel('Weights')
        plt.legend()
        plt.show()

    def plot_convergence(self, npc=0):
        """
        Plot the eigenvalues (bars) and the cumulative sum (line) to visualize
        the percent variance in the data explained by each principal component
        individually and by each principal component cumulatively.

        :param npc: (optional)
            how many principal components you want to visualize (default is all).

        :return: (plot)
        """
        if npc == 0:
            npc = self.__nvar
        npcvec = np.arange(0, npc)
        plt.plot(npcvec + 1, np.cumsum(self.L[npcvec]), 'b', label='Cumulative')
        plt.bar(npcvec + 1, self.L[npcvec])
        plt.xlim([0.5, npc + 0.5])
        plt.xticks(npcvec + 1)
        plt.xlabel('Principal Component')
        plt.ylabel('Percent Variance Explained')
        plt.legend()
        plt.grid()
        plt.show()

    def principal_variables(self, method='B2', x=[]):
        """
        Extract principal variables from a PCA

        **Example:**

        .. code:: python

            ikeep = principal_variables()
            ikeep = principal_variables('B4')
            ikeep = principal_variables('M2', X )

        :param method: (optional)
            the method for determining the principal variables.
            The following methods are currently supported:

            * ``'B4'`` - selects principal variables based on the variables\
            contained in the eigenvectors corresponding to the largest\
            eigenvalues.
            * ``'B2'`` - selects pvs based on variables contained in the\
            smallest eigenvalues.  These are discarded and the remaining\
            variables are used as the principal variables.  This is the default\
            method.
            * ``'M2'`` - at each iteration, each remaining variable is analyzed\
            via PCA. This is a very expensive method.

        :param x: (optional)
            data arranged with observations in rows and variables in columns.
            Note that this is only required for the ``'M2'`` method.

        :return:
            - **ikeep** - a vector of indices of retained variables
        """

        method = method.upper()

        if method == 'B2':  # B2 Method of Jolliffe (1972)
            nvar = self.__nvar
            neta = self.neta
            eigVec = self.Q  # eigenvectors

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
            ikeep = sd[np.argsort(sd)]

        elif method == 'B4':  # B4 Forward method
            nvar = self.__nvar
            neta = self.neta
            eigVec = self.Q  # eigenvectors

            # set indices for retained variables by looking at eigenvectors
            # corresponding to the retained eigenvalues
            ikeep = -1 * np.ones(neta)

            for i in range(neta):
                isrt = np.argsort(-np.abs(eigVec[:, i]))  # descending order

                # find the largest weight in this eigenvector
                # that has not yet been identified.
                for j in range(nvar):
                    ivar = isrt[j]
                    if np.all(ikeep != ivar):
                        ikeep[i] = ivar
                        break
            ikeep = ikeep[np.argsort(ikeep)]

        elif method == 'M2':  # Note: this is EXPENSIVE
            if len(x) == 0:
                raise ValueError('You must supply the data vector x when using the M2 method.')

            eta = self.transform(x)  # the PCs based on the full set of x.

            nvarTot = self.__nvar
            neta = self.neta

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
                print('discarding variable: %i\n' % (idisc + 1))

                q -= 1

            sd = np.setdiff1d(np.arange(nvarTot), idiscard)
            ikeep = sd[np.argsort(sd)]

        else:
            raise ValueError('Invalid method ' + method + ' for identifying principle variables')
        return ikeep

    def r2converge(self, data, names=[], fname=None):
        """
        Evaluate r2 values as a function of the number of retained eigenvalues.

        **Example:**

        .. code:: python

            r2, neta = pca.r2converge( data )
            r2, neta = pca.r2converge( data, names, 'r2.csv' )

        :param data:
            the data to fit.
        :param names: (optional)
            names of the data.
        :param fname: (optional)
            file to output r2 information to.

        :return:
            - **r2** - [neta,nvar] The r2 values.  Each column is a different variable and each row is for a different number of retained pcs.
        """
        nvar = self.__nvar
        neta = np.arange(nvar) + 1
        netapts = len(neta)

        n_observations, nvar = data.shape
        r2 = np.zeros((netapts, nvar))
        r2vec = r2.copy()

        self.neta = np.max(neta, axis=0)
        eta = self.transform(data)

        for i in range(netapts):
            self.neta = neta[i]
            r2[i, :] = self.calculate_r2(data)

        # dump out information
        if len(names) != 0:
            assert len(names) == nvar, "Number of names given is not consistent with number of variables."

            if fname:
                fid = open(fname, 'w')
                fid.write("neta:")
                for n in names:
                    fid.write(',%8s' % n)
                fid.write('\n')
                fid.close()

                for i in range(netapts):
                    fid = open(fname, 'a')
                    fid.write('%4i' % (i + 1))
                    fid.close()

                    with open(fname, 'ab') as fid:
                        np.savetxt(fid, np.array([r2[i, :]]), delimiter=' ', fmt=',%8.4f')
                    fid.close()
            else:
                row_format = '|'
                printwidth = 10
                for i in range(nvar + 1):
                    row_format += ' {' + str(i) + ':<' + str(printwidth) + '} |'
                rownames = names
                rownames.insert(0, 'neta')

                print(row_format.format(*rownames))
                for i, row in zip(neta, np.round(r2, 8)):
                    print(row_format.format(i, *row))

        return r2, neta

    def write_file_for_cpp(self, filename):
        """
        Writes the eigenvector matrix, centering and scaling vectors to .txt
        for reading into C++.
        *Note*: This function writes only the eigenvector matrix, centering and
        scaling factors - not all of the pca properties.

        **Example:**

        .. code:: python

            pca = PCA( x )
            pca.wite2file('pcaData.txt')

        :param filename:
            path (including name of text file) for destination of data file

        :return: (creates the ``.txt`` file in the destination specified by filename)
        """

        fid = open(filename, 'w')
        fid.write('%s\n' % "Eigenvectors:")
        fid.close()

        with open(filename, 'ab') as fid:
            np.savetxt(fid, self.Q, delimiter=',', fmt='%6.12f')
        fid.close()

        fid = open(filename, 'a')
        fid.write('\n%s\n' % "Centering Factors:")
        fid.close()

        with open(filename, 'ab') as fid:
            np.savetxt(fid, np.array([self.X_center]), delimiter=',', fmt='%6.12f')
        fid.close()

        fid = open(filename, 'a')
        fid.write('\n%s\n' % "Scaling Factors:")
        fid.close()

        with open(filename, 'ab') as fid:
            np.savetxt(fid, np.array([self.X_scale]), delimiter=',', fmt='%6.12f')
        fid.close()

    def set_retained_eigenvalues(self, method='SCREE GRAPH', option=None):
        """
        Help determine how many eigenvalues to retain.
        The following methods are available:

        - ``'TOTAL VARIANCE'`` retain the eigenvalues needed to account for a\
        specific percentage of the total variance (i.e. 80%). The required\
        number of PCs is then the smallest value of m for which this chosen\
        percentage is exceeded.

        * ``'INDIVIDUAL VARIANCE'`` retain the components whose eigenvalues are\
        greater than the average of the eigenvalues :cite:`Kaiser1960` or than 0.7\
        times he average of the eigenvalues :cite:`Jolliffe2002`. For a correlation\
        matrix this average equals 1.

        * ``'BROKEN STICK'`` select the retained PCs according to the Broken\
        Stick Model.

        * ``'SCREE GRAPH'`` use the scree graph, a plot of the eigenvalues\
        agaist their indexes, and look for a natural break between the large\
        and small eigenvalues.

        **Example:**

        .. code:: python

            pca = pca.set_retained_eigenvalues( method )

        This function provides a few methods to select the number of eigenvalues
        to be retained in the PCA reduction.

        :param method: (optional)
            method to use in selecting retained eigenvalues.
            Default is ``'SCREE GRAPH'``
        :param option: (optional)
            if not supplied, information will be obtained interactively.
            Only used for the ``'TOTAL VARIANCE'`` and ``'INDIVIDUAL VARIANCE'`` methods.

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
                raise ValueError('fraction of variance must be between 0 and 1')
            tot_var = np.sum(pca.L)
            neig = len(pca.L)
            fracVar = 0
            while (fracVar < frac) and (neta <= neig):
                fracVar += pca.L[neta - 1] / tot_var
                neta += 1
            pca.neta = neta - 1
        elif method == 'INDIVIDUAL VARIANCE':
            if option:
                fac = option
            else:
                print('Choose threshold between 0 and 1\n(1->Kaiser, 0.7->Joliffe)\n')
                fac = float(input(''))
            assert (fac > 0.) and (fac <= 1.), 'fraction of variance must be between 0 and 1'

            cutoff = fac * pca.L.mean(axis=0)
            neta = 1
            if np.any(pca.L > cutoff):
                neta = neig
            else:
                while (pca.L[neta - 1] > cutoff) and neta <= neig:
                    neta += 1
            pca.neta = neta - 1

        elif method == 'BROKEN STICK':
            neta = 1
            stick_stop = 1
            while (stick_stop == 1) and (neta <= neig):
                broken_stick = 0
                for j in np.arange(neta, neig + 1):
                    broken_stick += 1 / j
                stick_stop = pca.L[neta - 1] > broken_stick
                neta += 1
            pca.neta = neta - 1

        elif method == 'SCREE PLOT' or method == 'SCREE GRAPH':
            pca.plot_convergence()
            pca.neta = int(float(input('Select number of retained eigenvalues: ')))

        else:
            raise ValueError('Unsupported method: ' + method)

        return pca

    def u_scores(self, X):
        """
        Calculate the U-scores (Principal Components):

        .. math::

            \mathbf{U_{scores}} = \mathbf{X_{cs}} \mathbf{A_q}

        This function is equivalent to ``PCA.transform``.

        **Example:**

        .. code:: python

            from PCAfold import PCA
            import numpy as np

            X = np.random.rand(100,20)
            pca_X = PCA(X, scaling='auto', neta=10, useXTXeig=True, nocenter=False)

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

        u_scores = self.transform(X)

        return(u_scores)

    def w_scores(self, X):
        """
        This function calculates the W-scores which are the Principal Components
        scaled by the inverse square root of the corresponding eigenvalue:

        .. math::

            \mathbf{W_{scores}} = \\frac{\mathbf{Z_q}}{\\sqrt{\mathbf{L_q}}}

        where :math:`\mathbf{L_q}` are the :math:`q`-first eigenvalues.
        The W-scores are still uncorrelated and have variances equal unity.

        **Example:**

        .. code:: python

            from PCAfold import PCA
            import numpy as np

            X = np.random.rand(100,20)
            pca_X = PCA(X, scaling='auto', neta=10, useXTXeig=True, nocenter=False)

            # Calculate the U-scores:
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

        eval = self.L[0:self.neta]

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

        RErr = np.abs(a.R - b.R) / np.max(np.abs(a.R))
        LErr = np.abs(a.L - b.L) / np.max(np.abs(a.L))
        QErr = np.abs(a.Q - b.Q) / np.max(np.abs(a.Q))

        tol = 10 * np.finfo(float).eps

        if a.X_scale.all() == b.X_scale.all() and a.neta == b.neta and np.all(scalErr < tol) and np.all(
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
    `Biasing options <https://pcafold.readthedocs.io/en/latest/user/data-reduction.html#id4>`_
    section of the documentation for more information on the available options.

    :param X:
        original (full) data set.
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
        source terms corresponding to the state-space variables in ``X``.

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
        pca = PCA(X_r, scaling, n_components, useXTXeig=True)
        C_r = pca.X_center
        D_r = pca.X_scale

        # Compute eigenvectors:
        eigenvectors = pca.Q

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
        pca = PCA(X_r, 'none', n_components, useXTXeig=True, nocenter=True)
        C_r = pca.X_center
        D_r = pca.X_scale

        # Compute eigenvectors:
        eigenvectors = pca.Q

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
        pca = PCA(X_r, scaling, n_components, useXTXeig=True)
        C_r = pca.X_center
        D_r = pca.X_scale

        # Compute eigenvectors:
        eigenvectors = pca.Q

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
        pca = PCA(X_cs, 'none', n_components, useXTXeig=True, nocenter=True)

        # Compute eigenvectors:
        eigenvectors = pca.Q

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

    :param X:
        original (full) data set.
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
    pca_original = PCA(X, scaling, n_components, useXTXeig=True)

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
    plt.xlim(0, n_variables+1.5)
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

    Future implementation might include equilibration that slows down close to equilibrium. This might be helpful for sensitivity analysis.

    **Interpretation for the outputs:**

    This function returns 3D arrays ``eigenvectors``, ``pc_scores`` and
    ``pc_sources`` that have the following structure:

    .. image:: ../images/cbpca-equlibrate-outputs.png
        :width: 700
        :align: center

    :param X:
        original (full) data set.
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
    pca_global = PCA(X, scaling, n_components, useXTXeig=True)

    # Get a centered and scaled data set:
    X_cs = pca_global.X_cs
    X_center = pca_global.X_center
    X_scale = pca_global.X_scale

    # Compute global eigenvectors:
    global_eigenvectors = pca_global.Q

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

def plot_2d_manifold(manifold_2d, color_variable=[], x_label=None, y_label=None, colorbar_label=None, title=None, save_filename=None):
    """
    This function plots a 2-dimensional manifold given the matrix
    defining the manifold.

    :param manifold_2d:
        matrix specifying the 2-dimensional manifold. It is assumed that the
        first column will be :math:`x`-axis values and second column will be
        the :math:`y`-axis values.
    :param color_variable: (optional)
        a 1D vector or string specifying color for the manifold. If it is a
        vector, it has to have length consistent with the number of observations
        in ``manifold_2d`` matrix.
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

    :return:
        - **plt** - plot handle.
    """

    fig, axs = plt.subplots(1, 1, figsize=(6,6))

    if len(color_variable) == 0:
        scat = plt.scatter(manifold_2d[:,0].ravel(), manifold_2d[:,1].ravel(), c='k', marker='o', s=scatter_point_size, edgecolor='none', alpha=1)
    else:
        scat = plt.scatter(manifold_2d[:,0].ravel(), manifold_2d[:,1].ravel(), c=color_variable, marker='o', s=scatter_point_size, edgecolor='none', alpha=1)

    plt.xticks(fontsize=font_axes, **csfont)
    plt.yticks(fontsize=font_axes, **csfont)
    if x_label != None: plt.xlabel(x_label, fontsize=font_labels, **csfont)
    if y_label != None: plt.ylabel(y_label, fontsize=font_labels, **csfont)
    plt.grid(alpha=0.3)

    cb = fig.colorbar(scat)
    cb.ax.tick_params(labelsize=font_colorbar_axes)
    if colorbar_label != None: cb.set_label(colorbar_label, fontsize=font_colorbar, rotation=0, horizontalalignment='left')

    if title != None: plt.title(title, fontsize=font_title, **csfont)
    if save_filename != None: plt.savefig(save_filename, dpi = 500, bbox_inches='tight')

    return plt
