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

################################################################################
#
# Principal Component Analysis (PCA)
#
################################################################################

class PCA:
    """
    A class to support Principal Component Analysis.

    **Example:**

    .. code:: python

        pca = PCA(X)

    :param X:
        matrix of data to apply PCA to. Variables are in columns and
        observations are in rows.  Must have more observations than
        variables.
        *Note*: If a variable (column) is constant in the matrix ``X``, an error will
        arise telling the user to preprocess the data to remove that variable. The
        preprocess class can do this for the user.
    :param scaling: (optional)
        string specifying the scaling methodology as per
        ``preprocess.center_scale`` function.
    :param neta: (optional)
        number of retained eigenvalues - default is all
    :param useXTXeig: (optional)
        method for obtaining the eigenvalues ``L`` and eigenvectors ``Q``:

            * ``useXTXeig=False`` uses singular-value decomposition (from ``scipy.linalg.svd``)
            * ``useXTXeig=True`` (default) uses ``numpy.linalg.eigh`` on the covariance matrix ``R``

    """

    def __init__(self, X, scaling='std', neta=0, useXTXeig=True, nocenter=False):
        npts, nvar = X.shape
        if (npts < nvar):
            raise ValueError('Variables should be in columns; observations in rows.\n'
                             'Also ensure that you have more than one observation\n')

        manipulated, idx_removed, original, idx_retained = preprocess.remove_constant_vars(X)

        if len(idx_removed) != 0:
            raise ValueError('Constant variable detected. Must preprocess data for PCA.')

        self.scaling = scaling.upper()

        if neta > 0:
            self.neta = neta
        else:
            self.neta = nvar

        self.X, self.XCenter, self.XScale = preprocess.center_scale(X, self.scaling, nocenter)
        self.R = np.dot(self.X.transpose(), self.X) / npts
        if useXTXeig:
            L, Q = np.linalg.eigh(self.R)
            L = L / np.sum(L)
        else:
            U, s, vh = lg.svd(self.X)
            Q = vh.transpose()
            L = s * s / np.sum(s * s)

        isort = np.argsort(-np.diagonal(np.diag(L)))  # descending order
        Lsort = L[isort]
        Qsort = Q[:, isort]
        self.Q = Qsort
        self.L = Lsort

        self.nvar = len(self.L)
        val = np.zeros((self.nvar, self.neta))
        for i in range(self.neta):
            for j in range(self.nvar):
                val[j, i] = (self.Q[j, i] * np.sqrt(self.L[i])) / np.sqrt(self.R[j, j])
        self.loadings = val

    def x2eta(self, X, nocenter=False):
        """
        Calculate the principal components given the original data.

        :param X:
            a set of observations of variables x (observations in rows),
            unscaled, uncentered. These do not need to be the same
            observations as were used to construct the PCA object. They
            could be, e.g. functions of those variables.
        :param nocenter: (optional)
            Defaults to centering. A nonzero argument here will result in no
            centering being applied, even though it may be present in the
            original PCA transformation. Use this option only if you know what
            you are doing. PC source terms are an example of where we want this
            to be flagged.

        :return:
            - **eta** - the principal components.
        """
        neta = self.neta
        npts, nvar = X.shape
        assert nvar == len(self.L), "Number of variables inconsistent with number of eigenvectors."
        A = self.Q[:, 0:neta]
        x = np.zeros_like(X, dtype=float)

        if nocenter:
            for i in range(0, nvar):
                x[:, i] = X[:, i] / self.XScale[i]
            eta = x.dot(A)
        else:
            for i in range(0, nvar):
                x[:, i] = (X[:, i] - self.XCenter[i]) / self.XScale[i]
            eta = x.dot(A)
        return eta

    def eta2x(self, eta):
        """
        Calculate the principal components (or reconstructed variables).

        **Example:**

        .. code:: python

            eta = pca.eta2x(x) # calculate the principal components
            xrec = pca.eta2x(eta) # calculate reconstructed variables

        :param eta:
            the PCs.

        :return:
            - **X** - the unscaled, uncentered approximation to the data.
        """
        npts, neta = eta.shape
        assert neta == self.neta, "Number of variables provided inconsistent with number of PCs."
        A = self.Q[:, 0:neta]
        x = eta.dot(A.transpose())
        return preprocess.invert_center_scale(x, self.XCenter, self.XScale)

    def calculate_r2(self, X):
        """
        Calculates coefficient of determination :math:`R^2` values.
        Given the data used to construct the PCA, this calculates the
        :math:`R^2` values for the reduced representation of the data.
        If all of the eigenvalues are retained, then this should be unity.

        **Example:**

        .. code:: python

            r2 = pca.calculate_r2( X )

        :param X:
            data used to construct the PCA.

        :return:
            - **r2** coefficient of determination values for the reduced representation of the data.
        """
        npts, nvar = X.shape
        assert (npts > nvar), "Need more observations than variables."
        xapprox = self.eta2x(self.x2eta(X))
        r2 = np.zeros(nvar)
        for i in range(0, nvar):
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
        npts, nvar = X.shape
        self.neta = nvar
        err = X - self.eta2x(self.x2eta(X))
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
        npts, nvar = X.shape
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
        assert (neig <= self.nvar), "Number of eigenvectors specified is greater than the number of variables"
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
            npc = self.nvar
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
            nvar = self.nvar
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
            nvar = self.nvar
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

            eta = self.x2eta(x)  # the PCs based on the full set of x.

            nvarTot = self.nvar
            neta = self.neta

            idiscard = []
            q = nvarTot
            while q > neta:

                npts, nvar = x.shape
                m2cut = 1e12

                for i in range(nvar):

                    # look at a PCA obtained from a subset of x.
                    xs = np.hstack((x[:, np.arange(i)], x[:, np.arange(i + 1, nvar)]))
                    pca2 = PCA(xs, self.scaling, neta)
                    etaSub = pca2.x2eta(xs)

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
        nvar = self.nvar
        neta = np.arange(nvar) + 1
        netapts = len(neta)

        npts, nvar = data.shape
        r2 = np.zeros((netapts, nvar))
        r2vec = r2.copy()

        self.neta = np.max(neta, axis=0)
        eta = self.x2eta(data)

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
            np.savetxt(fid, np.array([self.XCenter]), delimiter=',', fmt='%6.12f')
        fid.close()

        fid = open(filename, 'a')
        fid.write('\n%s\n' % "Scaling Factors:")
        fid.close()

        with open(filename, 'ab') as fid:
            np.savetxt(fid, np.array([self.XScale]), delimiter=',', fmt='%6.12f')
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
        Calculate the U-scores (Principal Components).

        **Example:**

        .. code:: python

            uscores = pca.u_scores(X)

        U-scores = obtained by using the U-vectors, i.e. the eigenvectors of the
        covariance matrix S. The resulting U-scores are uncorrelated and have
        variances equal to the corresponding eigenvalues.

        This is entirely equivalent to x2eta.

        :param X:
            a set of observations of variables x (observations in rows),
            unscaled, uncentered. These do not need to be the same
            observations as were used to construct the PCA object. They
            could be, e.g. functions of those variables.

        :return:
            - **uscores** - U-scores or principal components (``eta``)
        """
        return self.x2eta(X)

    def w_scores(self, X):
        """
        Calculates the W-scores.

        **Example:**

        .. code:: python

            wscores = pca.w_scores( X )

        W-scores = The U vectors are scaled by the inverse of the eigenvalues
        square root, i.e. :math:`V = L^{-0.5} \cdot U`. The W-scores are still uncorrelated and
        have variances equal unity.

        :param X:
            a set of observations of variables x (observations in rows),
            unscaled, uncentered. These do not need to be the same
            observations as were used to construct the PCA object. They
            could be, e.g. functions of those variables.

        :return:
            - **wscores** - W-scores or principal components
        """
        eval = self.L[0:self.neta]
        return self.x2eta(X).dot(np.diag(1 / np.sqrt(eval)))

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
        scalErr = np.abs(a.XScale - b.XScale) / np.max(np.abs(a.XScale))
        centErr = np.abs(a.XCenter - b.XCenter) / np.max(np.abs(a.XCenter))

        RErr = np.abs(a.R - b.R) / np.max(np.abs(a.R))
        LErr = np.abs(a.L - b.L) / np.max(np.abs(a.L))
        QErr = np.abs(a.Q - b.Q) / np.max(np.abs(a.Q))

        tol = 10 * np.finfo(float).eps

        if a.XScale.all() == b.XScale.all() and a.neta == b.neta and np.all(scalErr < tol) and np.all(
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


def test():
    """
    Performs regression testing of the PCA class

    Testing (with no scaling) that PCA calculates the same mean that is added to a dataset with zero mean
    and returns the same Q and L from svd on the dataset with zero mean

    Then testing if feed already transformed eta's to PCA, will return same eta's when do x2eta

    Examples:
        result = PCA.test()  -> run tests

    :return: Boolean for whether tests passed or not
    """
    result = 1  # change to 0 if any failures occur

    tol = 10 * np.finfo(float).eps

    # create random dataset with zero mean
    npts = 100
    PHI = np.vstack(
        (np.sin(np.linspace(0, np.pi, npts)).T, np.cos(np.linspace(0, 2 * np.pi, npts)),
         np.linspace(0, np.pi, npts)))
    PHI, cntr, scl = preprocess.center_scale(PHI.T, 'NONE')

    # create random means for the dataset for comparison with PCA XCenter
    xbar = np.random.rand(1, PHI.shape[1])

    # svd on PHI to get Q and L for comparison with PCA Q and L
    U, s, V = lg.svd(PHI)
    L = s * s / np.sum(s * s)
    isort = np.argsort(-np.diagonal(np.diag(L)))  # descending order
    L = L[isort]
    Q = V.T[:, isort]

    # checking both methods for PCA:
    pca = PCA(PHI + xbar, 'NONE', useXTXeig=False)
    pca2 = PCA(PHI + xbar, 'NONE', useXTXeig=True)

    # comparing mean(centering), centered data, Q, and L
    if np.any(xbar - pca.XCenter > tol) or np.any(xbar - pca2.XCenter > tol):
        result = 0

    if np.any(PHI - pca.X > tol) or np.any(PHI - pca2.X > tol):
        result = 0

    if np.any(Q - pca.Q > tol) or np.any(Q - pca2.Q > tol):
        result = 0

    if np.any(L - pca.L > tol) or np.any(L - pca2.L > tol):
        result = 0

    # Check if feed eta's to PCA, return same eta's when do x2eta
    eta = pca.x2eta(PHI + xbar)  # dataset as example of eta's

    # both methods of PCA:
    pca = PCA(eta, 'NONE', useXTXeig=False)
    pca2 = PCA(eta, 'NONE', useXTXeig=True)

    # x2eta transformation:
    eta_new = pca.x2eta(eta)
    eta_new2 = pca2.x2eta(eta)

    # transformation can have different direction -> check sign is the same before compare eta's
    for i in range(pca.nvar):
        if np.sign(eta[0, i]) != np.sign(eta_new[0, i]):
            eta_new[:, i] *= -1
        if np.sign(eta[0, i]) != np.sign(eta_new2[0, i]):
            eta_new2[:, i] *= -1

    # checking eta's are the same from transformation of eta
    if np.any(eta - eta_new > tol) or np.any(eta - eta_new2 > tol):
        result = 0

    if result == 1:
        print('PCA tests passed')
    else:
        print('PCA tests failed')

    return result

################################################################################
#
# Principal Component Analysis on sampled data sets
#
################################################################################

def analyze_eigenvector_weights_movement(eigenvectors, variable_names, plot_variables=[], normalize=False, zero_norm=False, legend_label=[], title=None, save_filename=None):
    """
    This function analyzes the movement of weights on an eigenvector obtained
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
    :param variable_names:
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
    """

    (n_variables, n_versions) = np.shape(eigenvectors)

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

    return

def analyze_eigenvalue_distribution(X, idx_matrix, k_list, scaling, biasing_option, random_seed=None, title=None, save_filename=None):
    """
    This function analyzes the normalized eigenvalue distribution when PCA is
    performed on different versions of the reduced data sets
    :math:`\mathbf{X_r}` vs. on the original data set :math:`\mathbf{X}`.

    :param X:
        original (full) data set.
    :param idx_matrix:
        matrix of collected idx vectors.
    :param k_list:
        list of numerical labels for the idx_matrix columns.
    :param scaling:
        data scaling criterion.
    :param biasing_option:
        integer specifying biasing option.
        Can only attain values 1, 2, 3 or 4.
    :param random_seed: (optional)
        integer specifying random seed for random sample selection.
    :param title: (optional)
        boolean or string specifying plot title. If set to ``None``
        title will not be plotted.
    :param save_filename: (optional)
        plot save location/filename. If set to ``None`` plot will not be saved.

    :return:
        - **min_at_q2_k** - label for which the eigenvalue was smallest when q=2.
        - **min_at_q3_k** - label for which the eigenvalue was smallest when q=3.
        - **max_at_q2_k** - label for which the eigenvalue was largest when q=2.
        - **max_at_q3_k** - label for which the eigenvalue was largest when q=3.
    """

    n_k = len(k_list)
    (n_observations, n_variables) = np.shape(X)
    x_range = np.arange(1, n_variables+1)
    colors = plt.cm.Blues(np.linspace(0.3,1,n_k))

    fig, ax = plt.subplots(figsize=(n_variables, 6))

    for i_idx, k in enumerate(k_list):

        idx = idx_matrix[:,i_idx]
        (eigenvalues, _, _, _, _, _, _, _) = equilibrate_cluster_populations(X, idx, scaling, biasing_option=biasing_option, n_iterations=1, random_seed=random_seed)

        # Plot the eigenvalue distribution when PCA is performed on the original data set:
        if i_idx==0:
            original_distribution = plt.plot(np.arange(1, n_variables+1), eigenvalues[:,0], 'r-', linewidth=3, label='Original')

            # Initialize the minimum eigenvalue at q=2 and q=3:
            min_at_q2 = eigenvalues[1,0]
            min_at_q3 = eigenvalues[2,0]
            max_at_q2 = eigenvalues[1,0]
            max_at_q3 = eigenvalues[2,0]
            min_at_q2_k = 0
            min_at_q3_k = 0
            max_at_q2_k = 0
            max_at_q3_k = 0

        if eigenvalues[1,1] < min_at_q2:
            min_at_q2 = eigenvalues[1,1]
            min_at_q2_k = k
        if eigenvalues[2,1] < min_at_q3:
            min_at_q3 = eigenvalues[2,1]
            min_at_q3_k = k
        if eigenvalues[1,1] > max_at_q2:
            max_at_q2 = eigenvalues[1,1]
            max_at_q2_k = k
        if eigenvalues[2,1] > max_at_q3:
            max_at_q3 = eigenvalues[2,1]
            max_at_q3_k = k

        # Plot the eigenvalue distribution from the current equilibrated X_r for the current idx:
        plt.plot(np.arange(1, n_variables+1), eigenvalues[:,-1], 'o--', c=colors[i_idx], label='$k=' + str(k) + '$')

    plt.xticks(x_range, fontsize=font_annotations, **csfont)
    plt.xlabel('q [-]', fontsize=font_labels, **csfont)
    plt.ylabel('Normalized eigenvalue [-]', fontsize=font_labels, **csfont)
    plt.ylim(-0.05,1.05)
    plt.xlim(0, n_variables+1.5)
    plt.grid(alpha=0.3)

    if min_at_q2_k==0:
        plt.text(n_variables/3, 0.93, 'Min at $q=2$: Original, $\lambda=' + str(round(min_at_q2,3)) + '$')
    else:
        plt.text(n_variables/3, 0.93, 'Min at $q=2$: $k=' + str(min_at_q2_k) + '$, $\lambda=' + str(round(min_at_q2,3)) + '$')

    if min_at_q3_k==0:
        plt.text(n_variables/3, 0.79, 'Min at $q=3$: Original, $\lambda=' + str(round(min_at_q3,3)) + '$')
    else:
        plt.text(n_variables/3, 0.79, 'Min at $q=3$: $k=' + str(min_at_q3_k) + '$, $\lambda=' + str(round(min_at_q3,3)) + '$')

    if max_at_q2_k==0:
        plt.text(n_variables/3, 0.86, 'Max at $q=2$: Original, $\lambda=' + str(round(max_at_q2,3)) + '$')
    else:
        plt.text(n_variables/3, 0.86, 'Max at $q=2$: $k=' + str(max_at_q2_k) + '$, $\lambda=' + str(round(max_at_q2,3)) + '$')

    if max_at_q3_k==0:
        plt.text(n_variables/3, 0.72, 'Max at $q=3$: Original, $\lambda=' + str(round(max_at_q3,3)) + '$')
    else:
        plt.text(n_variables/3, 0.72, 'Max at $q=3$: $k=' + str(max_at_q3_k) + '$, $\lambda=' + str(round(max_at_q3,3)) + '$')

    lgnd = plt.legend(fontsize=font_legend-2, loc="upper right")
    plt.setp(lgnd.texts, **csfont)

    if title != False:
        plt.title(title, fontsize=font_title, **csfont)

    if save_filename != None:
        plt.savefig(save_filename, dpi = 500, bbox_inches='tight')

    return(min_at_q2_k, min_at_q3_k, max_at_q2_k, max_at_q3_k)

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
        - **X_center** - a vector of final centers that were used to center\
        the data set at the last (equlibration) iteration.
        - **X_scale** - a vector of final scales that were used to scale the\
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
    X_cs = pca_global.X
    X_center = pca_global.XCenter
    X_scale = pca_global.XScale

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
    global_pc_scores = pca_global.x2eta(X, nocenter=False)

    # Append the global PC-scores:
    pc_scores_matrix[:,:,0] = global_pc_scores

    if len(X_source) != 0:

        # Scale sources with the global scalings:
        X_source_cs = np.divide(X_source, X_scale)

        # Compute global PC-sources:
        global_pc_sources = pca_global.x2eta(X_source, nocenter=True)

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

            # Generate the reduced data set X_r:
            X_r = X[idx_train,:]

            # Perform PCA on X_r:
            pca = PCA(X_r, scaling, n_components, useXTXeig=True)

            # Compute local eigenvectors:
            eigenvectors = pca.Q

            # Compute local eigenvalues:
            local_eigenvalues = pca.L
            maximum_local_eigenvalue = np.max(local_eigenvalues)

            # Compute local PC-scores:
            pc_scores = X_cs.dot(eigenvectors[:,0:n_components])

            # Append the local PC-scores:
            pc_scores_matrix[:,:,iter+1] = pc_scores

            if len(X_source) != 0:

                # Compute local PC-sources:
                pc_sources = X_source_cs.dot(eigenvectors[:,0:n_components])

                # Append the global PC-sources:
                pc_sources_matrix[:,:,iter+1] = pc_sources

        # Biasing option 2 -----------------------------------------------------
        elif biasing_option == 2:

            # Generate the reduced data set X_r:
            X_r = X_cs[idx_train,:]

            # Perform PCA on X_r:
            pca = PCA(X_r, 'none', n_components, useXTXeig=True, nocenter=True)

            # Compute local eigenvectors:
            eigenvectors = pca.Q

            # Compute local eigenvalues:
            local_eigenvalues = pca.L
            maximum_local_eigenvalue = np.max(local_eigenvalues)

            # Compute local PC-scores:
            pc_scores = X_cs.dot(eigenvectors[:,0:n_components])

            # Append the local PC-scores:
            pc_scores_matrix[:,:,iter+1] = pc_scores

            if len(X_source) != 0:

                # Compute local PC-sources:
                pc_sources = X_source_cs.dot(eigenvectors[:,0:n_components])

                # Append the global PC-sources:
                pc_sources_matrix[:,:,iter+1] = pc_sources

        # Biasing option 3 -----------------------------------------------------
        elif biasing_option == 3:

            # Generate the reduced data set X_r:
            X_r = X[idx_train,:]

            # Perform PCA on X_r:
            pca = PCA(X_r, scaling, n_components, useXTXeig=True)
            X_center = pca.XCenter
            X_scale = pca.XScale

            # Compute local eigenvectors:
            eigenvectors = pca.Q

            # Compute local eigenvalues:
            local_eigenvalues = pca.L
            maximum_local_eigenvalue = np.max(local_eigenvalues)

            # Compute local PC-scores (the original data set will be centered and scaled with the local centers and scales):
            pc_scores = pca.x2eta(X, nocenter=False)

            # Append the local PC-scores:
            pc_scores_matrix[:,:,iter+1] = pc_scores

            if len(X_source) != 0:

                # Compute local PC-sources:
                pc_sources = pca.x2eta(X_source, nocenter=True)

                # Append the global PC-sources:
                pc_sources_matrix[:,:,iter+1] = pc_sources

        # Biasing option 4 -----------------------------------------------------
        elif biasing_option == 4:

            # Generate the reduced data set X_r:
            X_r = X[idx_train,:]

            # Compute the current centers and scales of X_r:
            (_, C_r, D_r) = preprocess.center_scale(X_r, scaling)
            X_center = C_r
            X_scale = D_r

            # Pre-process the global data set with the current C_r and D_r:
            X_cs = (X - C_r) / D_r

            # Perform PCA on the original data set X:
            pca = PCA(X_cs, 'none', n_components, useXTXeig=True, nocenter=True)

            # Compute local eigenvectors:
            eigenvectors = pca.Q

            # Compute local eigenvalues:
            local_eigenvalues = pca.L
            maximum_local_eigenvalue = np.max(local_eigenvalues)

            # Compute local PC-scores:
            pc_scores = X_cs.dot(eigenvectors[:,0:n_components])

            # Append the local PC-scores:
            pc_scores_matrix[:,:,iter+1] = pc_scores

            if len(X_source) != 0:

                # Compute local PC-sources:
                pc_sources = X_source_cs.dot(eigenvectors[:,0:n_components])

                # Append the global PC-sources:
                pc_sources_matrix[:,:,iter+1] = pc_sources

        # Append the local eigenvectors:
        eigenvectors_matrix[:,:,iter+1] = eigenvectors[:,0:n_components]

        # Append the local eigenvalues:
        eigenvalues = np.hstack((eigenvalues, np.reshape(local_eigenvalues, (n_variables, 1))/maximum_local_eigenvalue))

    # Remove the first column of zeros:
    eigenvalues = eigenvalues[:,1::]

    return(eigenvalues, eigenvectors_matrix, pc_scores_matrix, pc_sources_matrix, idx_train, X_center, X_scale)

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
