import numpy as np
from scipy import linalg as lg
import matplotlib.pyplot as plt
import copy as cp


def center_scale(X, scaling, nocenter=False):
    """
    Centers and scales data - used in constructing PCA objects

    Example:
      xs = center_scale( X, opts )

    :param X: uncentered, unscaled data
    :param scaling: the scaling methodology
    :return: the centered and scaled data (Xout), the value for centering (xbar), the value for scaling (d)
    """
    Xout = np.zeros_like(X, dtype=float)
    xbar = X.mean(axis=0)
    npts, nvar = X.shape

    dev = 0 * xbar
    kurt = 0 * xbar

    for i in range(0, nvar):
        # calculate the standard deviation (required for some scalings)
        dev[i] = np.std(X[:, i], ddof=0)

        # calculate the kurtosis (required for some scalings)
        kurt[i] = np.sum((X[:, i] - xbar[i]) ** 4) / npts / (np.sum((X[:, i] - xbar[i]) ** 2) / npts) ** 2

    scaling = scaling.upper()
    eps = np.finfo(float).eps
    if scaling == 'NONE' or scaling == '':
        d = np.ones(nvar)
    elif scaling == 'AUTO' or scaling == 'STD':
        d = dev
    elif scaling == 'VAST':
        d = dev * dev / (xbar + eps)
    elif scaling == 'VAST_2':
        d = dev * dev * kurt * kurt / (xbar + eps)
    elif scaling == 'VAST_3':
        d = dev * dev * kurt * kurt / np.max(X, axis=0)
    elif scaling == 'VAST_4':
        d = dev * dev * kurt * kurt / (np.max(X, axis=0) - np.min(X, axis=0))
    elif scaling == 'RANGE':
        d = np.max(X, axis=0) - np.min(X, axis=0)
    elif scaling == 'LEVEL':
        d = xbar
    elif scaling == 'MAX':
        d = np.max(X, axis=0)
    elif scaling == 'PARETO':
        d = np.zeros(nvar)
        for i in range(0, nvar):
            d[i] = np.sqrt(np.std(X[:, i], ddof=0))
    elif scaling == 'POISSON':
        d = np.sqrt(xbar)
    else:
        raise ValueError('Unsupported scaling option')

    for i in range(0, nvar):
        if nocenter:
            Xout[:, i] = (X[:, i]) / d[i]
        else:
            Xout[:, i] = (X[:, i] - xbar[i]) / d[i]

    if nocenter:
        xbar = np.zeros(nvar)

    return Xout, xbar, d


def inv_center_scale(x, xcenter, xscale):
    """
    Invert whatever scaling and centering was done by center_scale

    :param x: the dataset you want to un-center and un-scale
    :param xcenter: the centering done on the original dataset X
    :param xscale: the scaling done on the original dataset X
    :return: the unmanipulated/original dataset (X)
    """
    X = np.zeros_like(x, dtype=float)
    for i in range(0, len(xcenter)):
        X[:, i] = x[:, i] * xscale[i] + xcenter[i]
    return X


class preprocess:
    """
    class for preprocessing data which will check for the constant values and remove them, saving whatever manipulations
    were done so a user can manipulate new data in the same way

    Could make more complicated ones as needed
    """

    def __init__(self, X):
        self.manipulated, self.idx_removed, self.original, self.idx_retained = remove_constant_vars(X)


def remove_constant_vars(X, maxtol=1e-12, rangetol=1e-4):
    """
    Remove any constant variables (columns) in the data X
    Specifically preprocessing for PCA so the eigenvalue calculation doesn't break

    :param X: original data
    :param maxtol: tolerance for the maximum absolute value of a column (variable) in X to be saved
    :param rangetol: tolerance for the range (max-min) over the maximum absolute value of a column (variable) in X to
                     be saved
    :return: the manipulated data (manipulated), the indices of columns removed from X (idx_removed), the original
             data X (original), the indices of columns retained in X (idx_retained)
    """
    npts, nvar = X.shape
    original = np.copy(X)
    idx_removed = []
    idx_retained = []
    for i in reversed(range(0, nvar)):
        min = np.min(X[:, i], axis=0)
        max = np.max(X[:, i], axis=0)
        maxabs = np.max(np.abs(X[:, i]), axis=0)
        if (maxabs < maxtol) or ((max - min) / maxabs < rangetol):
            X = np.delete(X, i, 1)
            idx_removed.append(i)
        else:
            idx_retained.append(i)
    manipulated = X
    idx_removed = idx_removed[::-1]
    idx_retained = idx_retained[::-1]
    return manipulated, idx_removed, original, idx_retained


class PCA:
    """
    A class to support Principal Component Analysis

    Examples:
        pca = PCA(X)

    :param X: matrix of data to apply PCA to. Variables are in columns and
             observations are in rows.  Must have more observations than
             variables.

             NOTE: If a variable (column) is constant in the matrix X, an error will
             arise telling the user to preprocess the data to remove that variable. The
             preprocess class can do this for the user.
    :param scaling: (optional) default is 'AUTO'
                    'NONE'          no scaling
                    'AUTO' 'STD'    scale by std
                    'PARETO'        scale by std^2
                    'VAST'          scale by std^2 / mean
                    'VAST_2'        scale by std^2 * kurtosis^2 / mean
                    'VAST_3'        scale by std^2 * kurtosis^2 / max
                    'VAST_4'        scale by std^2 * kurtosis^2 / (max - min)
                    'RANGE'         scale by (max-min)
                    'LEVEL'         scale by mean
                    'MAX'           scale by max value
                    'POISSON'       scale by sqrt(mean)
    :param neta: (optional) number of retained eigenvalues - default is all
    :param useXTXeig: (optional) method for obtaining the eigenvalues (L) and eigenvectors (Q)
                      useXTXeig = False: uses singular-value decomposition (from scipy.linalg.svd)
                      useXTXeig = True (default): uses numpy.linalg.eigh on the covariance matrix (R)

    """

    def __init__(self, X, scaling='std', neta=0, useXTXeig=True, nocenter=False):
        npts, nvar = X.shape
        if (npts < nvar):
            raise ValueError('Variables should be in columns; observations in rows.\n'
                             'Also ensure that you have more than one observation\n')

        manipulated, idx_removed, original, idx_retained = remove_constant_vars(X)

        if len(idx_removed) != 0:
            raise ValueError('Constant variable detected. Must preprocess data for PCA.')

        self.scaling = scaling.upper()

        if neta > 0:
            self.neta = neta
        else:
            self.neta = nvar

        self.X, self.XCenter, self.XScale = center_scale(X, self.scaling, nocenter)
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

        :param X: a set of observations of variables x (observations in rows),
                  unscaled, uncentered. These do not need to be the same
                  observations as were used to construct the PCA object. They
                  could be, e.g. functions of those variables.
        :param nocenter:[OPTIONAL] Defaults to centering. A nonzero argument here
                        will result in no centering being applied, even though it may
                        be present in the original PCA transformation. Use this
                        option only if you know what you are doing. PC source terms
                        are an example of where we want this to be flagged.
        :return: the principal components (eta)
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
        Calculate the principal components (or reconstructed variables)

        Example:
            eta = pca.eta2x(x) : calculate the principal components
            xrec = pca.eta2x(eta) : calculate reconstructed variables

        :param eta: the PCs
        :return: the unscaled, uncentered approximation to the data (X)
        """
        npts, neta = eta.shape
        assert neta == self.neta, "Number of variables provided inconsistent with number of PCs."
        A = self.Q[:, 0:neta]
        x = eta.dot(A.transpose())
        return inv_center_scale(x, self.XCenter, self.XScale)

    def calculate_r2(self, X):
        """
        Calculates R-squared values.

        r2 = pca.calculate_r2( X )

        Given the data used to construct the PCA, this calculates the R2 values
        for the reduced representation of the data.  If all of the eigenvalues
        are retained, then this should be unity.

        :param X: data used to construct the PCA
        :return: R2 values for the reduced representation of the data (r2)
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
        Checks if the supplied data matrix X is consistent with the PCA object

        pca.data_consistency_check( X, errorsAreFatal )

        :param X: the independent variables
        :param errorsAreFatal: (OPTIONAL) flag indicating if an error should be raised
%                               if an incompatibility is detected - default is True
        :return: boolean for whether or not supplied data matrix X is consistent with the PCA object (okay)
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
        Print r2 values as a function of number of retained eigenvalues.

        pca.convergence( X, nmax )
        pca.convergence( X, nmax, names )

        :param X: the original dataset
        :param nmax: the maximum number of PCs to consider
        :param names: (OPTIONAL) the names of the variables - otherwise variables are numbered
        :param printwidth: (OPTIONAL) width of columns printed out
        :return: [nmax,nvar] matrix containing the R^2 values for each variable as a
                 function of the number of retained eigenvalues.

        Example:
            pca.convergence(X,5) prints R^2 values retaining 1-5 eigenvalues
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

        pca.eig_bar_plot_maker( neig )

        :param neig: Number of eigenvectors that you want to keep in the plot
        :param DataName: list containing the names of the variables
        :param barWidth: (OPTIONAL) width of each bar in the plot
        :param plotABS: (OPTIONAL) default False - plots the eigenvectors keeping their sign
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
        Plot the eigenvalues (bars) and the cumulative sum (line) to visualize the percent variance in the data
        explained by each principal component individually and by each principal component cumulatively

        :param npc: (OPTIONAL) how many principal components you want to visualize (default is all)
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

        Example:
          ikeep = principal_variables()
          ikeep = principal_variables('B4')
          ikeep = principal_variables('M2', X )

        :param method: [OPTIONAL] the method for determining the principal variables.
                       The following methods are currently supported:
                    "B4" : selects principal variables based on the variables
                           contained in the eigenvectors corresponding to the
                           largest eigenvalues.
                    "B2" : selects pvs based on variables contained in the smallest
                           eigenvalues.  These are discarded and the remaining
                           variables are used as the principal variables.  This is
                           the default method.
                    "M2" : At each iteration, each remaining variable is analyzed
                           via PCA.  This is a very expensive method.

        :param x: [OPTIONAL] data arranged with observations in rows and
                  variables in columns.  Note that this is only required for the
                  "M2" method.
        :return: a vector of indices of retained variables (ikeep)
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

        Examples:
            r2, neta = pca.r2converge( data )
            r2, neta = pca.r2converge( data, names, 'r2.csv' )

        :param data: the data to fit
        :param names: [optional] names of the data
        :param fname: [optional] file to output r2 information to
        :return: r2 - [neta,nvar] The r2 values.  Each column is a different variable and
%                     each row is for a different number of retained pcs.
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
        for reading into C++

        Example:
          pca = PCA( x );
          pca.wite2file('pcaData.txt');

        :param filename: path (including name of text file) for destination of data file
        :return: (creates the .txt file in the destination specified by filename)

        NOTE: This function writes only the eigenvector matrix, centering and
        scaling factors - not all of the pca properties
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
        Help determine how many eigenvalues to retain

        Example:
            pca = pca.set_retained_eigenvalues( method )

        This function provides a few methods to select the number of eigenvalues
        to be retained in the PCA reduction.

        :param method: (optional) method to use in selecting retained eigenvalues.
                       Default is 'SCREE GRAPH'
        :param option: (optional) if not supplied, information will be obtained
                       interactively.  Only used for the 'TOTAL VARIANCE' and
                       'INDIVIDUAL VARIANCE' methods.
        :return: the PCA object with the number of retained eigenvalues set on it. (pca)

        The following methods are available:
        'TOTAL VARIANCE'      retain the eigenvalues needed to account for a
                              specific percentage of the total variance (i.e.
                              80%). The required number of PCs is then the
                              smallest value of m for which this chosen
                              percentage is exceeded.

        'INDIVIDUAL VARIANCE' retain the components whose eigenvalues are
                              greater than the average of the eigenvalues
                              (Kaiser, 1960) or than 0.7 times he average of the
                              eigenvalues (Joliffe 1972). For a correlation
                              matrix this average equals 1.

        'BROKEN STICK'        select the retained PCs according to the
                              Broken Stick Model.

        'SCREE GRAPH'         use the scree graph, a plot of the eigenvalues
                              agaist their indexes, and look for a natural break
                              between the large and small eigenvalues.
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
        Calculate the u scores (principal components)

        Example:
            uscores = pca.u_scores(X)

        U-scores = obtained by using the U-vectors, i.e. the eigenvectors of the
        covariance matrix S. The resulting U-scores are uncorrelated and have
        variances equal to the corresponding eigenvalues.

        This is entirely equivalent to x2eta.

        :param X: a set of observations of variables x (observations in rows),
                  unscaled, uncentered. These do not need to be the same
                  observations as were used to construct the PCA object. They
                  could be, e.g. functions of those variables.
        :return:  u scores or principal components (eta)
        """
        return self.x2eta(X)

    def w_scores(self, X):
        """
        Calculates the w scores

        Example:
            wscores = pca.w_scores( X )

        W-scores = The U vectors are scaled by the inverse of the eigenvalues
        square root, i.e. V = L^-0.5 * U. The W-scores are still uncorrelated and
        have variances equal unity.

        :param X: a set of observations of variables x (observations in rows),
                  unscaled, uncentered. These do not need to be the same
                  observations as were used to construct the PCA object. They
                  could be, e.g. functions of those variables.
        :return:  u scores or principal components (eta)
        :return: w scores
        """
        eval = self.L[0:self.neta]
        return self.x2eta(X).dot(np.diag(1 / np.sqrt(eval)))

    def __eq__(a, b):
        """
        Compares two PCA objects for equality.
        :param a: first PCA object
        :param b: second PCA object
        :return: boolean for (a == b)
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
        :param a: first PCA object
        :param b: second PCA object
        :return: boolean for (a != b)
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
    PHI, cntr, scl = center_scale(PHI.T, 'NONE')

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
