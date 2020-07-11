import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import PCA.PCA as P
import PCA.clustering_data as cl
import PCA.train_test_select as tts

def plotting_styles():

    from matplotlib import rcParams

    # Plotting parameters:
    csfont = {'fontname':'Charter', 'fontweight':'regular'}
    hfont = {'fontname':'Charter', 'fontweight':'bold'}
    ifont = {'fontname':'Charter', 'fontweight':'regular', 'style':'italic'}
    rcParams["font.family"] = "serif"
    rcParams["font.serif"] = "Charter"
    rcParams['font.size'] = 16

    font_axes = 18
    font_labels = 24
    font_annotations = 18
    font_title = 18
    font_text = 16
    font_legend = 20
    font_colorbar = 16

def analyze_centers_movement(X, idx_X_r, variable_names=[], plot_variables=[], title=False, save_plot=False, save_filename=''):
    """
    This function analyzes the movement of centers in the subset of the original
    data set ``X_r`` with respect to the full original data set ``X``.

    It returns the normalized centers of the original data set
    ``norm_centers_X`` and normalized centers of the reduced data set
    ``norm_centers_X_r``. It also returns the percentage of the centers movement
    ``center_movement_percentage`` (the same as is plotted in the figure).

    *Note:*
    The original data set ``X`` is first normalized so that each variable ranges
    from 0 to 1. Samples are then extracted from the normalized data set to form
    ``X_r``. The normalization is done so that centers can be compared across
    variables on one plot.

    :param X:
        original (full) data set.
    :param idx_X_r:
        vector of indices that should be extracted from ``X`` to form ``X_r``.
        It could be obtained as training indices from
        ``training_data_generation`` module.
    :param variable_names: (optional)
        list of strings specifying variable names.
    :param plot_variables: (optional)
        list of integers specifying indices of variables to be plotted.
        By default, all variables are plotted.
    :param title: (optional)
        boolean or string specifying plot title. If set to ``False``,
        title will not be plotted.
    :param save_plot: (optional)
        boolean specifying whether the plot should be saved.
    :param save_filename: (optional)
        plot save location/filename.

    :return:
        - **norm_centers_X** - normalized centers of the original (full) data set ``X``.
        - **norm_centers_X_r** - normalized centers of the reduced data set ``X_r``.
        - **center_movement_percentage** - relative percentage specifying how the center has moved between ``X`` and ``X_r``. The movement is measured relative to the original (full) data set ``X``.
    """

    color_X = '#191b27'
    color_X_r = '#ff2f18'
    marker_size = 50

    (n_observations_X, n_variables_X) = np.shape(X)

    if len(variable_names) != 0:
        n_variables = len(variable_names)
    else:
        variable_names = ['X_' + str(i) for i in range(0, n_variables_X)]

    if len(plot_variables) != 0:
        X = X[:,plot_variables]
        variable_names = [variable_names[i] for i in plot_variables]
        (_, n_variables) = np.shape(X)

    X_normalized = (X - np.min(X, axis=0))
    X_normalized = X_normalized / np.max(X_normalized, axis=0)

    # Extract X_r using the provided idx_X_r:
    X_r_normalized = X_normalized[idx_X_r,:]

    # Find centers:
    norm_centers_X = np.mean(X_normalized, axis=0)
    norm_centers_X_r = np.mean(X_r_normalized, axis=0)

    # Compute the relative percentage by how much the center has moved:
    center_movement_percentage = (norm_centers_X_r - norm_centers_X) / norm_centers_X * 100

    x_range = np.arange(1, n_variables+1)

    fig, ax = plt.subplots(figsize=(n_variables, 6))

    plt.scatter(x_range, norm_centers_X, c=color_X, marker='o', s=marker_size, edgecolor='none', alpha=1)
    plt.scatter(x_range, norm_centers_X_r, c=color_X_r, marker='>', s=marker_size, edgecolor='none', alpha=1)

    plt.xticks(x_range, variable_names, fontsize=font_annotations, **csfont)
    plt.ylabel('Normalized center [-]', fontsize=font_labels, **csfont)
    plt.ylim(-0.05,1.05)
    plt.xlim(0, n_variables+1.5)
    plt.grid(alpha=0.3)

    if title != False:
        plt.title(title, fontsize=font_title, **csfont)

    for i, value in enumerate(center_movement_percentage):
        plt.text(i+1.05, norm_centers_X_r[i]+0.01, str(int(value)) + ' %', fontsize=font_text, c=color_X_r, **csfont)

    ax.spines["top"].set_visible(True)
    ax.spines["bottom"].set_visible(True)
    ax.spines["right"].set_visible(True)
    ax.spines["left"].set_visible(True)

    lgnd = plt.legend(['Original data set', 'Reduced data set'], fontsize=font_legend, markerscale=50, loc="upper right")

    lgnd.legendHandles[0]._sizes = [marker_size]
    lgnd.legendHandles[1]._sizes = [marker_size]
    plt.setp(lgnd.texts, **csfont)

    if save_plot == True:
        plt.savefig(save_filename + '.png', dpi = 500, bbox_inches='tight')

    return (norm_centers_X, norm_centers_X_r, center_movement_percentage)

def analyze_eigenvector_weights_movement(eigenvector_matrix, variable_names, plot_variables=[], normalize=False, zero_norm=False, title=False, save_plot=False, save_filename=''):
    """
    This function analyzes the movement of weights on an eigenvector obtained
    from a reduced data set at each iteration. The color-coding marks the
    iteration number. If there is a consistent trend, the coloring should form
    a clear trajectory. The zero-th iteration corresponds to eigenvectors found
    on the original data set ``X``. The last iteration corresponds to eigenvectors
    found on the "equilibrated" data set.

    *Note:*
    This function plots absolute, (and optionally normalized) values of weights on each
    variable. Columns are normalized dividing by the maximum value. This is
    done in order to compare the movement of weights equally, with the highest,
    normalized one being equal to 1. You can additionally set the
    ``zero_norm=True`` in order to normalize weights such that they are between
    0 and 1 (this is not done by default).

    :param eigenvector_matrix:
        matrix of concatenated eigenvectors coming from different data sets or
        from different iterations.
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
    :param title: (optional)
        boolean or string specifying plot title. If set to ``False``, title will
        not be plotted.
    :param save_plot: (optional)
        boolean specifying whether the plot should be saved.
    :param save_filename: (optional)
        plot save location/filename.

    :raises ValueError:
        if the number of variables in ``variable_names`` list does not
        correspond to variables in the ``eigenvectors_matrix``.
    """

    (n_versions, n_variables) = np.shape(eigenvector_matrix)

    # Check that the number of columns in the eigenvector matrix is equal to the
    # number of elements in the `variable_names` vector:
    if len(variable_names) != n_variables:
        raise ValueError("The number of variables in the eigenvector matrix is not equal to the number of variable names.")

    if len(plot_variables) != 0:

        eigenvector_matrix = eigenvector_matrix[:,plot_variables]
        variable_names = [variable_names[i] for i in plot_variables]
        (_, n_variables) = np.shape(eigenvector_matrix)

    # Normalize each column inside `eigenvector_weights`:
    if normalize == True:
        if zero_norm == True:
            eigenvector_matrix = np.abs(eigenvector_matrix) - np.min(np.abs(eigenvector_matrix), 0)

        eigenvector_matrix = np.divide(np.abs(eigenvector_matrix), np.max(np.abs(eigenvector_matrix), 0))
    else:
        eigenvector_matrix = np.abs(eigenvector_matrix)

    x_range = np.arange(0,n_variables)
    color_range = np.arange(0, n_versions)

    # Plot the eigenvector weights movement:
    fig, ax = plt.subplots(figsize=(n_variables, 6))

    for idx, variable in enumerate(variable_names):
        scat = ax.scatter(np.repeat(idx, n_versions), eigenvector_matrix[:,idx], c=color_range, cmap=plt.cm.Spectral)

    plt.xticks(x_range, variable_names, fontsize=font_annotations, **csfont)

    if normalize == True:
        plt.ylabel('Normalized weight [-]', fontsize=font_labels, **csfont)
    else:
        plt.ylabel('Absolute weight [-]', fontsize=font_labels, **csfont)

    plt.ylim(-0.05,1.05)
    plt.xlim(-1, n_variables)
    plt.grid(alpha=0.3)

    if title != False:
        plt.title(title, fontsize=font_title, **csfont)

    cbar = plt.colorbar(scat, ticks=[0, round((n_versions-1)/2), n_versions-1])

    if save_plot == True:
        plt.savefig(save_filename + '.png', dpi = 500, bbox_inches='tight')

    return

def analyze_eigenvalue_distribution(X, idx_matrix, k_list, scaling, biasing_option, title=False, save_plot=False, save_filename=''):
    """
    This function analyzes the normalized eigenvalue distribution when PCA is
    performed on different versions of the reduced data sets ``X_r`` vs. on the
    original data set ``X``.

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
        See documentation of cluster-biased PCA for more information.
        Can only attain values [1,2,3,4,5].
    :param title: (optional)
        boolean or string specifying plot title. If set to ``False``,
        title will not be plotted.
    :param save_plot: (optional)
        boolean specifying whether the plot should be saved.
    :param save_filename: (optional)
        plot save location/filename.

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
        (eigenvalues, _, _, _, _, _, _, _) = equilibrate_cluster_populations(X, idx, scaling, biasing_option=biasing_option, n_iterations=1)

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

    if save_plot == True:
        plt.savefig(save_filename + '.png', dpi = 500, bbox_inches='tight')

    return(min_at_q2_k, min_at_q3_k, max_at_q2_k, max_at_q3_k)

def equilibrate_cluster_populations(X, idx, scaling, X_source=[], biasing_option=1, n_iterations=10, stop_iter=0, verbose=False):
    """
    This function gradually equilibrates cluster populations heading towards
    population of the smallest cluster in ``n_iterations``.

    At each iteration it generates the reduced data set ``X_r(i)`` made up from
    new populations, performs PCA on that data set to find the ``i-th`` version of
    the eigenvectors. Depending on the option selected, it then does the
    projection of a data set (and optionally also its sources) onto the found
    eigenvectors.

    :param X:
        original (full) data set.
    :param idx:
        vector of indices classifying observations to clusters.
        The first cluster has index 0.
    :param scaling:
        data scaling criterion.
    :param X_source:
        source terms corresponding to the state-space variables in ``X``.
    :param biasing_option:
        integer specifying biasing option.
        See documentation of cluster-biased PCA for more information.
        Can only attain values [1,2,3,4,5].
    :param n_iterations:
        number of iterations to loop over.
    :param stop_iter:
        number of iteration to stop.
    :param verbose:
        boolean for printing verbose details.

    :raises ValueError:
        if ``biasing_option`` is not 1, 2, 3, 4 or 5.

    :return:
        - **eigenvalues** - collected eigenvalues from each iteration.
        - **eigenvectors** - collected eigenvectors from each iteration.
        - **pc_scores** - collected PC scores from each iteration.
        - **pc_sources** - collected PC-1 sources from each iteration. This variable is only returned if ``X_sources`` was passed as an input parameter.
        - **idx_train** - the final training indices from the equilibrated iteration.
        - **X_center** - a vector of final centers that were used to center the data set at the last (equlibration) iteration.
        - **X_scale** - a vector of final scales that were used to scale the data set at the last (equlibration) iteration.
    """

    # Check that `biasing_option` parameter was passed correctly:
    _biasing_options = [1,2,3,4,5]
    if biasing_option not in _biasing_options:
        raise ValueError("Option can only be 1-5.")

    (n_observations, n_variables) = np.shape(X)
    populations = cl.get_populations(idx)
    N_smallest_cluster = np.min(populations)
    k = len(populations)
    if verbose == True:
        print("The initial cluster populations are:")
        print(populations)

    # Initialize matrices:
    eigenvectors_1 = np.zeros((1, n_variables))
    eigenvectors_2 = np.zeros((1, n_variables))
    eigenvalues = np.zeros((n_variables, 1))
    if biasing_option != 4:
        pc_scores_1 = np.zeros((n_observations,1))
        pc_scores_2 = np.zeros((n_observations,1))
        if len(X_source) != 0:
            pc_sources_1 = np.zeros((n_observations,1))
            pc_sources_2 = np.zeros((n_observations,1))
    else:
        # If biasing_option is 4, we need to use a list because the number of observations will decrease at each iteration:
        pc_scores_1 = []
        pc_scores_2 = []
        if len(X_source) != 0:
            pc_sources_1 = []
            pc_sources_2 = []

    # Perform global PCA on the original data set X:
    pca_global = P.PCA(X, scaling, 2, useXTXeig=True)

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
    eigenvectors_1 = np.vstack((eigenvectors_1, global_eigenvectors[:,0].T))
    eigenvectors_2 = np.vstack((eigenvectors_2, global_eigenvectors[:,1].T))

    # Append the global eigenvalues:
    eigenvalues = np.hstack((eigenvalues, np.reshape(global_eigenvalues, (n_variables, 1))/maximum_global_eigenvalue))

    # Compute global PC-scores:
    global_pc_scores = pca_global.x2eta(X, nocenter=False)

    # Append the global PC-scores:
    if biasing_option != 4:
        pc_scores_1 = np.hstack((pc_scores_1, global_pc_scores[:,0:1]))
        pc_scores_2 = np.hstack((pc_scores_2, global_pc_scores[:,1:2]))
    else:
        pc_scores_1.append(global_pc_scores[:,0:1])
        pc_scores_2.append(global_pc_scores[:,1:2])

    if len(X_source) != 0:

        # Scale sources with the global scalings:
        X_source_cs = np.divide(X_source, X_scale)

        # Compute global PC-sources:
        global_pc_sources = pca_global.x2eta(X_source, nocenter=True)

        # Append the global PC-sources:
        if biasing_option != 4:
            pc_sources_1 = np.hstack((pc_sources_1, global_pc_sources[:,0:1]))
            pc_sources_2 = np.hstack((pc_sources_2, global_pc_sources[:,1:2]))
        else:
            pc_sources_1.append(global_pc_sources[:,0:1])
            pc_sources_2.append(global_pc_sources[:,1:2])

    # Number of observations that should be taken from each cluster at each iteration:
    eat_ups = np.zeros((k,))
    for cluster in range(0,k):
        eat_ups[cluster] = (populations[cluster] - N_smallest_cluster)/n_iterations

    if verbose == True:
        print('Biasing will be performed with option ' + str(biasing_option) + '.')

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

        (idx_train, _) = tts.train_test_split_manual_from_idx(idx, sampling_dictionary, sampling_type='number', bar50=False, verbose=False)

        if biasing_option == 1:

            # Generate the reduced data set X_r:
            X_r = X[idx_train,:]

            # Perform PCA on X_r:
            pca = P.PCA(X_r, scaling, 2, useXTXeig=True)

            # Compute local eigenvectors:
            eigenvectors = pca.Q

            # Compute local eigenvalues:
            local_eigenvalues = pca.L
            maximum_local_eigenvalue = np.max(local_eigenvalues)

            # Compute local PC-scores:
            pc_scores = X_cs.dot(eigenvectors)

            if len(X_source) != 0:

                # Compute local PC-sources:
                pc_sources = X_source_cs.dot(eigenvectors)

                # Append the global PC-sources:
                pc_sources_1 = np.hstack((pc_sources_1, pc_sources[:,0:1]))
                pc_sources_2 = np.hstack((pc_sources_2, pc_sources[:,1:2]))

            # Append the local PC-scores:
            pc_scores_1 = np.hstack((pc_scores_1, pc_scores[:,0:1]))
            pc_scores_2 = np.hstack((pc_scores_2, pc_scores[:,1:2]))

        elif biasing_option == 2:

            # Generate the reduced data set X_r:
            X_r = X_cs[idx_train,:]

            # Perform PCA on X_r:
            pca = P.PCA(X_r, 'none', 2, useXTXeig=True, nocenter=True)

            # Compute local eigenvectors:
            eigenvectors = pca.Q

            # Compute local eigenvalues:
            local_eigenvalues = pca.L
            maximum_local_eigenvalue = np.max(local_eigenvalues)

            # Compute local PC-scores:
            pc_scores = X_cs.dot(eigenvectors)

            if len(X_source) != 0:

                # Compute local PC-sources:
                pc_sources = X_source_cs.dot(eigenvectors)

                # Append the global PC-sources:
                pc_sources_1 = np.hstack((pc_sources_1, pc_sources[:,0:1]))
                pc_sources_2 = np.hstack((pc_sources_2, pc_sources[:,1:2]))

            # Append the local PC-scores:
            pc_scores_1 = np.hstack((pc_scores_1, pc_scores[:,0:1]))
            pc_scores_2 = np.hstack((pc_scores_2, pc_scores[:,1:2]))

        elif biasing_option == 3:

            # Generate the reduced data set X_r:
            X_r = X[idx_train,:]

            # Perform PCA on X_r:
            pca = P.PCA(X_r, scaling, 2, useXTXeig=True)
            X_center = pca.XCenter
            X_scale = pca.XScale

            # Compute local eigenvectors:
            eigenvectors = pca.Q

            # Compute local eigenvalues:
            local_eigenvalues = pca.L
            maximum_local_eigenvalue = np.max(local_eigenvalues)

            # Compute local PC-scores (the original data set will be centered and scaled with the local centers and scales):
            pc_scores = pca.x2eta(X, nocenter=False)

            if len(X_source) != 0:

                # Compute local PC-sources:
                pc_sources = pca.x2eta(X_source, nocenter=True)

                # Append the global PC-sources:
                pc_sources_1 = np.hstack((pc_sources_1, pc_sources[:,0:1]))
                pc_sources_2 = np.hstack((pc_sources_2, pc_sources[:,1:2]))

            # Append the local PC-scores:
            pc_scores_1 = np.hstack((pc_scores_1, pc_scores[:,0:1]))
            pc_scores_2 = np.hstack((pc_scores_2, pc_scores[:,1:2]))

        elif biasing_option == 4:

            # Generate the reduced data set X_r:
            X_r = X[idx_train,:]

            if len(X_source) != 0:

                # Sample the sources using the same idx:
                X_source_r = X_source[idx_train,:]

            # Perform PCA on X_r:
            pca = P.PCA(X_r, scaling, 2, useXTXeig=True)
            X_center = pca.XCenter
            X_scale = pca.XScale

            # Compute local eigenvectors:
            eigenvectors = pca.Q

            # Compute local eigenvalues:
            local_eigenvalues = pca.L
            maximum_local_eigenvalue = np.max(local_eigenvalues)

            # Compute local PC-scores:
            pc_scores = pca.x2eta(X_r, nocenter=False)

            if len(X_source) != 0:

                # Compute local PC-sources:
                pc_sources = pca.x2eta(X_source_r, nocenter=True)

                # Append the global PC-sources:
                pc_sources_1.append(pc_sources[:,0:1])
                pc_sources_2.append(pc_sources[:,1:2])

            # Append the local PC-scores:
            pc_scores_1.append(pc_scores[:,0:1])
            pc_scores_2.append(pc_scores[:,1:2])

        elif biasing_option == 5:

            # Generate the reduced data set X_r:
            X_r = X[idx_train,:]

            # Compute the current centers and scales of X_r:
            (_, C_r, D_r) = P.center_scale(X_r, scaling)
            X_center = C_r
            X_scale = D_r

            # Pre-process the global data set with the current C_r and D_r:
            X_cs = (X - C_r) / D_r

            # Perform PCA on the original data set X:
            pca = P.PCA(X_cs, 'none', 2, useXTXeig=True, nocenter=True)

            # Compute local eigenvectors:
            eigenvectors = pca.Q

            # Compute local eigenvalues:
            local_eigenvalues = pca.L
            maximum_local_eigenvalue = np.max(local_eigenvalues)

            # Compute local PC-scores:
            pc_scores = X_cs.dot(eigenvectors)

            if len(X_source) != 0:

                # Compute local PC-sources:
                pc_sources = X_source_cs.dot(eigenvectors)

                # Append the global PC-sources:
                pc_sources_1 = np.hstack((pc_sources_1, pc_sources[:,0:1]))
                pc_sources_2 = np.hstack((pc_sources_2, pc_sources[:,1:2]))

            # Append the local PC-scores:
            pc_scores_1 = np.hstack((pc_scores_1, pc_scores[:,0:1]))
            pc_scores_2 = np.hstack((pc_scores_2, pc_scores[:,1:2]))

        # Append the local eigenvectors:
        eigenvectors_1 = np.vstack((eigenvectors_1, eigenvectors[:,0].T))
        eigenvectors_2 = np.vstack((eigenvectors_2, eigenvectors[:,1].T))

        # Append the local eigenvalues:
        eigenvalues = np.hstack((eigenvalues, np.reshape(local_eigenvalues, (n_variables, 1))/maximum_local_eigenvalue))

    # Remove the first row of zeros:
    eigenvectors_1 = eigenvectors_1[1::,:]
    eigenvectors_2 = eigenvectors_2[1::,:]

    # Remove the first column of zeros:
    eigenvalues = eigenvalues[:,1::]

    if biasing_option != 4:
        pc_scores_1 = pc_scores_1[:,1::]
        pc_scores_2 = pc_scores_2[:,1::]
        if len(X_source) != 0:
            pc_sources_1 = pc_sources_1[:,1::]
            pc_sources_2 = pc_sources_2[:,1::]

    if len(X_source) != 0:
        return(eigenvalues, eigenvectors_1, eigenvectors_2, pc_scores_1, pc_scores_2, pc_sources_1, pc_sources_2, idx_train, X_center, X_scale)
    else:
        return(eigenvalues, eigenvectors_1, eigenvectors_2, pc_scores_1, pc_scores_2, idx_train, X_center, X_scale)
