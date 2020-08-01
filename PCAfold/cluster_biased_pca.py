import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
from PCAfold import pca_impl as P
from PCAfold import clustering_data as cld
from PCAfold import sampling
#
# def plotting_styles(func):
#
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

# @plotting_styles
def analyze_centers_movement(X, idx_X_r, variable_names=[], plot_variables=[], title=None, save_filename=None):
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

    if title != None:
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

    if save_filename != None:
        plt.savefig(save_filename, dpi = 500, bbox_inches='tight')

    return (norm_centers_X, norm_centers_X_r, center_movement_percentage)

def analyze_eigenvector_weights_movement(eigenvectors, variable_names, plot_variables=[], normalize=False, zero_norm=False, title=None, save_filename=None):
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
    :param title: (optional)
        boolean or string specifying plot title. If set to ``False``, title will
        not be plotted.
    :param save_filename: (optional)
        plot save location/filename.

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
    color_range = np.arange(0, n_versions)

    # Plot the eigenvector weights movement:
    fig, ax = plt.subplots(figsize=(n_variables, 6))

    for idx, variable in enumerate(variable_names):
        scat = ax.scatter(np.repeat(idx, n_versions), eigenvectors[idx,:], c=color_range, cmap=plt.cm.Spectral)

    plt.xticks(x_range, variable_names, fontsize=font_annotations, **csfont)

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

    if save_plot != None:
        plt.savefig(save_filename, dpi = 500, bbox_inches='tight')

    return

def analyze_eigenvalue_distribution(X, idx_matrix, k_list, scaling, biasing_option, random_seed=None, title=None, save_filename=None):
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
    :param random_seed: (optional)
        integer specifying random seed for random sample selection.
    :param title: (optional)
        boolean or string specifying plot title. If set to ``False``,
        title will not be plotted.
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

    if save_plot == True:
        plt.savefig(save_filename + '.png', dpi = 500, bbox_inches='tight')

    return(min_at_q2_k, min_at_q3_k, max_at_q2_k, max_at_q3_k)

def equilibrate_cluster_populations(X, idx, scaling, n_components, biasing_option, X_source=[], n_iterations=10, stop_iter=0, random_seed=None, verbose=False):
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
        See documentation of cluster-biased PCA for more information.
        Can only attain values [1,2,3,4,5].
    :param n_iterations: (optional)
        number of iterations to loop over.
    :param stop_iter: (optional)
        index of iteration to stop.
    :param random_seed: (optional)
        integer specifying random seed for random sample selection.
    :param verbose: (optional)
        boolean for printing verbose details.

    :raises ValueError:
        if ``biasing_option`` is not 1, 2, 3, 4 or 5.

    :raises ValueError:
        if ``random_seed`` is not an integer.

    :return:
        - **eigenvalues** - collected eigenvalues from each iteration.
        - **eigenvectors_matrix** - collected eigenvectors from each iteration. This is a 3D array of size ``(n_variables, n_components, n_iterations+1)``.
        - **pc_scores_matrix** - collected PC scores from each iteration. This is a 3D array of size ``(n_observations, n_components, n_iterations+1)``.
        - **pc_sources_matrix** - collected PC sources from each iteration. This is a 3D array of size ``(n_observations, n_components, n_iterations+1)``.
        - **idx_train** - the final training indices from the equilibrated iteration.
        - **X_center** - a vector of final centers that were used to center the data set at the last (equlibration) iteration.
        - **X_scale** - a vector of final scales that were used to scale the data set at the last (equlibration) iteration.
    """

    # Check that `biasing_option` parameter was passed correctly:
    _biasing_options = [1,2,3,5]
    if biasing_option not in _biasing_options:
        raise ValueError("Option can only be 1-5. Option 4 is temporarily removed.")

    if random_seed != None:
        if not isinstance(random_seed, int):
            raise ValueError("Random seed has to be an integer.")

    (n_observations, n_variables) = np.shape(X)
    populations = cld.get_populations(idx)
    N_smallest_cluster = np.min(populations)
    k = len(populations)

    # Initialize matrices:
    eigenvectors_matrix = np.zeros((n_variables,n_components,n_iterations+1))
    pc_scores_matrix = np.zeros((n_observations,n_components,n_iterations+1))
    pc_sources_matrix = np.zeros((n_observations,n_components,n_iterations+1))
    eigenvalues = np.zeros((n_variables, 1))
    idx_train = []

    # Perform global PCA on the original data set X: ---------------------------
    pca_global = P.PCA(X, scaling, n_components, useXTXeig=True)

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

        sampling_object = sampling.TrainTestSelect(idx, bar50=False, random_seed=random_seed)
        (idx_train, _) = sampling_object.manual(sampling_dictionary, sampling_type='number')

        # Biasing option 1 -----------------------------------------------------
        if biasing_option == 1:

            # Generate the reduced data set X_r:
            X_r = X[idx_train,:]

            # Perform PCA on X_r:
            pca = P.PCA(X_r, scaling, n_components, useXTXeig=True)

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
            pca = P.PCA(X_r, 'none', n_components, useXTXeig=True, nocenter=True)

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
            pca = P.PCA(X_r, scaling, n_components, useXTXeig=True)
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

            if len(X_source) != 0:

                # Sample the sources using the same idx:
                X_source_r = X_source[idx_train,:]

            # Perform PCA on X_r:
            pca = P.PCA(X_r, scaling, n_components, useXTXeig=True)
            X_center = pca.XCenter
            X_scale = pca.XScale

            # Compute local eigenvectors:
            eigenvectors = pca.Q

            # Compute local eigenvalues:
            local_eigenvalues = pca.L
            maximum_local_eigenvalue = np.max(local_eigenvalues)

            # Compute local PC-scores:
            pc_scores = pca.x2eta(X_r, nocenter=False)

            # Append the local PC-scores:
            pc_scores_matrix[:,:,iter+1] = pc_scores

            if len(X_source) != 0:

                # Compute local PC-sources:
                pc_sources = pca.x2eta(X_source_r, nocenter=True)

                # Append the global PC-sources:
                pc_sources_matrix[:,:,iter+1] = pc_sources

        # Biasing option 5 -----------------------------------------------------
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
            pca = P.PCA(X_cs, 'none', n_components, useXTXeig=True, nocenter=True)

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

def resample_at_equilibration_with_kmeans_on_pc_sources(X, X_source, scaling, biasing_option, n_clusters, n_components, n_resamples=20, idx_all=True, random_seed=None, verbose=False):
    """
    This function performs re-sampling using K-Means clustering on
    ``n_components`` first PC-sources at equilibration step. Re-sampling is done
    until convergence of cluster centroids is reached but for a maximum of
    ``n_resamples`` times. At each step the current ``idx`` containing
    cluster classifications is saved in the global ``idx_matrix``.

    :param X:
        original (full) data set.
    :param X_source:
        source terms corresponding to the state-space variables in ``X``.
    :param scaling:
        data scaling criterion.
    :param biasing_option:
        integer specifying biasing option.
        See documentation of cluster-biased PCA for more information.
        Can only attain values [1,2,3,4,5].
    :param n_clusters:
        number of clusters to use for K-Means partitioning.
    :param n_components:
        number of Principal Components that will be used (this directly
        translates to how many first PC-sources the partitioning is based on).
    :param n_resamples: (optional)
        number of times that the re-sampling will be performed.
    :param idx_all: (optional)
        boolean specifying whether all ``idx`` vectors should be returned (``idx_all=True``) or only the last one (``idx_all=False``).
    :param random_seed: (optional)
        integer specifying random seed for random sample selection.
    :param verbose: (optional)
        boolean for printing verbose details.

    :raises ValueError:
        if ``biasing_option`` is not 1, 2, 3, 4 or 5.

    :raises ValueError:
        if ``random_seed`` is not an integer.

    :return:
        - **idx_matrix** (returned if ``idx_all=True``) - matrix of collected cluster classifications. This is a 2D array of size ``(n_observations, n_resamples+1)``.
        - **idx** (returned if ``idx_all=False``) - vector of cluster classifications from the last re-sampling step. It is the same as ``idx_matrix[:,-1]``.
        - **converged** - boolean specifying whether the re-sampling algorithm have converged based on cluster centroids movement.
    """

    # Check that `biasing_option` parameter was passed correctly:
    _biasing_options = [1,2,3,5]
    if biasing_option not in _biasing_options:
        raise ValueError("Option can only be 1-5. Option 4 is temporarily removed.")

    if random_seed != None:
        if not isinstance(random_seed, int):
            raise ValueError("Random seed has to be an integer.")

    (n_observations, n_variables) = np.shape(X)

    centroids_threshold = 0.01
    converged = False

    # Initialize idx_matrix:
    idx_matrix = np.zeros((n_observations, n_resamples+1))

    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    from numpy import linalg

    # Perform global PCA to obtain initial PC-sources:
    pca_global = P.PCA(X, scaling, n_components, useXTXeig=True)

    # Compute initial PC-sources:
    global_pc_sources = pca_global.x2eta(X_source, nocenter=True)

    # Make the initial clustering with K-Means based on n_components first PC-sources:
    scaler = StandardScaler()
    global_pc_sources_pp = scaler.fit_transform(global_pc_sources[:,0:n_components])
    kmeans = KMeans(n_clusters=n_clusters).fit(global_pc_sources_pp)
    idx = kmeans.labels_
    idx_matrix[:,0] = idx

    current_centroids = cld.get_centroids(X, idx)
    current_centroids = np.divide(current_centroids, linalg.norm(current_centroids))

    for iter in range(0,n_resamples):

        old_centroids = current_centroids

        (_, _, _, pc_sources_matrix, _, _, _) = equilibrate_cluster_populations(X, idx, scaling, X_source=X_source, n_components=n_components, biasing_option=biasing_option, n_iterations=1, stop_iter=0, random_seed=random_seed, verbose=verbose)
        scaler = StandardScaler()
        current_equilibrated_pc_sources_pp = scaler.fit_transform(pc_sources_matrix[:,:,-1])
        kmeans = KMeans(n_clusters=n_clusters).fit(current_equilibrated_pc_sources_pp)
        idx = kmeans.labels_
        idx_matrix[:,iter+1] = idx

        current_centroids = cld.get_centroids(X, idx)
        current_centroids = np.divide(current_centroids, linalg.norm(current_centroids))

        distance_between_centroids = linalg.norm((current_centroids - old_centroids))

        if verbose==True:
            print('Current norm of the centroids difference: ' + str(round(distance_between_centroids, 5)))

        if distance_between_centroids <= centroids_threshold:
            converged = True
            print('Centroids have converged. Norm of the centroids difference: ' + str(round(distance_between_centroids, 5)))
            break

    if idx_all:
        return(idx_matrix, converged)
    else:
        return(idx, converged)

def resample_at_equilibration_with_kmeans_on_pc_scores(X, scaling, biasing_option, n_clusters, n_components, n_resamples=20, idx_all=True, random_seed=None, verbose=False):
    """
    This function performs re-sampling using K-Means clustering on
    ``n_components`` first PC-scores at equilibration step. Re-sampling is done
    until convergence of cluster centroids is reached but for a maximum of
    ``n_resamples`` times. At each step the current ``idx`` containing
    cluster classifications is saved in the global ``idx_matrix``.

    :param X:
        original (full) data set.
    :param scaling:
        data scaling criterion.
    :param biasing_option:
        integer specifying biasing option.
        See documentation of cluster-biased PCA for more information.
        Can only attain values [1,2,3,4,5].
    :param n_clusters:
        number of clusters to use for K-Means partitioning.
    :param n_components:
        number of Principal Components that will be used (this directly
        translates to how many first PC-scores the partitioning is based on).
    :param n_resamples: (optional)
        number of times that the re-sampling will be performed.
    :param idx_all: (optional)
        boolean specifying whether all ``idx`` vectors should be returned (``idx_all=True``) or only the last one (``idx_all=False``).
    :param random_seed: (optional)
        integer specifying random seed for random sample selection.
    :param verbose: (optional)
        boolean for printing verbose details.

    :raises ValueError:
        if ``biasing_option`` is not 1, 2, 3, 4 or 5.

    :raises ValueError:
        if ``random_seed`` is not an integer.

    :return:
        - **idx_matrix** (returned if ``idx_all=True``) - matrix of collected cluster classifications. This is a 2D array of size ``(n_observations, n_resamples+1)``.
        - **idx** (returned if ``idx_all=False``) - vector of cluster classifications from the last re-sampling step. It is the same as ``idx_matrix[:,-1]``.
        - **converged** - boolean specifying whether the re-sampling algorithm have converged based on cluster centroids movement.
    """

    # Check that `biasing_option` parameter was passed correctly:
    _biasing_options = [1,2,3,5]
    if biasing_option not in _biasing_options:
        raise ValueError("Option can only be 1-5. Option 4 is temporarily removed.")

    if random_seed != None:
        if not isinstance(random_seed, int):
            raise ValueError("Random seed has to be an integer.")

    (n_observations, n_variables) = np.shape(X)

    centroids_threshold = 0.01
    converged = False

    # Initialize idx_matrix:
    idx_matrix = np.zeros((n_observations, n_resamples+1))

    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    from numpy import linalg

    # Perform global PCA to obtain initial PC-scores:
    pca_global = P.PCA(X, scaling, n_components, useXTXeig=True)

    # Compute initial PC-scores:
    global_pc_scores = pca_global.x2eta(X, nocenter=False)

    # Make the initial clustering with K-Means based on n_components first PC-scores:
    scaler = StandardScaler()
    global_pc_scores_pp = scaler.fit_transform(global_pc_scores[:,0:n_components])
    kmeans = KMeans(n_clusters=n_clusters).fit(global_pc_scores_pp)
    idx = kmeans.labels_
    idx_matrix[:,0] = idx

    current_centroids = cld.get_centroids(X, idx)
    current_centroids = np.divide(current_centroids, linalg.norm(current_centroids))

    for iter in range(0,n_resamples):

        old_centroids = current_centroids

        (_, _, pc_scores_matrix, _, _, _, _) = equilibrate_cluster_populations(X, idx, scaling, X_source=[], n_components=n_components, biasing_option=biasing_option, n_iterations=1, stop_iter=0, random_seed=random_seed, verbose=verbose)
        scaler = StandardScaler()
        current_equilibrated_pc_scores_pp = scaler.fit_transform(pc_scores_matrix[:,:,-1])
        kmeans = KMeans(n_clusters=n_clusters).fit(current_equilibrated_pc_scores_pp)
        idx = kmeans.labels_
        idx_matrix[:,iter+1] = idx

        current_centroids = cld.get_centroids(X, idx)
        current_centroids = np.divide(current_centroids, linalg.norm(current_centroids))

        distance_between_centroids = linalg.norm((current_centroids - old_centroids))

        if verbose==True:
            print('Current norm of the centroids difference: ' + str(round(distance_between_centroids, 5)))

        if distance_between_centroids <= centroids_threshold:
            converged = True
            print('Centroids have converged. Norm of the centroids difference: ' + str(round(distance_between_centroids, 5)))
            break

    if idx_all:
        return(idx_matrix, converged)
    else:
        return(idx, converged)
