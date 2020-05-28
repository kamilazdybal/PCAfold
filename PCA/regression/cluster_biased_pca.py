import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

# Plotting parameters:
csfont = {'fontname':'Charter', 'fontweight':'regular'}
hfont = {'fontname':'Charter', 'fontweight':'bold'}
ifont = {'fontname':'Charter', 'fontweight':'regular', 'style':'italic'}

font_axes = 18
font_labels = 24
font_annotations = 18
font_title = 22
font_text = 16
font_legend = 20

def analyze_centers_movement(X, idx_X_r, variable_names=[], save_plot=False, save_filename=''):
    """
    This function analyzes the movement of centers in the subset of the original
    data set `X_r` with respect to the full original data set `X`.

    NOTE: It first normalizes the variables to be in range between 0 and 1, so
    that the means (centers) computed can be compared across variables.

    Input:
    ----------
    `X`           - original (full) data set.
    `idx_X_r`     - indices that should be extracted from `X` to form `X_r`.
                    It could be obtained as training indices from
                    `training_data_generation` module.
    `variable_names`
                  - list of strings specifying variable names.
    `save_plot`   - boolean specifying whether the plot should be saved.
    `save_filename`
                  - plot save location/filename.

    Output:
    ----------
    `norm_centers_X`
                  - normalized centers of the original (full) data set `X`.
    `norm_centers_X_r`
                  - normalized centers of the reduced data set `X_r`.
    `center_movement_percentage`
                  - relative percentage specifying how the center has moved
                    between `X` and `X_r`. The movement is measured relative to
                    the original (full) data set `X`.
    """

    color_X = '#191b27'
    color_X_r = '#ff2f18'
    marker_size = 50

    (n_obs_X, n_vars_X) = np.shape(X)

    if len(variable_names) != 0:
        n_vars = len(variable_names)
    else:
        variable_names = ['X_' + str(i) for i in range(0, n_vars_X)]

    X_normalized = (X - np.min(X, axis=0))
    X_normalized = X_normalized / np.max(X_normalized, axis=0)

    # Extract X_r using the provided idx_X_r:
    X_r_normalized = X_normalized[idx_X_r,:]

    # Find centers:
    norm_centers_X = np.mean(X_normalized, axis=0)
    norm_centers_X_r = np.mean(X_r_normalized, axis=0)

    # Compute the relative percentage by how much the center has moved:
    center_movement_percentage = (norm_centers_X_r - norm_centers_X) / norm_centers_X * 100

    x_range = np.arange(1, n_vars+1)

    fig, ax = plt.subplots(figsize=(n_vars, 6))

    plt.scatter(x_range, norm_centers_X, c=color_X, marker='o', s=marker_size, edgecolor='none', alpha=1)
    plt.scatter(x_range, norm_centers_X_r, c=color_X_r, marker='>', s=marker_size, edgecolor='none', alpha=1)

    plt.xticks(x_range, variable_names, fontsize=font_annotations)
    plt.ylabel('Normalized center [-]', fontsize=font_labels)
    plt.ylim(0,1)

    for i, value in enumerate(center_movement_percentage):
        plt.text(i+1.05, norm_centers_X_r[i]+0.01, str(int(value)) + ' %', fontsize=font_text, c=color_X_r)

    ax.spines["top"].set_visible(True)
    ax.spines["bottom"].set_visible(True)
    ax.spines["right"].set_visible(True)
    ax.spines["left"].set_visible(True)

    lgnd = plt.legend(['Original data set', 'Reduced data set'], fontsize=font_legend, markerscale=50, loc="upper right")

    lgnd.legendHandles[0]._sizes = [marker_size]
    lgnd.legendHandles[1]._sizes = [marker_size]

    if save_plot == True:
        plt.savefig(save_filename + '.png', dpi = 500, bbox_inches='tight')

    return (norm_centers_X, norm_centers_X_r, center_movement_percentage)

def analyze_eigenvector_weights_movement(eigenvector_matrix, variable_names, plot_variables=[], save_plot=False, save_filename=''):
    """
    This function analyzes the movement of weights of variables on a single
    eigenvector when PCA is performed on different versions of the reduced data
    sets `X_r`.

    The `eigenvector_matrix` should be formed in the following way:

                 T Y_1 Y_2  ...  Y_n
                [                   ] PC-i on a biased set X_r(1)
                [                   ] PC-i on a biased set X_r(2)
                [                   ] .
                [                   ] .
                [                   ] PC-i on a biased set X_r(j)
                [                   ] .
                [                   ] PC-i on a biased set X_r(N)

    Each row is a selected i-th eigenvector (for instance PC-1) computed by
    performing PCA on a specific (j)-th version of the reduced data set.
    Each column are changing weights on that eigenvector of a particular
    variable in a data set.

    NOTE: This function plots absolute, normalized values of weights on each
    variable. Columns are normalized by dividing by the maximum value. This is
    done in order to compare the movement of weights equally, with the highest,
    normalized one being equal to 1.0.

    Input:
    ----------
    `eigenvector_matrix`
                  - matrix of concatenated eigenvectors coming from either
                    different data sets or from different iterations.
    `variable_names`
                  - list of strings specifying variable names.
    `plot_variables`
                  - list of integers specifying indices of variables to be plotted.
                    By default, all variables are plotted.
    `save_plot`   - boolean specifying whether the plot should be saved.
    `save_filename`
                  - plot save location/filename.

    Output:
    ----------

    """

    (n_versions, n_vars) = np.shape(eigenvector_matrix)

    # Check that the number of columns in the eigenvector matrix is equal to the
    # number of elements in the `variable_names` vector:
    if len(variable_names) != n_vars:
        raise ValueError("The number of variables in the eigenvector matrix is not equal to the number of variable names.")

    if len(plot_variables) != 0:

        eigenvector_matrix = eigenvector_matrix[:,plot_variables]
        variable_names = [variable_names[i] for i in plot_variables]
        (_, n_vars) = np.shape(eigenvector_matrix)

    # Normalize each column inside `eigenvector_weights`:
    eigenvector_matrix = np.divide(np.abs(eigenvector_matrix), np.max(np.abs(eigenvector_matrix), 0))

    x_range = np.arange(0,n_vars)
    color_range = np.arange(1, n_versions+1)

    # Plot the eigenvector weights movement:
    fig, ax = plt.subplots(figsize=(n_vars, 6))

    for idx, variable in enumerate(variable_names):
        scat = ax.scatter(np.repeat(idx, n_versions), eigenvector_matrix[:,idx], c=color_range, cmap=plt.cm.Spectral)

    plt.xticks(x_range, variable_names, fontsize=font_annotations)
    plt.ylabel('Normalized weight [-]', fontsize=font_labels)
    plt.ylim(-0.05,1.05)
    plt.xlim(-1, n_vars)
    plt.grid(alpha=0.3)
    cbar = plt.colorbar(scat, ticks=[1, round(n_versions/2), n_versions])

    if save_plot == True:
        plt.savefig(save_filename + '.png', dpi = 500, bbox_inches='tight')

    return
