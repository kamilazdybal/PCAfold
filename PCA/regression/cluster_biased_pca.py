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
font_labels = 20
font_title = 22
font_text = 16
font_legend = 18

def analyze_centers_movement(X, X_r, variable_names=[], save_plot=False, save_filename=''):
    """
    This function analyzed the movement of centers in the subset of the original
    data set `X_r` with respect to the full original data set `X`.

    NOTE: It first normalizes the variables to be in range between 0 and 1, so
    that the means (centers) computed can be compared across variables.

    Input:
    ----------
    `X`           - original (full) data set.
    `X_r`         - reduced data set.
    `variable_names`
                  - list of strings specifying the variable names.
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
    (n_obs_X_r, n_vars_X_r) = np.shape(X_r)

    if len(variable_names) != 0:
        n_vars = len(variable_names)
    else:
        variable_names = ['X_' + str(i) for i in range(0, n_vars_X)]

    if n_vars_X != n_vars or n_vars_X_r != n_vars:
        raise ValueError("The number of observations between X and X_r is not consistent with the variable names.")

    minimums_X = np.min(X, axis=0)
    maximums_X = np.max(X, axis=0)
    X_normalized = (X - minimums_X)/maximums_X
    X_r_normalized = (X_r - minimums_X)/maximums_X

    # Find centers:
    norm_centers_X = np.mean(X_normalized, axis=0)
    norm_centers_X_r = np.mean(X_r_normalized, axis=0)

    # Compute the relative percentage by how much the center has moved:
    center_movement_percentage = abs((norm_centers_X - norm_centers_X_r))/norm_centers_X * 100

    x_range = np.arange(1, n_vars+1)

    fig, ax = plt.subplots(figsize=(20, 6))

    plt.scatter(x_range, norm_centers_X, c=color_X, marker='o', s=marker_size, edgecolor='none', alpha=1)
    plt.scatter(x_range, norm_centers_X_r, c=color_X_r, marker='>', s=marker_size, edgecolor='none', alpha=1)

    plt.xticks(x_range, variable_names, fontsize=font_labels)
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
