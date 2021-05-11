# ------------------------------------------------------------------------------
#
# This file sets plotting parameters and styles that will be used throughout
# PCAfold modules.
#
# You can modify this file with your own settings.
#
# ------------------------------------------------------------------------------

from matplotlib import rcParams

# Set resolution for the saved plots (in dpi - dots-per-inch):
save_dpi = 100

# Set font styles:
csfont = {'fontname':'Charter', 'fontweight':'regular'}
hfont = {'fontname':'Charter', 'fontweight':'bold'}
ifont = {'fontname':'Charter', 'fontweight':'regular', 'style':'italic'}
rcParams["font.family"] = "serif"
rcParams["font.serif"] = "Charter"
rcParams["font.sans-serif"] = "Charter"
rcParams["font.cursive"] = "Charter"
rcParams["font.monospace"] = "Charter"
rcParams['mathtext.fontset'] = 'custom'
rcParams['mathtext.rm'] = 'Charter'
rcParams['mathtext.it'] = 'Charter:italic'
rcParams['mathtext.bf'] = 'Charter:bold'
rcParams['font.size'] = 16
rcParams["text.usetex"] = False

# Set grid opacity:
grid_opacity = 0.3

# Set font size for axis ticks:
font_axes = 16

# Set font size for axis labels:
font_labels = 24

# Set font size for
font_annotations = 20

# Set font size for plot title:
font_title = 18

# Set font size for plotted text:
font_text = 16

# Set font size for legend entries:
font_legend = 16

# Set font size for colorbar axis label:
font_colorbar = 24

# Set font size for colorbar axis ticks:
font_colorbar_axes = 18

# Set marker size for all line markers:
marker_size = 50

# Set the scale for marker size plotted in the legend entries:
marker_scale_legend = 1

# Set the scale for marker size plotted in the legend entries of clustering plots:
marker_scale_legend_clustering = 5

# Set point size for all scatter plots:
scatter_point_size = 2

# Set line width for all line plots:
line_width = 1

# Set bar width for plotting eigenvector weights:
eigenvector_bar_width = 0.4
