# ------------------------------------------------------------------------------
#
# This file sets plotting parameters and styles that will be used throughout
# PCAfold modules.
# You can replace this file or its portion with your own settings.
#
# ------------------------------------------------------------------------------

from matplotlib import rcParams

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
rcParams['font.size'] = 20
rcParams["text.usetex"] = False

font_axes = 20
font_labels = 24
font_annotations = 20
font_title = 18
font_text = 16
font_legend = 20
font_colorbar = 24

marker_size = 50
marker_scale_legend = 50
