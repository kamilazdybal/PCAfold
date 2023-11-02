# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath('../PCAfold/'))

# -- Project information -----------------------------------------------------

project = 'PCAfold'
copyright = '2020-2023, Elizabeth Armstrong, Alessandro Parente, James C. Sutherland, Kamila Zdybał'
author = 'Elizabeth Armstrong, Alessandro Parente, James C. Sutherland, Kamila Zdybał'
release = '2.0.0'

# -- General configuration ---------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
#
# needs_sphinx = '1.0'

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.todo',
    'sphinx.ext.githubpages',
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.autosummary",
    'sphinxcontrib.bibtex',
]

autosectionlabel_prefix_document = True
# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#
# source_suffix = ['.rst', '.md']
source_suffix = '.rst'

# The master toctree document.
master_doc = 'index'

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#
# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
language = 'English'

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path .
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'

# Path to bibliographic references:
bibtex_bibfiles = "docs/user/data-preprocessing.bib"

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
#
# html_theme_options = {}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = []

# Custom sidebar templates, must be a dictionary that maps document names
# to template names.
#
# The default sidebars (for documents that don't match any pattern) are
# defined by theme itself.  Builtin themes are using these templates by
# default: ``['localtoc.html', 'relations.html', 'sourcelink.html',
# 'searchbox.html']``.
#
# html_sidebars = {}
# import sphinx_rtd_theme
#
# html_theme = "sphinx_rtd_theme"
#
# html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]

html_theme = "furo"

html_logo = "images/PCAfold-logo.svg"

html_theme_options = {
    "dark_css_variables": {
        "color-problematic": "#b30000",
        "color-foreground-primary": "black",
        "color-foreground-secondary": "#5a5c63",
        "color-foreground-muted": "#72747e",
        "color-foreground-border": "#878787",
        "color-background-primary": "white",
        "color-background-secondary": "#f8f9fb",
        "color-background-hover": "#efeff4ff",
        "color-background-hover--transparent": "#efeff400",
        "color-background-border": "#eeebee",
        "color-inline-code-background": "#f2f2f2",

        # Announcements
        "color-announcement-background": "#000000dd",
        "color-announcement-text": "#eeebee",

        # Brand colors
        "color-brand-primary": "#2962ff",
        "color-brand-content": "#2a5adf",

        # Highlighted text (search)
        "color-highlighted-background": "#ddeeff",

        # GUI Labels
        "color-guilabel-background": "#ddeeff80",
        "color-guilabel-border": "#bedaf580",

        # API documentation
        "color-api-highlight-on-target": "#ffffcc",

        # Admonitions
        "color-admonition-background": "transparent",
    },
}
