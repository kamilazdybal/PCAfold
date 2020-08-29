[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Documentation Status](https://readthedocs.org/projects/pcafold/badge/?version=latest)](https://pcafold.readthedocs.io/en/latest/?badge=latest)

# PCAfold

**PCAfold** is a Python software for generating, improving and analyzing empirical
low-dimensional manifolds obtained via *Principal Component Analysis* (PCA).
It incorporates a variety of data pre-processing tools (including data clustering
and sampling), uses PCA as a dimensionality reduction technique and introduces
metrics to judge the topology of the low-dimensional manifolds.

### [PCAfold Documentation](https://pcafold.readthedocs.io/en/latest/)

## Software architecture

A general overview for using **PCAfold** modules is presented in the diagram
below:

![Screenshot](docs/images/PCAfold-diagram.png)

Each module's functionalities can also be used as a standalone tool for
performing a specific task and can easily combine with techniques outside of
this software, such as K-Means algorithm or Artificial Neural Networks.

## Installation

### Dependencies

**PCAfold** requires `python3.7` and the following packages:

- `copy`
- `Cython`
- `matplotlib`
- `numpy`
- `random`
- `scipy`

### Build from source

Clone the `PCAfold` repository and move into the `PCAfold` directory created:

```
git clone http://gitlab.multiscale.utah.edu/common/PCAfold.git
cd PCAfold
```

Run the `setup.py` script as below to complete the installation:

```
python3.7 setup.py install
```

You are ready to `import PCAfold`!

### Local documentation build

To build the documentation locally, you need `sphinx` installed on your machine,
along with the following extensions:

```
sphinx.ext.todo
sphinx.ext.githubpages
sphinx.ext.autodoc
sphinx.ext.napoleon
sphinx.ext.mathjax
sphinx.ext.autosummary
sphinxcontrib.bibtex
```

Then, navigate to `docs/` directory and build the documentation:

```
sphinx-build -b html . builddir

make html
```

Documentation main page `_build/html/index.html` can be opened in a web browser.

In MacOS you can open it directly from the terminal:

```
open _build/html/index.html
```

### Testing

To run regression tests from the base repo directory run:

```
python3.7 -m unittest discover
```

To switch verbose on, use the `-v` flag.

## Plotting

Some functions within **PCAfold** result in plot outputs. Global styles for the
plots are set using the `styles.py` file. This file can be updated with new
settings that will be seen globally by **PCAfold** modules. Re-build the project
after changing `styles.py` file:

```
python3.7 setup.py install
```

All plotting functions return handles to generated plots.
