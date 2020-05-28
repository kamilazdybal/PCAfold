`PCA-python` is the python version of the PCA Matlab suite found at: https://gitlab.multiscale.utah.edu/common/PCA

# Dependencies

`PCA-python` requires `python3` (developed with `python3.5`) and the following packages:
- `numpy`
- `scipy`
- `matplotlib`

# Installation
Clone the `PCA-python` repository and move into the `PCA-python` directory created:
```
git clone http://gitlab.multiscale.utah.edu/common/PCA-python.git
cd PCA-python
```

The `manifold_dimensionality` package in `PCA-python` requires the installation of the cythonized kernel regression function found in the `PCA-python/cython-kernel-regression` directory.
Navigate to the `cython-kernel-regression` directory and run the `setup.py` script located there.
```
cd cython-kernel-regression
python3.5 setup.py install
```

Navigate back to the main directory and run the `setup.py` script as below to complete the installation.
```
cd ..
python3.5 setup.py install
```

# Documentation

Check out the [Wiki Pages](https://gitlab.multiscale.utah.edu/common/PCA-python/-/wikis/pages) for documentation.

# Examples
The example file [Example.py](Example.py) demonstrates the use of preprocessing data before using PCA then transforming the variables into the principal components using a given scaling. Next, the absolute values of the eigenvectors are plotted. Finally, the regression tests are run and a statement of whether or not the tests passed is printed.

This example can be run from the terminal using:
```
python3.5 Example.py
```
