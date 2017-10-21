`PCA-python` is the python version of the PCA Matlab suite found at: https://software.crsim.utah.edu:8443/James_Research_Group/PCA

# Dependencies
`PCA-python` requires `python3` (developed with `python3.5`) and the following packages:
- `numpy`
- `scipy`
- `matplotlib`

# Installation
Clone the `PCA-python` repository and move into the `PCA-python` directory created:
```
git clone https://software.crsim.utah.edu:8443/elizabeth/PCA-python.git
cd PCA-python
```

Run the `setup.py` script as below to complete the installation.
```
python3.5 setup.py install
```

# Examples
The example file [Example.py](Example.py) demonstrates the use of preprocessing data before using PCA then transforming the variables into the principal components using a given scaling. Next, the absolute values of the eigenvectors are plotted. Finally, the regression tests are run and a statement of whether or not the tests passed is printed.

This example can be run from the terminal using:
```
python3.5 Example.py
```
