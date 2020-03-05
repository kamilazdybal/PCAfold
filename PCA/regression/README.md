# Regression

This directory contains auxiliary functions that can be used with non-linear regression techniques.

## Standardizing input/output

`standardize.py`

Contains functions for standardizing data for use in machine learning algorithms. Some of the common scalings were implemented such as normalizing data between 0 and 1 or between -1 and 1.

This module includes a `test` function for regression testing. Run:

```
import PCA.regression.standardize as st
st.test()
```

## Training data generation

`training_data_generation.py`

Contains functions for selecting train and test data for machine learning algorithms.

More details can be found in the [documentation](https://gitlab.multiscale.utah.edu/common/PCA-python/-/wikis/Training-data-generation).

This module includes a `test` function for regression testing. Run:

```
import PCA.regression.training_data_generation as tdg
tdg.test()
```
