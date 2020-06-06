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

## Train and test data selection

`train_test_select.py`

Contains functions for selecting train and test data for machine learning algorithms.

More details can be found in the [documentation](https://gitlab.multiscale.utah.edu/common/PCA-python/-/wikis/Train-and-test-data-selection).

This module includes a `test` function for regression testing. Run:

```
import PCA.regression.train_test_select as tts
tts.test()
```

## Cluster-biased PCA

`cluster_biased_pca.py`

Contains functions for performing cluster-biased PCA.

More details can be found in the [documentation](https://gitlab.multiscale.utah.edu/common/PCA-python/-/wikis/Cluster-biased-PCA).

This module includes a `test` function for regression testing. Run:

```
import PCA.regression.cluster_biased_pca as cbpca
cbpca.test()
```
