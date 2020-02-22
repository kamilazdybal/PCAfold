# Machine learning models

## Standardising input/output

`standardize.py`

Contains functions for standardising data for use in machine learning algorithms. Some of the common scalings were implemented such as normalising data between 0 and 1 or between -1 and 1.

This module includes a `test` function for sanity tests. Run:

```
import PCA.regression.standardize as st
st.test()
```

## Training data generation

`trainingDataGeneration.py` [documentation](https://gitlab.multiscale.utah.edu/common/PCA-python/-/wikis/Training-data-generation)

Contains functions for selecting train and test data for machine learning algorithms.
