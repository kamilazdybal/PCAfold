# PCA library

## PCA

`PCA.py`

Contains functions for performing Principal Component Analysis.

This module includes a `test` function for regression testing. Run:

```
import PCA.PCA as P
P.test()
```

## Cluster-biased PCA

`cluster_biased_pca.py`

Contains functions for performing cluster-biased PCA.

More details can be found in the [documentation](https://gitlab.multiscale.utah.edu/common/PCA-python/-/wikis/Cluster-biased-PCA).

This module includes a `test` function for regression testing. Run:

```
import PCA.cluster_biased_pca as cbpca
cbpca.test()
```

## Clustering

`clustering.py`

Contains functions for clustering data sets and performing basic operations on clusters.

More details can be found in the [documentation](https://gitlab.multiscale.utah.edu/common/PCA-python/-/wikis/Clustering).

This module includes a `test` function for regression testing. Run:

```
import PCA.clustering as cl
cl.test()
```

## Train and test data selection

`train_test_select.py`

Contains functions for selecting train and test data for machine learning algorithms.

More details can be found in the [documentation](https://gitlab.multiscale.utah.edu/common/PCA-python/-/wikis/Train-and-test-data-selection).

This module includes a `test` function for regression testing. Run:

```
import PCA.train_test_select as tts
tts.test()
```
