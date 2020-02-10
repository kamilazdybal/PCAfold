# Non-linear regression

## Training data generation for machine learning

`trainingDataGeneration.py` [documentation](https://gitlab.multiscale.utah.edu/common/PCA-python/-/wikis/Training-data-generation-documentation)

## Centering and scaling of input/output for machine learning models



## Activation functions

An example of defining new activation functions from the existing ones:

```python
from keras import backend as K

def relux(x):
    return K.relu(x) * x

def swish(x):
    return K.sigmoid(x) * x
```

## Loss functions

An example of defining new loss functions:

```python
from keras import backend as K

def root_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true)))
```
