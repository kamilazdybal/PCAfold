# PCAfold

**PCAfold** is a Python software for generating, improving and analyzing empirical
low-dimensional manifolds obtained via Principal Component Analysis (PCA).
It incorporates advanced data pre-processing tools (including data clustering
and sampling), uses PCA as a dimensionality reduction technique and introduces
metrics to judge the topology of the low-dimensional manifolds.

#### [PCAfold Documentation](https://pca-python.readthedocs.io/en/latest/)

## Dependencies

**PCAfold** requires `python3.7` and the following packages:

- `copy`
- `matplotlib`
- `numpy`
- `random`
- `scipy`
- `sklearn`

## Installation

Clone the `PCAfold` repository and move into the `PCAfold` directory created:

```
git clone http://gitlab.multiscale.utah.edu/common/PCAfold.git
cd PCAfold
```

Run the `setup.py` script as below to complete the installation.

```
python3.7 setup.py install
```

## Testing

To run regression tests of all modules execute:

```python
from PCAfold import test

test.test()
```

You can also test each module separately:

```python
from PCAfold import test

test.test_preprocess()
test.test_reduction()
test.test_analysis()
```
