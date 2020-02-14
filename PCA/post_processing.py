import numpy as np

# This will become a class of error metrics

def r2(x, f):
    """
    This function computes the coefficient of determination R2 as:

    R2 = 1 - ((x-f)^2)/(y^2)

    Input:

    `x` original data
    `f` model fit
    """

    from sklearn.metrics import r2_score

    r2 = r2_score(x, f)

    return r2

def rmse(x, f):
    """
    This function computes the Root Mean Squared Error as:

    RMSE = sqrt(mean((f-x)^2))

    Input:

    `x` original data
    `f` model fit
    """

    from sklearn.metrics import mean_squared_error
    from math import sqrt

    rmse = sqrt(mean_squared_error(x, f))

    return rmse

def nrmse(x, f, norm=0):
    """
    This function computes the Normalized Root Mean Squared Error as:

    If norm == 0:
    NRMSE = sqrt(mean((f-x)^2)/mean(x^2))

    If norm == 1:
    NRMSE = sqrt(mean((f-x)^2))/std(x)

    If norm == 2:
    NRMSE = sqrt(mean((f-x)^2)))/(max(x) - min(x))

    If norm == 3:
    NRMSE = sqrt(mean((f-x)^2))/(max(x^2) - min(x^2))

    If norm == 4:
    NRMSE = sqrt(mean((f-x)^2))/(std(x^2))

    Input:

    `x` original data
    `f` model fit
    """

    from sklearn.metrics import mean_squared_error
    from math import sqrt

    rmse = sqrt(mean_squared_error(x, f))

    if norm == 0:
        nrmse = rmse/sqrt(np.mean(x**2))
    elif norm == 1:
        nrmse = rmse/(np.std(x))
    elif norm == 2:
        nrmse = rmse/(np.max(x) - np.min(x))
    elif norm == 3:
        nrmse = rmse/sqrt(np.max(x**2) - np.min(x**2))
    elif norm == 4:
        nrmse = rmse/sqrt(np.std(x**2))
    elif norm == 5:
        nrmse = rmse/abs(np.mean(x))
    return nrmse
