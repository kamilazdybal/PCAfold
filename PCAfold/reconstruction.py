"""reconstruction.py: module for reconstruction of QoIs from manifolds."""

__author__ = "Kamila Zdybal, Elizabeth Armstrong, Alessandro Parente and James C. Sutherland"
__copyright__ = "Copyright (c) 2020-2023, Kamila Zdybal, Elizabeth Armstrong, Alessandro Parente and James C. Sutherland"
__credits__ = ["Department of Chemical Engineering, University of Utah, Salt Lake City, Utah, USA", "Universite Libre de Bruxelles, Aero-Thermo-Mechanics Laboratory, Brussels, Belgium"]
__license__ = "MIT"
__version__ = "2.0.0"
__maintainer__ = ["Kamila Zdybal", "Elizabeth Armstrong"]
__email__ = ["kamilazdybal@gmail.com", "Elizabeth.Armstrong@chemeng.utah.edu", "James.Sutherland@chemeng.utah.edu"]
__status__ = "Production"

from tqdm import tqdm
import matplotlib.pyplot as plt
from PCAfold.styles import *
from PCAfold import preprocess
from PCAfold import reduction
import pickle
from matplotlib.colors import ListedColormap
import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
from PCAfold.preprocess import center_scale
import pickle
import tensorflow.compat.v1 as tf
tf.compat.v1.disable_eager_execution()

################################################################################
#
# Regression assessment tools
#
################################################################################

class RegressionAssessment:
    """
    Wrapper class for storing all regression assessment metrics for a given
    regression solution given by the observed dependent variables, :math:`\\pmb{\\phi}_o`,
    and the predicted dependent variables, :math:`\\pmb{\\phi}_p`.

    **Example:**

    .. code:: python

        from PCAfold import PCA, RegressionAssessment
        import numpy as np

        # Generate dummy data set:
        X = np.random.rand(100,3)

        # Instantiate PCA class object:
        pca_X = PCA(X, scaling='auto', n_components=2)

        # Approximate the data set:
        X_rec = pca_X.reconstruct(pca_X.transform(X))

        # Instantiate RegressionAssessment class object:
        regression_metrics = RegressionAssessment(X, X_rec)

        # Access mean absolute error values:
        MAE = regression_metrics.mean_absolute_error

    In addition, all stratified regression metrics can be computed on a single variable:

    .. code:: python

        from PCAfold import variable_bins

        # Generate bins:
        (idx, bins_borders) = variable_bins(X[:,0], k=5, verbose=False)

        # Instantiate RegressionAssessment class object:
        stratified_regression_metrics = RegressionAssessment(X[:,0], X_rec[:,0], idx=idx)

        # Access stratified mean absolute error values:
        stratified_MAE = stratified_regression_metrics.stratified_mean_absolute_error

    :param observed:
        ``numpy.ndarray`` specifying the observed values of dependent variables, :math:`\\pmb{\\phi}_o`. It should be of size ``(n_observations,)`` or ``(n_observations,n_variables)``.
    :param predicted:
        ``numpy.ndarray`` specifying the predicted values of dependent variables, :math:`\\pmb{\\phi}_p`. It should be of size ``(n_observations,)`` or ``(n_observations,n_variables)``.
    :param idx:
        ``numpy.ndarray`` of cluster classifications. It should be of size ``(n_observations,)`` or ``(n_observations,1)``.
    :param variable_names: (optional)
        ``list`` of ``str`` specifying variable names.
    :param use_global_mean: (optional)
        ``bool`` specifying if global mean of the observed variable should be used as a reference in :math:`R^2` calculation.
    :param norm:
        ``str`` specifying the normalization, :math:`d_{norm}`, for NRMSE computation. It can be one of the following: ``std``, ``range``, ``root_square_mean``, ``root_square_range``, ``root_square_std``, ``abs_mean``.
    :param use_global_norm: (optional)
        ``bool`` specifying if global norm of the observed variable should be used in NRMSE calculation.
    :param tolerance:
        ``float`` specifying the tolerance for GDE computation.

    **Attributes:**

    - **coefficient_of_determination** - (read only) ``numpy.ndarray`` specifying the coefficient of determination, :math:`R^2`, values. It has size ``(1,n_variables)``.
    - **mean_absolute_error** - (read only) ``numpy.ndarray`` specifying the mean absolute error (MAE) values. It has size ``(1,n_variables)``.
    - **mean_squared_error** - (read only) ``numpy.ndarray`` specifying the mean squared error (MSE) values. It has size ``(1,n_variables)``.
    - **root_mean_squared_error** - (read only) ``numpy.ndarray`` specifying the root mean squared error (RMSE) values. It has size ``(1,n_variables)``.
    - **normalized_root_mean_squared_error** - (read only) ``numpy.ndarray`` specifying the normalized root mean squared error (NRMSE) values. It has size ``(1,n_variables)``.
    - **good_direction_estimate** - (read only) ``float`` specifying the good direction estimate (GDE) value, treating the entire :math:`\\pmb{\\phi}_o` and :math:`\\pmb{\\phi}_p` as vectors. Note that if a single dependent variable is passed, GDE cannot be computed and is set to ``NaN``.

    If ``idx`` has been specified:

    - **stratified_coefficient_of_determination** - (read only) ``numpy.ndarray`` specifying the coefficient of determination, :math:`R^2`, values. It has size ``(1,n_variables)``.
    - **stratified_mean_absolute_error** - (read only) ``numpy.ndarray`` specifying the mean absolute error (MAE) values. It has size ``(1,n_variables)``.
    - **stratified_mean_squared_error** - (read only) ``numpy.ndarray`` specifying the mean squared error (MSE) values. It has size ``(1,n_variables)``.
    - **stratified_root_mean_squared_error** - (read only) ``numpy.ndarray`` specifying the root mean squared error (RMSE) values. It has size ``(1,n_variables)``.
    - **stratified_normalized_root_mean_squared_error** - (read only) ``numpy.ndarray`` specifying the normalized root mean squared error (NRMSE) values. It has size ``(1,n_variables)``.
    """

    def __init__(self, observed, predicted, idx=None, variable_names=None, use_global_mean=False, norm='std', use_global_norm=False, tolerance=0.05):

        if not isinstance(observed, np.ndarray):
            raise ValueError("Parameter `observed` has to be of type `numpy.ndarray`.")

        try:
            (n_observed,) = np.shape(observed)
            n_var_observed = 1
            observed = observed[:,None]
        except:
            (n_observed, n_var_observed) = np.shape(observed)

        if not isinstance(predicted, np.ndarray):
            raise ValueError("Parameter `predicted` has to be of type `numpy.ndarray`.")

        try:
            (n_predicted,) = np.shape(predicted)
            n_var_predicted = 1
            predicted = predicted[:,None]
        except:
            (n_predicted, n_var_predicted) = np.shape(predicted)

        if n_observed != n_predicted:
            raise ValueError("Parameter `observed` has different number of elements than `predicted`.")

        if n_var_observed != n_var_predicted:
            raise ValueError("Parameter `observed` has different number of elements than `predicted`.")

        self.__n_variables = n_var_observed

        if idx is not None:

            if isinstance(idx, np.ndarray):
                if not all(isinstance(i, np.integer) for i in idx.ravel()):
                    raise ValueError("Parameter `idx` can only contain integers.")
            else:
                raise ValueError("Parameter `idx` has to be of type `numpy.ndarray`.")

            try:
                (n_observations_idx, ) = np.shape(idx)
                n_idx = 1
            except:
                (n_observations_idx, n_idx) = np.shape(idx)

            if n_idx != 1:
                raise ValueError("Parameter `idx` has to have size `(n_observations,)` or `(n_observations,1)`.")

            if n_observations_idx != n_observed:
                raise ValueError('Vector of cluster classifications `idx` has different number of observations than the original data set `X`.')

            if n_var_observed != 1:
                raise ValueError('Stratified regression metrics can only be computed on a single vector.')

            self.__n_clusters = len(np.unique(idx))
            self.__cluster_populations = preprocess.get_populations(idx)

            self.__cluster_min = []
            self.__cluster_max = []
            for i in range(0,self.__n_clusters):
                (cluster_indices, ) = np.where(idx==i)
                self.__cluster_min.append(np.min(observed[cluster_indices,:]))
                self.__cluster_max.append(np.max(observed[cluster_indices,:]))

        if not isinstance(use_global_mean, bool):
            raise ValueError("Parameter `use_global_mean` has to be a boolean.")

        if variable_names is not None:
            if not isinstance(variable_names, list):
                raise ValueError("Parameter `variable_names` has to be of type `list`.")
            else:
                if self.__n_variables != len(variable_names):
                    raise ValueError("Parameter `variable_names` has different number of variables than `observed` and `predicted`.")
        else:
            variable_names = []
            for i in range(0,self.__n_variables):
                variable_names.append('X' + str(i+1))

        self.__variable_names = variable_names

        self.__coefficient_of_determination_matrix = np.ones((1,self.__n_variables))
        self.__mean_absolute_error_matrix = np.ones((1,self.__n_variables))
        self.__mean_squared_error_matrix = np.ones((1,self.__n_variables))
        self.__mean_squared_logarithmic_error_matrix = np.ones((1,self.__n_variables))
        self.__root_mean_squared_error_matrix = np.ones((1,self.__n_variables))
        self.__normalized_root_mean_squared_error_matrix = np.ones((1,self.__n_variables))

        if n_var_observed > 1:
            _, self.__good_direction_estimate_value = good_direction_estimate(observed, predicted, tolerance=tolerance)
            self.__good_direction_estimate_matrix = self.__good_direction_estimate_value * np.ones((1,self.__n_variables))
        else:
            self.__good_direction_estimate_value = np.NAN
            self.__good_direction_estimate_matrix = self.__good_direction_estimate_value * np.ones((1,self.__n_variables))

        for i in range(0,self.__n_variables):

            self.__coefficient_of_determination_matrix[0,i] = coefficient_of_determination(observed[:,i], predicted[:,i])
            self.__mean_absolute_error_matrix[0,i] = mean_absolute_error(observed[:,i], predicted[:,i])
            self.__mean_squared_error_matrix[0,i] = mean_squared_error(observed[:,i], predicted[:,i])
            self.__mean_squared_logarithmic_error_matrix[0,i] = mean_squared_logarithmic_error(np.abs(observed[:,i]), np.abs(predicted[:,i]))
            self.__root_mean_squared_error_matrix[0,i] = root_mean_squared_error(observed[:,i], predicted[:,i])
            self.__normalized_root_mean_squared_error_matrix[0,i] = normalized_root_mean_squared_error(observed[:,i], predicted[:,i], norm=norm)

        if idx is not None:

            self.__stratified_coefficient_of_determination = stratified_coefficient_of_determination(observed, predicted, idx=idx, use_global_mean=use_global_mean)
            self.__stratified_mean_absolute_error = stratified_mean_absolute_error(observed, predicted, idx=idx)
            self.__stratified_mean_squared_error = stratified_mean_squared_error(observed, predicted, idx=idx)
            self.__stratified_mean_squared_logarithmic_error = stratified_mean_squared_logarithmic_error(np.abs(observed), np.abs(predicted), idx=idx)
            self.__stratified_root_mean_squared_error = stratified_root_mean_squared_error(observed, predicted, idx=idx)
            self.__stratified_normalized_root_mean_squared_error = stratified_normalized_root_mean_squared_error(observed, predicted, idx=idx, norm=norm, use_global_norm=use_global_norm)

        else:

            self.__stratified_coefficient_of_determination = None
            self.__stratified_mean_absolute_error = None
            self.__stratified_mean_squared_error = None
            self.__stratified_mean_squared_logarithmic_error = None
            self.__stratified_root_mean_squared_error = None
            self.__stratified_normalized_root_mean_squared_error = None

    @property
    def coefficient_of_determination(self):
        return self.__coefficient_of_determination_matrix

    @property
    def mean_absolute_error(self):
        return self.__mean_absolute_error_matrix

    @property
    def mean_squared_error(self):
        return self.__mean_squared_error_matrix

    @property
    def mean_squared_logarithmic_error(self):
        return self.__mean_squared_logarithmic_error_matrix

    @property
    def root_mean_squared_error(self):
        return self.__root_mean_squared_error_matrix

    @property
    def normalized_root_mean_squared_error(self):
        return self.__normalized_root_mean_squared_error_matrix

    @property
    def good_direction_estimate(self):
        return self.__good_direction_estimate_value

    @property
    def stratified_coefficient_of_determination(self):
        return self.__stratified_coefficient_of_determination

    @property
    def stratified_mean_absolute_error(self):
        return self.__stratified_mean_absolute_error

    @property
    def stratified_mean_squared_error(self):
        return self.__stratified_mean_squared_error

    @property
    def stratified_mean_squared_logarithmic_error(self):
        return self.__stratified_mean_squared_logarithmic_error

    @property
    def stratified_root_mean_squared_error(self):
        return self.__stratified_root_mean_squared_error

    @property
    def stratified_normalized_root_mean_squared_error(self):
        return self.__stratified_normalized_root_mean_squared_error

# ------------------------------------------------------------------------------

    def print_metrics(self, table_format=['raw'], float_format='.4f', metrics=None, comparison=None):
        """
        Prints regression assessment metrics as raw text, in ``tex`` format and/or as ``pandas.DataFrame``.

        **Example:**

        .. code:: python

            from PCAfold import PCA, RegressionAssessment
            import numpy as np

            # Generate dummy data set:
            X = np.random.rand(100,3)

            # Instantiate PCA class object:
            pca_X = PCA(X, scaling='auto', n_components=2)

            # Approximate the data set:
            X_rec = pca_X.reconstruct(pca_X.transform(X))

            # Instantiate RegressionAssessment class object:
            regression_metrics = RegressionAssessment(X, X_rec)

            # Print regression metrics:
            regression_metrics.print_metrics(table_format=['raw', 'tex', 'pandas'],
                                             float_format='.4f',
                                             metrics=['R2', 'NRMSE', 'GDE'])

        .. note::

            Adding ``'raw'`` to the ``table_format`` list will result in printing:

            .. code-block:: text

                -------------------------
                X1
                R2:	0.9900
                NRMSE:	0.0999
                GDE:	70.0000
                -------------------------
                X2
                R2:	0.6126
                NRMSE:	0.6224
                GDE:	70.0000
                -------------------------
                X3
                R2:	0.6368
                NRMSE:	0.6026
                GDE:	70.0000

            Adding ``'tex'`` to the ``table_format`` list will result in printing:

            .. code-block:: text

                \\begin{table}[h!]
                \\begin{center}
                \\begin{tabular}{llll} \\toprule
                 & \\textit{X1} & \\textit{X2} & \\textit{X3} \\\\ \\midrule
                R2 & 0.9900 & 0.6126 & 0.6368 \\\\
                NRMSE & 0.0999 & 0.6224 & 0.6026 \\\\
                GDE & 70.0000 & 70.0000 & 70.0000 \\\\
                \\end{tabular}
                \\caption{}\\label{}
                \\end{center}
                \\end{table}

            Adding ``'pandas'`` to the ``table_format`` list (works well in Jupyter notebooks) will result in printing:

            .. image:: ../images/generate-pandas-table.png
                :width: 300
                :align: center

        Additionally, the current object of ``RegressionAssessment`` class can be compared with another object:

        .. code:: python

            from PCAfold import PCA, RegressionAssessment
            import numpy as np

            # Generate dummy data set:
            X = np.random.rand(100,3)
            Y = np.random.rand(100,3)

            # Instantiate PCA class object:
            pca_X = PCA(X, scaling='auto', n_components=2)
            pca_Y = PCA(Y, scaling='auto', n_components=2)

            # Approximate the data set:
            X_rec = pca_X.reconstruct(pca_X.transform(X))
            Y_rec = pca_Y.reconstruct(pca_Y.transform(Y))

            # Instantiate RegressionAssessment class object:
            regression_metrics_X = RegressionAssessment(X, X_rec)
            regression_metrics_Y = RegressionAssessment(Y, Y_rec)

            # Print regression metrics:
            regression_metrics_X.print_metrics(table_format=['raw', 'pandas'],
                                               float_format='.4f',
                                               metrics=['R2', 'NRMSE', 'GDE'],
                                               comparison=regression_metrics_Y)

        .. note::

            Adding ``'raw'`` to the ``table_format`` list will result in printing:

            .. code-block:: text

                -------------------------
                X1
                R2:	0.9133	BETTER
                NRMSE:	0.2944	BETTER
                GDE:	67.0000	WORSE
                -------------------------
                X2
                R2:	0.5969	WORSE
                NRMSE:	0.6349	WORSE
                GDE:	67.0000	WORSE
                -------------------------
                X3
                R2:	0.6175	WORSE
                NRMSE:	0.6185	WORSE
                GDE:	67.0000	WORSE

            Adding ``'pandas'`` to the ``table_format`` list (works well in Jupyter notebooks) will result in printing:

            .. image:: ../images/generate-pandas-table-comparison.png
                :width: 300
                :align: center

        :param table_format: (optional)
            ``list`` of ``str`` specifying the format(s) in which the table should be printed.
            Strings can only be ``'raw'``, ``'tex'`` and/or ``'pandas'``.
        :param float_format: (optional)
            ``str`` specifying the display format for the numerical entries inside the
            table. By default it is set to ``'.4f'``.
        :param metrics: (optional)
            ``list`` of ``str`` specifying which metrics should be printed. Strings can only be ``'R2'``, ``'MAE'``, ``'MSE'``, ``'MSLE'``, ``'RMSE'``, ``'NRMSE'``, ``'GDE'``.
            If metrics is set to ``None``, all available metrics will be printed.
        :param comparison: (optional)
            object of ``RegressionAssessment`` class specifying the metrics that should be compared with the current regression metrics.
        """

        __table_formats = ['raw', 'tex', 'pandas']
        __metrics_names = ['R2', 'MAE', 'MSE', 'MSLE', 'RMSE', 'NRMSE', 'GDE']
        __metrics_dict = {'R2': self.__coefficient_of_determination_matrix,
                          'MAE': self.__mean_absolute_error_matrix,
                          'MSE': self.__mean_squared_error_matrix,
                          'MSLE': self.__mean_squared_logarithmic_error_matrix,
                          'RMSE': self.__root_mean_squared_error_matrix,
                          'NRMSE': self.__normalized_root_mean_squared_error_matrix,
                          'GDE': self.__good_direction_estimate_matrix}
        if comparison is not None:
            __comparison_metrics_dict = {'R2': comparison.coefficient_of_determination,
                                         'MAE': comparison.mean_absolute_error,
                                         'MSE': comparison.mean_squared_error,
                                         'MSLE': comparison.mean_squared_logarithmic_error,
                                         'RMSE': comparison.root_mean_squared_error,
                                         'NRMSE': comparison.normalized_root_mean_squared_error,
                                         'GDE': comparison.good_direction_estimate * np.ones_like(comparison.coefficient_of_determination)}

        if not isinstance(table_format, list):
            raise ValueError("Parameter `table_format` has to be of type `list`.")

        for item in table_format:
            if item not in __table_formats:
                raise ValueError("Parameter `table_format` can only contain 'raw', 'tex' and/or 'pandas'.")

        if not isinstance(float_format, str):
            raise ValueError("Parameter `float_format` has to be of type `str`.")

        if metrics is not None:
            if not isinstance(metrics, list):
                raise ValueError("Parameter `metrics` has to be of type `list`.")

            for item in metrics:
                if item not in __metrics_names:
                    raise ValueError("Parameter `metrics` can only be: 'R2', 'MAE', 'MSE', 'MSLE', 'RMSE', 'NRMSE', 'GDE'.")
        else:
            metrics = __metrics_names

        if comparison is None:

            for item in set(table_format):

                if item=='raw':

                    for i in range(0,self.__n_variables):

                        print('-'*25 + '\n' + self.__variable_names[i])

                        metrics_to_print = []
                        for metric in metrics:
                            metrics_to_print.append(__metrics_dict[metric][0,i])

                        for j in range(0,len(metrics)):
                            print(metrics[j] + ':\t' + ('%' + float_format) % metrics_to_print[j])

                if item=='tex':

                    import pandas as pd

                    metrics_to_print = np.zeros_like(self.__coefficient_of_determination_matrix)
                    for metric in metrics:
                        metrics_to_print = np.vstack((metrics_to_print, __metrics_dict[metric]))
                    metrics_to_print = metrics_to_print[1::,:]

                    metrics_table = pd.DataFrame(metrics_to_print, columns=self.__variable_names, index=metrics)
                    generate_tex_table(metrics_table, float_format=float_format)

                if item=='pandas':

                    import pandas as pd
                    from IPython.display import display
                    pandas_format = '{:,' + float_format + '}'

                    metrics_to_print = np.zeros_like(self.__coefficient_of_determination_matrix.T)
                    for metric in metrics:
                        metrics_to_print = np.hstack((metrics_to_print, __metrics_dict[metric].T))
                    metrics_to_print = metrics_to_print[:,1::]

                    metrics_table = pd.DataFrame(metrics_to_print, columns=metrics, index=self.__variable_names)
                    formatted_table = metrics_table.style.format(pandas_format)
                    display(formatted_table)

        else:

            for item in set(table_format):

                if item=='raw':

                    for i in range(0,self.__n_variables):

                        print('-'*25 + '\n' + self.__variable_names[i])

                        metrics_to_print = []
                        comparison_metrics_to_print = []
                        for metric in metrics:
                            metrics_to_print.append(__metrics_dict[metric][0,i])
                            comparison_metrics_to_print.append(__comparison_metrics_dict[metric][0,i])

                        for j, metric in enumerate(metrics):

                            if metric == 'R2' or metric == 'GDE':
                                if metrics_to_print[j] > comparison_metrics_to_print[j]:
                                    print(metrics[j] + ':\t' + ('%' + float_format) % metrics_to_print[j] + colored('\tBETTER', 'green'))
                                elif metrics_to_print[j] < comparison_metrics_to_print[j]:
                                    print(metrics[j] + ':\t' + ('%' + float_format) % metrics_to_print[j] + colored('\tWORSE', 'red'))
                                elif metrics_to_print[j] == comparison_metrics_to_print[j]:
                                    print(metrics[j] + ':\t' + ('%' + float_format) % metrics_to_print[j] + '\tSAME')
                            else:
                                if metrics_to_print[j] > comparison_metrics_to_print[j]:
                                    print(metrics[j] + ':\t' + ('%' + float_format) % metrics_to_print[j] + colored('\tWORSE', 'red'))
                                elif metrics_to_print[j] < comparison_metrics_to_print[j]:
                                    print(metrics[j] + ':\t' + ('%' + float_format) % metrics_to_print[j] + colored('\tBETTER', 'green'))
                                elif metrics_to_print[j] == comparison_metrics_to_print[j]:
                                    print(metrics[j] + ':\t' + ('%' + float_format) % metrics_to_print[j] + '\tSAME')

                if item=='pandas':

                    import pandas as pd
                    from IPython.display import display
                    pandas_format = '{:,' + float_format + '}'

                    metrics_to_print = np.zeros_like(self.__coefficient_of_determination_matrix.T)
                    comparison_metrics_to_print = np.zeros_like(comparison.coefficient_of_determination.T)
                    for metric in metrics:
                        metrics_to_print = np.hstack((metrics_to_print, __metrics_dict[metric].T))
                        comparison_metrics_to_print = np.hstack((comparison_metrics_to_print, __comparison_metrics_dict[metric].T))
                    metrics_to_print = metrics_to_print[:,1::]
                    comparison_metrics_to_print = comparison_metrics_to_print[:,1::]

                    def highlight_better(data, data_comparison, color='lightgreen'):

                        attr = 'background-color: {}'.format(color)

                        is_better = False * data

                        # Lower value is better (MAE, MSE, RMSE, NRMSE):
                        try:
                            is_better['MAE'] = data['MAE'].astype(float) < data_comparison['MAE']
                        except:
                            pass
                        try:
                            is_better['MSE'] = data['MSE'].astype(float) < data_comparison['MSE']
                        except:
                            pass
                        try:
                            is_better['MSLE'] = data['MSLE'].astype(float) < data_comparison['MSLE']
                        except:
                            pass
                        try:
                            is_better['RMSE'] = data['RMSE'].astype(float) < data_comparison['RMSE']
                        except:
                            pass
                        try:
                            is_better['NRMSE'] = data['NRMSE'].astype(float) < data_comparison['NRMSE']
                        except:
                            pass

                        # Higher value is better (R2 and GDE):
                        try:
                            is_better['R2'] = data['R2'].astype(float) > data_comparison['R2']
                        except:
                            pass
                        try:
                            is_better['GDE'] = data['GDE'].astype(float) > data_comparison['GDE']
                        except:
                            pass

                        formatting = [attr if v else '' for v in is_better]

                        formatting = pd.DataFrame(np.where(is_better, attr, ''), index=data.index, columns=data.columns)

                        return formatting

                    def highlight_worse(data, data_comparison, color='salmon'):

                        attr = 'background-color: {}'.format(color)

                        is_worse = False * data

                        # Higher value is worse (MAE, MSE, MSLE, RMSE, NRMSE):
                        try:
                            is_worse['MAE'] = data['MAE'].astype(float) > data_comparison['MAE']
                        except:
                            pass
                        try:
                            is_worse['MSE'] = data['MSE'].astype(float) > data_comparison['MSE']
                        except:
                            pass
                        try:
                            is_worse['MSLE'] = data['MSLE'].astype(float) > data_comparison['MSLE']
                        except:
                            pass
                        try:
                            is_worse['RMSE'] = data['RMSE'].astype(float) > data_comparison['RMSE']
                        except:
                            pass
                        try:
                            is_worse['NRMSE'] = data['NRMSE'].astype(float) > data_comparison['NRMSE']
                        except:
                            pass

                        # Lower value is worse (R2 and GDE):
                        try:
                            is_worse['R2'] = data['R2'].astype(float) < data_comparison['R2']
                        except:
                            pass
                        try:
                            is_worse['GDE'] = data['GDE'].astype(float) < data_comparison['GDE']
                        except:
                            pass

                        formatting = [attr if v else '' for v in is_worse]

                        formatting = pd.DataFrame(np.where(is_worse, attr, ''), index=data.index, columns=data.columns)

                        return formatting

                    metrics_table = pd.DataFrame(metrics_to_print, columns=metrics, index=self.__variable_names)
                    comparison_metrics_table = pd.DataFrame(comparison_metrics_to_print, columns=metrics, index=self.__variable_names)

                    formatted_table = metrics_table.style.apply(highlight_better, data_comparison=comparison_metrics_table, axis=None)\
                                                         .apply(highlight_worse, data_comparison=comparison_metrics_table, axis=None)\
                                                         .format(pandas_format)

                    display(formatted_table)

# ------------------------------------------------------------------------------

    def print_stratified_metrics(self, table_format=['raw'], float_format='.4f', metrics=None, comparison=None):
        """
        Prints stratified regression assessment metrics as raw text, in ``tex`` format and/or as ``pandas.DataFrame``.
        In each cluster, in addition to the regression metrics, number of observations is printed,
        along with the minimum and maximum values of the observed variable in that cluster.

        **Example:**

        .. code:: python

            from PCAfold import PCA, variable_bins, RegressionAssessment
            import numpy as np

            # Generate dummy data set:
            X = np.random.rand(100,3)

            # Instantiate PCA class object:
            pca_X = PCA(X, scaling='auto', n_components=2)

            # Approximate the data set:
            X_rec = pca_X.reconstruct(pca_X.transform(X))

            # Generate bins:
            (idx, bins_borders) = variable_bins(X[:,0], k=3, verbose=False)

            # Instantiate RegressionAssessment class object:
            stratified_regression_metrics = RegressionAssessment(X[:,0], X_rec[:,0], idx=idx)

            # Print regression metrics:
            stratified_regression_metrics.print_stratified_metrics(table_format=['raw', 'tex', 'pandas'],
                                                                   float_format='.4f',
                                                                   metrics=['R2', 'MAE', 'NRMSE'])

        .. note::

            Adding ``'raw'`` to the ``table_format`` list will result in printing:

            .. code-block:: text

                -------------------------
                k1
                Observations:	31
                Min:	0.0120
                Max:	0.3311
                R2:	-3.3271
                MAE:	0.1774
                NRMSE:	2.0802
                -------------------------
                k2
                Observations:	38
                Min:	0.3425
                Max:	0.6665
                R2:	-1.4608
                MAE:	0.1367
                NRMSE:	1.5687
                -------------------------
                k3
                Observations:	31
                Min:	0.6853
                Max:	0.9959
                R2:	-3.7319
                MAE:	0.1743
                NRMSE:	2.1753

            Adding ``'tex'`` to the ``table_format`` list will result in printing:

            .. code-block:: text

                \\begin{table}[h!]
                \\begin{center}
                \\begin{tabular}{llll} \\toprule
                 & \\textit{k1} & \\textit{k2} & \\textit{k3} \\\\ \\midrule
                Observations & 31.0000 & 38.0000 & 31.0000 \\\\
                Min & 0.0120 & 0.3425 & 0.6853 \\\\
                Max & 0.3311 & 0.6665 & 0.9959 \\\\
                R2 & -3.3271 & -1.4608 & -3.7319 \\\\
                MAE & 0.1774 & 0.1367 & 0.1743 \\\\
                NRMSE & 2.0802 & 1.5687 & 2.1753 \\\\
                \\end{tabular}
                \\caption{}\\label{}
                \\end{center}
                \\end{table}

            Adding ``'pandas'`` to the ``table_format`` list (works well in Jupyter notebooks) will result in printing:

            .. image:: ../images/generate-pandas-table-stratified.png
                :width: 500
                :align: center

        Additionally, the current object of ``RegressionAssessment`` class can be compared with another object:

        .. code:: python

            from PCAfold import PCA, variable_bins, RegressionAssessment
            import numpy as np

            # Generate dummy data set:
            X = np.random.rand(100,3)

            # Instantiate PCA class object:
            pca_X = PCA(X, scaling='auto', n_components=2)

            # Approximate the data set:
            X_rec = pca_X.reconstruct(pca_X.transform(X))

            # Generate bins:
            (idx, bins_borders) = variable_bins(X[:,0], k=3, verbose=False)

            # Instantiate RegressionAssessment class object:
            stratified_regression_metrics_0 = RegressionAssessment(X[:,0], X_rec[:,0], idx=idx)
            stratified_regression_metrics_1 = RegressionAssessment(X[:,1], X_rec[:,1], idx=idx)

            # Print regression metrics:
            stratified_regression_metrics_0.print_stratified_metrics(table_format=['raw', 'pandas'],
                                                                     float_format='.4f',
                                                                     metrics=['R2', 'MAE', 'NRMSE'],
                                                                     comparison=stratified_regression_metrics_1)

        .. note::

            Adding ``'raw'`` to the ``table_format`` list will result in printing:

            .. code-block:: text

                -------------------------
                k1
                Observations:	39
                Min:	0.0013
                Max:	0.3097
                R2:	0.9236	BETTER
                MAE:	0.0185	BETTER
                NRMSE:	0.2764	BETTER
                -------------------------
                k2
                Observations:	29
                Min:	0.3519
                Max:	0.6630
                R2:	0.9380	BETTER
                MAE:	0.0179	BETTER
                NRMSE:	0.2491	BETTER
                -------------------------
                k3
                Observations:	32
                Min:	0.6663
                Max:	0.9943
                R2:	0.9343	BETTER
                MAE:	0.0194	BETTER
                NRMSE:	0.2563	BETTER

            Adding ``'pandas'`` to the ``table_format`` list (works well in Jupyter notebooks) will result in printing:

            .. image:: ../images/generate-pandas-table-comparison-stratified.png
                :width: 500
                :align: center

        :param table_format: (optional)
            ``list`` of ``str`` specifying the format(s) in which the table should be printed.
            Strings can only be ``'raw'``, ``'tex'`` and/or ``'pandas'``.
        :param float_format: (optional)
            ``str`` specifying the display format for the numerical entries inside the
            table. By default it is set to ``'.4f'``.
        :param metrics: (optional)
            ``list`` of ``str`` specifying which metrics should be printed. Strings can only be ``'R2'``, ``'MAE'``, ``'MSE'``,  ``'MSLE'``, ``'RMSE'``, ``'NRMSE'``.
            If metrics is set to ``None``, all available metrics will be printed.
        :param comparison: (optional)
            object of ``RegressionAssessment`` class specifying the metrics that should be compared with the current regression metrics.
        """

        __table_formats = ['raw', 'tex', 'pandas']
        __metrics_names = ['R2', 'MAE', 'MSE', 'MSLE', 'RMSE', 'NRMSE']
        __clusters_names = ['k' + str(i) for i in range(1,self.__n_clusters+1)]
        __metrics_dict = {'R2': self.__stratified_coefficient_of_determination,
                          'MAE': self.__stratified_mean_absolute_error,
                          'MSE': self.__stratified_mean_squared_error,
                          'MSLE': self.__stratified_mean_squared_logarithmic_error,
                          'RMSE': self.__stratified_root_mean_squared_error,
                          'NRMSE': self.__stratified_normalized_root_mean_squared_error}
        if comparison is not None:
            __comparison_metrics_dict = {'R2': comparison.stratified_coefficient_of_determination,
                                         'MAE': comparison.stratified_mean_absolute_error,
                                         'MSE': comparison.stratified_mean_squared_error,
                                         'MSLE': comparison.stratified_mean_squared_logarithmic_error,
                                         'RMSE': comparison.stratified_root_mean_squared_error,
                                         'NRMSE': comparison.stratified_normalized_root_mean_squared_error}

        if not isinstance(table_format, list):
            raise ValueError("Parameter `table_format` has to be of type `str`.")

        for item in table_format:
            if item not in __table_formats:
                raise ValueError("Parameter `table_format` can only contain 'raw', 'tex' and/or 'pandas'.")

        if not isinstance(float_format, str):
            raise ValueError("Parameter `float_format` has to be of type `str`.")

        if metrics is not None:
            if not isinstance(metrics, list):
                raise ValueError("Parameter `metrics` has to be of type `list`.")

            for item in metrics:
                if item not in __metrics_names:
                    raise ValueError("Parameter `metrics` can only be: 'R2', 'MAE', 'MSE', 'MSLE', 'RMSE', 'NRMSE'.")
        else:
            metrics = __metrics_names

        if comparison is None:

            for item in set(table_format):

                if item=='raw':

                    for i in range(0,self.__n_clusters):

                        print('-'*25 + '\n' + __clusters_names[i])

                        metrics_to_print = [self.__cluster_populations[i], self.__cluster_min[i], self.__cluster_max[i]]
                        for metric in metrics:
                            metrics_to_print.append(__metrics_dict[metric][i])

                        print('Observations' + ':\t' + str(metrics_to_print[0]))
                        print('Min' + ':\t' + ('%' + float_format) % metrics_to_print[1])
                        print('Max' + ':\t' + ('%' + float_format) % metrics_to_print[2])
                        for j in range(0,len(metrics)):
                            print(metrics[j] + ':\t' + ('%' + float_format) % metrics_to_print[j+3])

                if item=='tex':

                    import pandas as pd

                    metrics_to_print = np.vstack((self.__cluster_populations, self.__cluster_min, self.__cluster_max))

                    for metric in metrics:
                        metrics_to_print = np.vstack((metrics_to_print, __metrics_dict[metric]))

                    metrics_table = pd.DataFrame(metrics_to_print, columns=__clusters_names, index=['Observations', 'Min', 'Max'] +  metrics)
                    generate_tex_table(metrics_table, float_format=float_format)

                if item=='pandas':

                    import pandas as pd
                    from IPython.display import display
                    pandas_format = '{:,' + float_format + '}'

                    metrics_to_print = np.hstack((np.array(self.__cluster_populations)[:,None], np.array(self.__cluster_min)[:,None], np.array(self.__cluster_max)[:,None]))

                    for metric in metrics:
                        metrics_to_print = np.hstack((metrics_to_print, np.array(__metrics_dict[metric])[:,None]))

                    metrics_table = pd.DataFrame(metrics_to_print, columns=['Observations', 'Min', 'Max'] + metrics, index=__clusters_names)

                    metrics_table['Observations'] = metrics_table['Observations'].astype(int)
                    metrics_table['Min'] = metrics_table['Min'].map(pandas_format.format)
                    metrics_table['Max'] = metrics_table['Max'].map(pandas_format.format)
                    for metric in metrics:
                        metrics_table[metric] = metrics_table[metric].map(pandas_format.format)

                    display(metrics_table)

        else:

            for item in set(table_format):

                if item=='raw':

                    for i in range(0,self.__n_clusters):

                        print('-'*25 + '\n' + __clusters_names[i])

                        metrics_to_print = [self.__cluster_populations[i], self.__cluster_min[i], self.__cluster_max[i]]
                        comparison_metrics_to_print = [self.__cluster_populations[i], self.__cluster_min[i], self.__cluster_max[i]]
                        for metric in metrics:
                            metrics_to_print.append(__metrics_dict[metric][i])
                            comparison_metrics_to_print.append(__comparison_metrics_dict[metric][i])

                        print('Observations' + ':\t' + str(metrics_to_print[0]))
                        print('Min' + ':\t' + ('%' + float_format) % metrics_to_print[1])
                        print('Max' + ':\t' + ('%' + float_format) % metrics_to_print[2])
                        for j, metric in enumerate(metrics):

                            if metric=='R2':
                                if metrics_to_print[j+3] > comparison_metrics_to_print[j+3]:
                                    print(metric + ':\t' + ('%' + float_format) % metrics_to_print[j+3] + colored('\tBETTER', 'green'))
                                elif metrics_to_print[j+3] < comparison_metrics_to_print[j+3]:
                                    print(metric + ':\t' + ('%' + float_format) % metrics_to_print[j+3] + colored('\tWORSE', 'red'))
                                elif metrics_to_print[j+3] == comparison_metrics_to_print[j+3]:
                                    print(metric + ':\t' + ('%' + float_format) % metrics_to_print[j+3] + '\tSAME')
                            else:
                                if metrics_to_print[j+3] > comparison_metrics_to_print[j+3]:
                                    print(metric + ':\t' + ('%' + float_format) % metrics_to_print[j+3] + colored('\tWORSE', 'red'))
                                elif metrics_to_print[j+3] < comparison_metrics_to_print[j+3]:
                                    print(metric + ':\t' + ('%' + float_format) % metrics_to_print[j+3] + colored('\tBETTER', 'green'))
                                elif metrics_to_print[j+3] == comparison_metrics_to_print[j+3]:
                                    print(metric + ':\t' + ('%' + float_format) % metrics_to_print[j+3] + '\tSAME')

                if item=='pandas':

                    import pandas as pd
                    from IPython.display import display
                    pandas_format = '{:,' + float_format + '}'

                    metrics_to_print = np.hstack((np.array(self.__cluster_populations)[:,None], np.array(self.__cluster_min)[:,None], np.array(self.__cluster_max)[:,None]))
                    comparison_metrics_to_print = np.hstack((np.array(self.__cluster_populations)[:,None], np.array(self.__cluster_min)[:,None], np.array(self.__cluster_max)[:,None]))
                    for metric in metrics:
                        metrics_to_print = np.hstack((metrics_to_print, np.array(__metrics_dict[metric])[:,None]))
                        comparison_metrics_to_print = np.hstack((comparison_metrics_to_print, np.array(__comparison_metrics_dict[metric])[:,None]))

                    def highlight_better(data, data_comparison, color='lightgreen'):

                        attr = 'background-color: {}'.format(color)

                        is_better = False * data

                        # Lower value is better (MAE, MSE, RMSE, NRMSE):
                        try:
                            is_better['MAE'] = data['MAE'].astype(float) < data_comparison['MAE']
                        except:
                            pass
                        try:
                            is_better['MSE'] = data['MSE'].astype(float) < data_comparison['MSE']
                        except:
                            pass
                        try:
                            is_better['MSLE'] = data['MSLE'].astype(float) < data_comparison['MSLE']
                        except:
                            pass
                        try:
                            is_better['RMSE'] = data['RMSE'].astype(float) < data_comparison['RMSE']
                        except:
                            pass
                        try:
                            is_better['NRMSE'] = data['NRMSE'].astype(float) < data_comparison['NRMSE']
                        except:
                            pass

                        # Higher value is better (R2):
                        try:
                            is_better['R2'] = data['R2'].astype(float) > data_comparison['R2']
                        except:
                            pass

                        formatting = [attr if v else '' for v in is_better]

                        formatting = pd.DataFrame(np.where(is_better, attr, ''), index=data.index, columns=data.columns)

                        return formatting

                    def highlight_worse(data, data_comparison, color='salmon'):

                        attr = 'background-color: {}'.format(color)

                        is_worse = False * data

                        # Higher value is worse (MAE, MSE, RMSE, NRMSE):
                        try:
                            is_worse['MAE'] = data['MAE'].astype(float) > data_comparison['MAE']
                        except:
                            pass
                        try:
                            is_worse['MSE'] = data['MSE'].astype(float) > data_comparison['MSE']
                        except:
                            pass
                        try:
                            is_worse['MSLE'] = data['MSLE'].astype(float) > data_comparison['MSLE']
                        except:
                            pass
                        try:
                            is_worse['RMSE'] = data['RMSE'].astype(float) > data_comparison['RMSE']
                        except:
                            pass
                        try:
                            is_worse['NRMSE'] = data['NRMSE'].astype(float) > data_comparison['NRMSE']
                        except:
                            pass

                        # Lower value is worse (R2):
                        try:
                            is_worse['R2'] = data['R2'].astype(float) < data_comparison['R2']
                        except:
                            pass

                        formatting = [attr if v else '' for v in is_worse]

                        formatting = pd.DataFrame(np.where(is_worse, attr, ''), index=data.index, columns=data.columns)

                        return formatting

                    metrics_table = pd.DataFrame(metrics_to_print, columns=['Observations', 'Min', 'Max'] + metrics, index=__clusters_names)
                    comparison_metrics_table = pd.DataFrame(comparison_metrics_to_print, columns=['Observations', 'Min', 'Max'] + metrics, index=__clusters_names)

                    metrics_table['Observations'] = metrics_table['Observations'].astype(int)
                    metrics_table['Min'] = metrics_table['Min'].map(pandas_format.format)
                    metrics_table['Max'] = metrics_table['Max'].map(pandas_format.format)
                    for metric in metrics:
                        metrics_table[metric] = metrics_table[metric].map(pandas_format.format)

                    formatted_table = metrics_table.style.apply(highlight_better, data_comparison=comparison_metrics_table, axis=None)\
                                                         .apply(highlight_worse, data_comparison=comparison_metrics_table, axis=None)

                    display(formatted_table)

# ------------------------------------------------------------------------------

def coefficient_of_determination(observed, predicted):
    """
    Computes the coefficient of determination, :math:`R^2`, value:

    .. math::

        R^2 = 1 - \\frac{\\sum_{i=1}^N (\\phi_{o,i} - \\phi_{p,i})^2}{\\sum_{i=1}^N (\\phi_{o,i} - \\mathrm{mean}(\\phi_{o,i}))^2}

    where :math:`N` is the number of observations, :math:`\\phi_o` is the observed and
    :math:`\\phi_p` is the predicted dependent variable.

    **Example:**

    .. code:: python

        from PCAfold import PCA, coefficient_of_determination
        import numpy as np

        # Generate dummy data set:
        X = np.random.rand(100,3)

        # Instantiate PCA class object:
        pca_X = PCA(X, scaling='auto', n_components=2)

        # Approximate the data set:
        X_rec = pca_X.reconstruct(pca_X.transform(X))

        # Compute the coefficient of determination for the first variable:
        r2 = coefficient_of_determination(X[:,0], X_rec[:,0])

    :param observed:
        ``numpy.ndarray`` specifying the observed values of a single dependent variable, :math:`\\phi_o`. It should be of size ``(n_observations,)`` or ``(n_observations, 1)``.
    :param predicted:
        ``numpy.ndarray`` specifying the predicted values of a single dependent variable, :math:`\\phi_p`. It should be of size ``(n_observations,)`` or ``(n_observations, 1)``.

    :return:
        - **r2** - coefficient of determination, :math:`R^2`.
    """

    if not isinstance(observed, np.ndarray):
        raise ValueError("Parameter `observed` has to be of type `numpy.ndarray`.")

    try:
        (n_observed,) = np.shape(observed)
        n_var_observed = 1
    except:
        (n_observed, n_var_observed) = np.shape(observed)

    if n_var_observed != 1:
        raise ValueError("Parameter `observed` has to be a 0D or 1D vector.")

    if not isinstance(predicted, np.ndarray):
        raise ValueError("Parameter `predicted` has to be of type `numpy.ndarray`.")

    try:
        (n_predicted,) = np.shape(predicted)
        n_var_predicted = 1
    except:
        (n_predicted, n_var_predicted) = np.shape(predicted)

    if n_var_predicted != 1:
        raise ValueError("Parameter `predicted` has to be a 0D or 1D vector.")

    if n_observed != n_predicted:
        raise ValueError("Parameter `observed` has different number of elements than `predicted`.")

    __observed = observed.ravel()
    __predicted = predicted.ravel()

    r2 = 1. - np.sum((__observed - __predicted) * (__observed - __predicted)) / np.sum(
        (__observed - np.mean(__observed)) * (__observed - np.mean(__observed)))

    return r2

# ------------------------------------------------------------------------------

def stratified_coefficient_of_determination(observed, predicted, idx, use_global_mean=True, verbose=False):
    """
    Computes the stratified coefficient of determination,
    :math:`R^2`, values. Stratified :math:`R^2` is computed separately in each
    bin (cluster) of an observed dependent variable, :math:`\\phi_o`.

    :math:`R_j^2` in the :math:`j^{th}` bin can be computed in two ways:

    - If ``use_global_mean=True``, the mean of the entire observed variable is used as a reference:

    .. math::

        R_j^2 = 1 - \\frac{\\sum_{i=1}^{N_j} (\\phi_{o,i}^{j} - \\phi_{p,i}^{j})^2}{\\sum_{i=1}^{N_j} (\\phi_{o,i}^{j} - \\mathrm{mean}(\\phi_o))^2}

    - If ``use_global_mean=False``, the mean of the considered :math:`j^{th}` bin is used as a reference:

    .. math::

        R_j^2 = 1 - \\frac{\\sum_{i=1}^{N_j} (\\phi_{o,i}^{j} - \\phi_{p,i}^{j})^2}{\\sum_{i=1}^{N_j} (\\phi_{o,i}^{j} - \\mathrm{mean}(\\phi_o^{j}))^2}

    where :math:`N_j` is the number of observations in the :math:`j^{th}` bin and
    :math:`\\phi_p` is the predicted dependent variable.

    .. note::

        After running this function you can call
        ``analysis.plot_stratified_coefficient_of_determination(r2_in_bins, bins_borders)`` on the
        function outputs and it will visualize how stratified :math:`R^2` changes across bins.

    .. warning::

        The stratified :math:`R^2` metric can be misleading if there are large
        variations in point density in an observed variable. For instance, below is a data set
        composed of lines of points that have uniform spacing on the :math:`x` axis
        but become more and more sparse in the direction of increasing :math:`\\phi`
        due to an increasing gradient of :math:`\\phi`.
        If bins are narrow enough (number of bins is high enough), a single bin
        (like the bin bounded by the red dashed lines) can contain only one of
        those lines of points for high value of :math:`\\phi`. :math:`R^2` will then be computed
        for constant, or almost constant observations, even though globally those
        observations lie in a location of a large gradient of the observed variable!

        .. image:: ../images/stratified-r2.png
            :width: 500
            :align: center

    **Example:**

    .. code:: python

        from PCAfold import PCA, variable_bins, stratified_coefficient_of_determination, plot_stratified_coefficient_of_determination
        import numpy as np

        # Generate dummy data set:
        X = np.random.rand(100,10)

        # Instantiate PCA class object:
        pca_X = PCA(X, scaling='auto', n_components=2)

        # Approximate the data set:
        X_rec = pca_X.reconstruct(pca_X.transform(X))

        # Generate bins:
        (idx, bins_borders) = variable_bins(X[:,0], k=10, verbose=False)

        # Compute stratified R2 in 10 bins of the first variable in a data set:
        r2_in_bins = stratified_coefficient_of_determination(X[:,0], X_rec[:,0], idx=idx, use_global_mean=True, verbose=True)

        # Plot the stratified R2 values:
        plot_stratified_coefficient_of_determination(r2_in_bins, bins_borders)

    :param observed:
        ``numpy.ndarray`` specifying the observed values of a single dependent variable, :math:`\\phi_o`. It should be of size ``(n_observations,)`` or ``(n_observations, 1)``.
    :param predicted:
        ``numpy.ndarray`` specifying the predicted values of a single dependent variable, :math:`\\phi_p`. It should be of size ``(n_observations,)`` or ``(n_observations, 1)``.
    :param idx:
        ``numpy.ndarray`` of cluster classifications. It should be of size ``(n_observations,)`` or ``(n_observations,1)``.
    :param use_global_mean: (optional)
            ``bool`` specifying if global mean of the observed variable should be used as a reference in :math:`R^2` calculation.
    :param verbose: (optional)
        ``bool`` for printing sizes (number of observations) and :math:`R^2` values in each bin.

    :return:
        - **r2_in_bins** - ``list`` specifying the coefficients of determination :math:`R^2` in each bin. It has length ``k``.
    """

    if not isinstance(observed, np.ndarray):
        raise ValueError("Parameter `observed` has to be of type `numpy.ndarray`.")

    try:
        (n_observed,) = np.shape(observed)
        n_var_observed = 1
    except:
        (n_observed, n_var_observed) = np.shape(observed)

    if n_var_observed != 1:
        raise ValueError("Parameter `observed` has to be a 0D or 1D vector.")

    if not isinstance(predicted, np.ndarray):
        raise ValueError("Parameter `predicted` has to be of type `numpy.ndarray`.")

    try:
        (n_predicted,) = np.shape(predicted)
        n_var_predicted = 1
    except:
        (n_predicted, n_var_predicted) = np.shape(predicted)

    if n_var_predicted != 1:
        raise ValueError("Parameter `predicted` has to be a 0D or 1D vector.")

    if n_observed != n_predicted:
        raise ValueError("Parameter `observed` has different number of elements than `predicted`.")

    if isinstance(idx, np.ndarray):
        if not all(isinstance(i, np.integer) for i in idx.ravel()):
            raise ValueError("Parameter `idx` can only contain integers.")
    else:
        raise ValueError("Parameter `idx` has to be of type `numpy.ndarray`.")

    try:
        (n_observations_idx, ) = np.shape(idx)
        n_idx = 1
    except:
        (n_observations_idx, n_idx) = np.shape(idx)

    if n_idx != 1:
        raise ValueError("Parameter `idx` has to have size `(n_observations,)` or `(n_observations,1)`.")

    if n_observations_idx != n_observed:
        raise ValueError('Vector of cluster classifications `idx` has different number of observations than the original data set `X`.')

    if not isinstance(use_global_mean, bool):
        raise ValueError("Parameter `use_global_mean` has to be a boolean.")

    if not isinstance(verbose, bool):
        raise ValueError("Parameter `verbose` has to be a boolean.")

    __observed = observed.ravel()
    __predicted = predicted.ravel()

    r2_in_bins = []

    if use_global_mean:
        global_mean = np.mean(__observed)

    for cl in np.unique(idx):

        (idx_bin,) = np.where(idx==cl)

        if use_global_mean:
            r2 = 1. - np.sum((__observed[idx_bin] - __predicted[idx_bin]) * (__observed[idx_bin] - __predicted[idx_bin])) / np.sum(
                (__observed[idx_bin] - global_mean) * (__observed[idx_bin] - global_mean))
        else:
            r2 = coefficient_of_determination(__observed[idx_bin], __predicted[idx_bin])

        constant_bin_metric_min = np.min(__observed[idx_bin])/np.mean(__observed[idx_bin])
        constant_bin_metric_max = np.max(__observed[idx_bin])/np.mean(__observed[idx_bin])

        if verbose:
            if (abs(constant_bin_metric_min - 1) < 0.01) and (abs(constant_bin_metric_max - 1) < 0.01):
                print('Bin\t' + str(cl+1) + '\t| size\t ' + str(len(idx_bin)) + '\t| R2\t' + str(round(r2,6)) + '\t| ' + colored('This bin has almost constant values.', 'red'))
            else:
                print('Bin\t' + str(cl+1) + '\t| size\t ' + str(len(idx_bin)) + '\t| R2\t' + str(round(r2,6)))

        r2_in_bins.append(r2)

    return r2_in_bins

# ------------------------------------------------------------------------------

def mean_absolute_error(observed, predicted):
    """
    Computes the mean absolute error (MAE):

    .. math::

        \\mathrm{MAE} = \\frac{1}{N} \\sum_{i=1}^N | \\phi_{o,i} - \\phi_{p,i} |

    where :math:`N` is the number of observations, :math:`\\phi_o` is the observed and
    :math:`\\phi_p` is the predicted dependent variable.

    **Example:**

    .. code:: python

        from PCAfold import PCA, mean_absolute_error
        import numpy as np

        # Generate dummy data set:
        X = np.random.rand(100,3)

        # Instantiate PCA class object:
        pca_X = PCA(X, scaling='auto', n_components=2)

        # Approximate the data set:
        X_rec = pca_X.reconstruct(pca_X.transform(X))

        # Compute the mean absolute error for the first variable:
        mae = mean_absolute_error(X[:,0], X_rec[:,0])

    :param observed:
        ``numpy.ndarray`` specifying the observed values of a single dependent variable, :math:`\\phi_o`. It should be of size ``(n_observations,)`` or ``(n_observations, 1)``.
    :param predicted:
        ``numpy.ndarray`` specifying the predicted values of a single dependent variable, :math:`\\phi_p`. It should be of size ``(n_observations,)`` or ``(n_observations, 1)``.

    :return:
        - **mae** - mean absolute error (MAE).
    """

    if not isinstance(observed, np.ndarray):
        raise ValueError("Parameter `observed` has to be of type `numpy.ndarray`.")

    try:
        (n_observed,) = np.shape(observed)
        n_var_observed = 1
    except:
        (n_observed, n_var_observed) = np.shape(observed)

    if n_var_observed != 1:
        raise ValueError("Parameter `observed` has to be a 0D or 1D vector.")

    if not isinstance(predicted, np.ndarray):
        raise ValueError("Parameter `predicted` has to be of type `numpy.ndarray`.")

    try:
        (n_predicted,) = np.shape(predicted)
        n_var_predicted = 1
    except:
        (n_predicted, n_var_predicted) = np.shape(predicted)

    if n_var_predicted != 1:
        raise ValueError("Parameter `predicted` has to be a 0D or 1D vector.")

    if n_observed != n_predicted:
        raise ValueError("Parameter `observed` has different number of elements than `predicted`.")

    __observed = observed.ravel()
    __predicted = predicted.ravel()

    mae = np.sum(abs(__observed - __predicted)) / n_observed

    return mae

# ------------------------------------------------------------------------------

def stratified_mean_absolute_error(observed, predicted, idx, verbose=False):
    """
    Computes the stratified mean absolute error (MAE) values. Stratified MAE is computed separately in each
    bin (cluster) of an observed dependent variable, :math:`\\phi_o`.

    MAE in the :math:`j^{th}` bin can be computed as:

    .. math::

        \\mathrm{MAE}_j = \\frac{1}{N_j} \\sum_{i=1}^{N_j} | \\phi_{o,i}^j - \\phi_{p,i}^j |

    where :math:`N_j` is the number of observations in the :math:`j^{th}` bin, :math:`\\phi_o` is the observed and
    :math:`\\phi_p` is the predicted dependent variable.

    **Example:**

    .. code:: python

        from PCAfold import PCA, variable_bins, stratified_mean_absolute_error
        import numpy as np

        # Generate dummy data set:
        X = np.random.rand(100,10)

        # Instantiate PCA class object:
        pca_X = PCA(X, scaling='auto', n_components=2)

        # Approximate the data set:
        X_rec = pca_X.reconstruct(pca_X.transform(X))

        # Generate bins:
        (idx, bins_borders) = variable_bins(X[:,0], k=10, verbose=False)

        # Compute stratified MAE in 10 bins of the first variable in a data set:
        mae_in_bins = stratified_mean_absolute_error(X[:,0], X_rec[:,0], idx=idx, verbose=True)

    :param observed:
        ``numpy.ndarray`` specifying the observed values of a single dependent variable, :math:`\\phi_o`. It should be of size ``(n_observations,)`` or ``(n_observations, 1)``.
    :param predicted:
        ``numpy.ndarray`` specifying the predicted values of a single dependent variable, :math:`\\phi_p`. It should be of size ``(n_observations,)`` or ``(n_observations, 1)``.
    :param idx:
        ``numpy.ndarray`` of cluster classifications. It should be of size ``(n_observations,)`` or ``(n_observations,1)``.
    :param verbose: (optional)
        ``bool`` for printing sizes (number of observations) and MAE values in each bin.

    :return:
        - **mae_in_bins** - ``list`` specifying the mean absolute error (MAE) in each bin. It has length ``k``.
    """

    if not isinstance(observed, np.ndarray):
        raise ValueError("Parameter `observed` has to be of type `numpy.ndarray`.")

    try:
        (n_observed,) = np.shape(observed)
        n_var_observed = 1
    except:
        (n_observed, n_var_observed) = np.shape(observed)

    if n_var_observed != 1:
        raise ValueError("Parameter `observed` has to be a 0D or 1D vector.")

    if not isinstance(predicted, np.ndarray):
        raise ValueError("Parameter `predicted` has to be of type `numpy.ndarray`.")

    try:
        (n_predicted,) = np.shape(predicted)
        n_var_predicted = 1
    except:
        (n_predicted, n_var_predicted) = np.shape(predicted)

    if n_var_predicted != 1:
        raise ValueError("Parameter `predicted` has to be a 0D or 1D vector.")

    if n_observed != n_predicted:
        raise ValueError("Parameter `observed` has different number of elements than `predicted`.")

    if isinstance(idx, np.ndarray):
        if not all(isinstance(i, np.integer) for i in idx.ravel()):
            raise ValueError("Parameter `idx` can only contain integers.")
    else:
        raise ValueError("Parameter `idx` has to be of type `numpy.ndarray`.")

    try:
        (n_observations_idx, ) = np.shape(idx)
        n_idx = 1
    except:
        (n_observations_idx, n_idx) = np.shape(idx)

    if n_idx != 1:
        raise ValueError("Parameter `idx` has to have size `(n_observations,)` or `(n_observations,1)`.")

    if n_observations_idx != n_observed:
        raise ValueError('Vector of cluster classifications `idx` has different number of observations than the original data set `X`.')

    if not isinstance(verbose, bool):
        raise ValueError("Parameter `verbose` has to be a boolean.")

    __observed = observed.ravel()
    __predicted = predicted.ravel()

    mae_in_bins = []

    for cl in np.unique(idx):

        (idx_bin,) = np.where(idx==cl)

        mae = mean_absolute_error(__observed[idx_bin], __predicted[idx_bin])

        constant_bin_metric_min = np.min(__observed[idx_bin])/np.mean(__observed[idx_bin])
        constant_bin_metric_max = np.max(__observed[idx_bin])/np.mean(__observed[idx_bin])

        if verbose:
            if (abs(constant_bin_metric_min - 1) < 0.01) and (abs(constant_bin_metric_max - 1) < 0.01):
                print('Bin\t' + str(cl+1) + '\t| size\t ' + str(len(idx_bin)) + '\t| MAE\t' + str(round(mae,6)) + '\t| ' + colored('This bin has almost constant values.', 'red'))
            else:
                print('Bin\t' + str(cl+1) + '\t| size\t ' + str(len(idx_bin)) + '\t| MAE\t' + str(round(mae,6)))

        mae_in_bins.append(mae)

    return mae_in_bins

# ------------------------------------------------------------------------------

def max_absolute_error(observed, predicted):
    """
    Computes the maximum absolute error (MaxAE):

    .. math::

        \\mathrm{MaxAE} = \\mathrm{max}( | \\phi_{o,i} - \\phi_{p,i} | )

    where :math:`\\phi_o` is the observed and
    :math:`\\phi_p` is the predicted dependent variable.

    **Example:**

    .. code:: python

        from PCAfold import PCA, max_absolute_error
        import numpy as np

        # Generate dummy data set:
        X = np.random.rand(100,3)

        # Instantiate PCA class object:
        pca_X = PCA(X, scaling='auto', n_components=2)

        # Approximate the data set:
        X_rec = pca_X.reconstruct(pca_X.transform(X))

        # Compute the maximum absolute error for the first variable:
        maxae = max_absolute_error(X[:,0], X_rec[:,0])

    :param observed:
        ``numpy.ndarray`` specifying the observed values of a single dependent variable, :math:`\\phi_o`. It should be of size ``(n_observations,)`` or ``(n_observations, 1)``.
    :param predicted:
        ``numpy.ndarray`` specifying the predicted values of a single dependent variable, :math:`\\phi_p`. It should be of size ``(n_observations,)`` or ``(n_observations, 1)``.

    :return:
        - **maxae** - max absolute error (MaxAE).
    """

    if not isinstance(observed, np.ndarray):
        raise ValueError("Parameter `observed` has to be of type `numpy.ndarray`.")

    try:
        (n_observed,) = np.shape(observed)
        n_var_observed = 1
    except:
        (n_observed, n_var_observed) = np.shape(observed)

    if n_var_observed != 1:
        raise ValueError("Parameter `observed` has to be a 0D or 1D vector.")

    if not isinstance(predicted, np.ndarray):
        raise ValueError("Parameter `predicted` has to be of type `numpy.ndarray`.")

    try:
        (n_predicted,) = np.shape(predicted)
        n_var_predicted = 1
    except:
        (n_predicted, n_var_predicted) = np.shape(predicted)

    if n_var_predicted != 1:
        raise ValueError("Parameter `predicted` has to be a 0D or 1D vector.")

    if n_observed != n_predicted:
        raise ValueError("Parameter `observed` has different number of elements than `predicted`.")

    __observed = observed.ravel()
    __predicted = predicted.ravel()

    maxae = np.max(abs(__observed - __predicted))

    return maxae

# ------------------------------------------------------------------------------

def mean_squared_error(observed, predicted):
    """
    Computes the mean squared error (MSE):

    .. math::

        \\mathrm{MSE} = \\frac{1}{N} \\sum_{i=1}^N (\\phi_{o,i} - \\phi_{p,i}) ^2

    where :math:`N` is the number of observations, :math:`\\phi_o` is the observed and
    :math:`\\phi_p` is the predicted dependent variable.

    **Example:**

    .. code:: python

        from PCAfold import PCA, mean_squared_error
        import numpy as np

        # Generate dummy data set:
        X = np.random.rand(100,3)

        # Instantiate PCA class object:
        pca_X = PCA(X, scaling='auto', n_components=2)

        # Approximate the data set:
        X_rec = pca_X.reconstruct(pca_X.transform(X))

        # Compute the mean squared error for the first variable:
        mse = mean_squared_error(X[:,0], X_rec[:,0])

    :param observed:
        ``numpy.ndarray`` specifying the observed values of a single dependent variable, :math:`\\phi_o`. It should be of size ``(n_observations,)`` or ``(n_observations, 1)``.
    :param predicted:
        ``numpy.ndarray`` specifying the predicted values of a single dependent variable, :math:`\\phi_p`. It should be of size ``(n_observations,)`` or ``(n_observations, 1)``.

    :return:
        - **mse** - mean squared error (MSE).
    """

    if not isinstance(observed, np.ndarray):
        raise ValueError("Parameter `observed` has to be of type `numpy.ndarray`.")

    try:
        (n_observed,) = np.shape(observed)
        n_var_observed = 1
    except:
        (n_observed, n_var_observed) = np.shape(observed)

    if n_var_observed != 1:
        raise ValueError("Parameter `observed` has to be a 0D or 1D vector.")

    if not isinstance(predicted, np.ndarray):
        raise ValueError("Parameter `predicted` has to be of type `numpy.ndarray`.")

    try:
        (n_predicted,) = np.shape(predicted)
        n_var_predicted = 1
    except:
        (n_predicted, n_var_predicted) = np.shape(predicted)

    if n_var_predicted != 1:
        raise ValueError("Parameter `predicted` has to be a 0D or 1D vector.")

    if n_observed != n_predicted:
        raise ValueError("Parameter `observed` has different number of elements than `predicted`.")

    __observed = observed.ravel()
    __predicted = predicted.ravel()

    mse = 1.0 / n_observed * np.sum((__observed - __predicted) * (__observed - __predicted))

    return mse

# ------------------------------------------------------------------------------

def stratified_mean_squared_error(observed, predicted, idx, verbose=False):
    """
    Computes the stratified mean squared error (MSE) values. Stratified MSE is computed separately in each
    bin (cluster) of an observed dependent variable, :math:`\\phi_o`.

    MSE in the :math:`j^{th}` bin can be computed as:

    .. math::

        \\mathrm{MSE}_j = \\frac{1}{N_j} \\sum_{i=1}^{N_j} (\\phi_{o,i}^j - \\phi_{p,i}^j) ^2

    where :math:`N_j` is the number of observations in the :math:`j^{th}` bin, :math:`\\phi_o` is the observed and
    :math:`\\phi_p` is the predicted dependent variable.

    **Example:**

    .. code:: python

        from PCAfold import PCA, variable_bins, stratified_mean_squared_error
        import numpy as np

        # Generate dummy data set:
        X = np.random.rand(100,10)

        # Instantiate PCA class object:
        pca_X = PCA(X, scaling='auto', n_components=2)

        # Approximate the data set:
        X_rec = pca_X.reconstruct(pca_X.transform(X))

        # Generate bins:
        (idx, bins_borders) = variable_bins(X[:,0], k=10, verbose=False)

        # Compute stratified MSE in 10 bins of the first variable in a data set:
        mse_in_bins = stratified_mean_squared_error(X[:,0], X_rec[:,0], idx=idx, verbose=True)

    :param observed:
        ``numpy.ndarray`` specifying the observed values of a single dependent variable, :math:`\\phi_o`. It should be of size ``(n_observations,)`` or ``(n_observations, 1)``.
    :param predicted:
        ``numpy.ndarray`` specifying the predicted values of a single dependent variable, :math:`\\phi_p`. It should be of size ``(n_observations,)`` or ``(n_observations, 1)``.
    :param idx:
        ``numpy.ndarray`` of cluster classifications. It should be of size ``(n_observations,)`` or ``(n_observations,1)``.
    :param verbose: (optional)
        ``bool`` for printing sizes (number of observations) and MSE values in each bin.

    :return:
        - **mse_in_bins** - ``list`` specifying the mean squared error (MSE) in each bin. It has length ``k``.
    """

    if not isinstance(observed, np.ndarray):
        raise ValueError("Parameter `observed` has to be of type `numpy.ndarray`.")

    try:
        (n_observed,) = np.shape(observed)
        n_var_observed = 1
    except:
        (n_observed, n_var_observed) = np.shape(observed)

    if n_var_observed != 1:
        raise ValueError("Parameter `observed` has to be a 0D or 1D vector.")

    if not isinstance(predicted, np.ndarray):
        raise ValueError("Parameter `predicted` has to be of type `numpy.ndarray`.")

    try:
        (n_predicted,) = np.shape(predicted)
        n_var_predicted = 1
    except:
        (n_predicted, n_var_predicted) = np.shape(predicted)

    if n_var_predicted != 1:
        raise ValueError("Parameter `predicted` has to be a 0D or 1D vector.")

    if n_observed != n_predicted:
        raise ValueError("Parameter `observed` has different number of elements than `predicted`.")

    if isinstance(idx, np.ndarray):
        if not all(isinstance(i, np.integer) for i in idx.ravel()):
            raise ValueError("Parameter `idx` can only contain integers.")
    else:
        raise ValueError("Parameter `idx` has to be of type `numpy.ndarray`.")

    try:
        (n_observations_idx, ) = np.shape(idx)
        n_idx = 1
    except:
        (n_observations_idx, n_idx) = np.shape(idx)

    if n_idx != 1:
        raise ValueError("Parameter `idx` has to have size `(n_observations,)` or `(n_observations,1)`.")

    if n_observations_idx != n_observed:
        raise ValueError('Vector of cluster classifications `idx` has different number of observations than the original data set `X`.')

    if not isinstance(verbose, bool):
        raise ValueError("Parameter `verbose` has to be a boolean.")

    __observed = observed.ravel()
    __predicted = predicted.ravel()

    mse_in_bins = []

    for cl in np.unique(idx):

        (idx_bin,) = np.where(idx==cl)

        mse = mean_squared_error(__observed[idx_bin], __predicted[idx_bin])

        constant_bin_metric_min = np.min(__observed[idx_bin])/np.mean(__observed[idx_bin])
        constant_bin_metric_max = np.max(__observed[idx_bin])/np.mean(__observed[idx_bin])

        if verbose:
            if (abs(constant_bin_metric_min - 1) < 0.01) and (abs(constant_bin_metric_max - 1) < 0.01):
                print('Bin\t' + str(cl+1) + '\t| size\t ' + str(len(idx_bin)) + '\t| MSE\t' + str(round(mse,6)) + '\t| ' + colored('This bin has almost constant values.', 'red'))
            else:
                print('Bin\t' + str(cl+1) + '\t| size\t ' + str(len(idx_bin)) + '\t| MSE\t' + str(round(mse,6)))

        mse_in_bins.append(mse)

    return mse_in_bins

# ------------------------------------------------------------------------------

def mean_squared_logarithmic_error(observed, predicted):
    """
    Computes the mean squared logarithmic error (MSLE):

    .. math::

        \\mathrm{MSLE} = \\frac{1}{N} \\sum_{i=1}^N (\\log(\\phi_{o,i} + 1) - \\log(\\phi_{p,i} + 1)) ^2

    where :math:`N` is the number of observations, :math:`\\phi_o` is the observed and
    :math:`\\phi_p` is the predicted dependent variable.

    .. warning::

        The MSLE metric can only be used on non-negative samples.

    **Example:**

    .. code:: python

        from PCAfold import PCA, mean_squared_logarithmic_error
        import numpy as np

        # Generate dummy data set:
        X = np.random.rand(100,3)

        # Instantiate PCA class object:
        pca_X = PCA(X, scaling='auto', n_components=2)

        # Approximate the data set:
        X_rec = pca_X.reconstruct(pca_X.transform(X))

        # Compute the mean squared error for the first variable:
        msle = mean_squared_logarithmic_error(X[:,0], X_rec[:,0])

    :param observed:
        ``numpy.ndarray`` specifying the observed values of a single dependent variable, :math:`\\phi_o`. It should be of size ``(n_observations,)`` or ``(n_observations, 1)``.
    :param predicted:
        ``numpy.ndarray`` specifying the predicted values of a single dependent variable, :math:`\\phi_p`. It should be of size ``(n_observations,)`` or ``(n_observations, 1)``.

    :return:
        - **msle** - mean squared logarithmic error (MSLE).
    """

    if not isinstance(observed, np.ndarray):
        raise ValueError("Parameter `observed` has to be of type `numpy.ndarray`.")

    try:
        (n_observed,) = np.shape(observed)
        n_var_observed = 1
    except:
        (n_observed, n_var_observed) = np.shape(observed)

    if n_var_observed != 1:
        raise ValueError("Parameter `observed` has to be a 0D or 1D vector.")

    if not isinstance(predicted, np.ndarray):
        raise ValueError("Parameter `predicted` has to be of type `numpy.ndarray`.")

    try:
        (n_predicted,) = np.shape(predicted)
        n_var_predicted = 1
    except:
        (n_predicted, n_var_predicted) = np.shape(predicted)

    if n_var_predicted != 1:
        raise ValueError("Parameter `predicted` has to be a 0D or 1D vector.")

    if n_observed != n_predicted:
        raise ValueError("Parameter `observed` has different number of elements than `predicted`.")

    if np.any(observed<0) or np.any(predicted<0):
        raise ValueError("Parameters `observed` and `predicted` can only contain non-negative samples.")

    __observed = observed.ravel()
    __predicted = predicted.ravel()

    msle = 1.0 / n_observed * np.sum((np.log(__observed + 1) - np.log(__predicted+1)) * (np.log(__observed + 1) - np.log(__predicted+1)))

    return msle

# ------------------------------------------------------------------------------

def stratified_mean_squared_logarithmic_error(observed, predicted, idx, verbose=False):
    """
    Computes the stratified mean squared logarithmic error (MSLE) values. Stratified MSLE is computed separately in each
    bin (cluster) of an observed dependent variable, :math:`\\phi_o`.

    MSLE in the :math:`j^{th}` bin can be computed as:

    .. math::

        \\mathrm{MSLE}_j = \\frac{1}{N_j} \\sum_{i=1}^{N_j} (\\log(\\phi_{o,i}^j + 1) - \\log(\\phi_{p,i}^j + 1)) ^2

    where :math:`N_j` is the number of observations in the :math:`j^{th}` bin, :math:`\\phi_o` is the observed and
    :math:`\\phi_p` is the predicted dependent variable.

    .. warning::

        The MSLE metric can only be used on non-negative samples.

    **Example:**

    .. code:: python

        from PCAfold import PCA, variable_bins, stratified_mean_squared_logarithmic_error
        import numpy as np

        # Generate dummy data set:
        X = np.random.rand(100,10)

        # Instantiate PCA class object:
        pca_X = PCA(X, scaling='auto', n_components=2)

        # Approximate the data set:
        X_rec = pca_X.reconstruct(pca_X.transform(X))

        # Generate bins:
        (idx, bins_borders) = variable_bins(X[:,0], k=10, verbose=False)

        # Compute stratified MSLE in 10 bins of the first variable in a data set:
        msle_in_bins = stratified_mean_squared_logarithmic_error(X[:,0], X_rec[:,0], idx=idx, verbose=True)

    :param observed:
        ``numpy.ndarray`` specifying the observed values of a single dependent variable, :math:`\\phi_o`. It should be of size ``(n_observations,)`` or ``(n_observations, 1)``.
    :param predicted:
        ``numpy.ndarray`` specifying the predicted values of a single dependent variable, :math:`\\phi_p`. It should be of size ``(n_observations,)`` or ``(n_observations, 1)``.
    :param idx:
        ``numpy.ndarray`` of cluster classifications. It should be of size ``(n_observations,)`` or ``(n_observations,1)``.
    :param verbose: (optional)
        ``bool`` for printing sizes (number of observations) and MSLE values in each bin.

    :return:
        - **msle_in_bins** - ``list`` specifying the mean squared logarithmic error (MSLE) in each bin. It has length ``k``.
    """

    if not isinstance(observed, np.ndarray):
        raise ValueError("Parameter `observed` has to be of type `numpy.ndarray`.")

    try:
        (n_observed,) = np.shape(observed)
        n_var_observed = 1
    except:
        (n_observed, n_var_observed) = np.shape(observed)

    if n_var_observed != 1:
        raise ValueError("Parameter `observed` has to be a 0D or 1D vector.")

    if not isinstance(predicted, np.ndarray):
        raise ValueError("Parameter `predicted` has to be of type `numpy.ndarray`.")

    try:
        (n_predicted,) = np.shape(predicted)
        n_var_predicted = 1
    except:
        (n_predicted, n_var_predicted) = np.shape(predicted)

    if n_var_predicted != 1:
        raise ValueError("Parameter `predicted` has to be a 0D or 1D vector.")

    if n_observed != n_predicted:
        raise ValueError("Parameter `observed` has different number of elements than `predicted`.")

    if np.any(observed<0) or np.any(predicted<0):
        raise ValueError("Parameters `observed` and `predicted` can only contain non-negative samples.")

    if isinstance(idx, np.ndarray):
        if not all(isinstance(i, np.integer) for i in idx.ravel()):
            raise ValueError("Parameter `idx` can only contain integers.")
    else:
        raise ValueError("Parameter `idx` has to be of type `numpy.ndarray`.")

    try:
        (n_observations_idx, ) = np.shape(idx)
        n_idx = 1
    except:
        (n_observations_idx, n_idx) = np.shape(idx)

    if n_idx != 1:
        raise ValueError("Parameter `idx` has to have size `(n_observations,)` or `(n_observations,1)`.")

    if n_observations_idx != n_observed:
        raise ValueError('Vector of cluster classifications `idx` has different number of observations than the original data set `X`.')

    if not isinstance(verbose, bool):
        raise ValueError("Parameter `verbose` has to be a boolean.")

    __observed = observed.ravel()
    __predicted = predicted.ravel()

    msle_in_bins = []

    for cl in np.unique(idx):

        (idx_bin,) = np.where(idx==cl)

        msle = mean_squared_logarithmic_error(__observed[idx_bin], __predicted[idx_bin])

        constant_bin_metric_min = np.min(__observed[idx_bin])/np.mean(__observed[idx_bin])
        constant_bin_metric_max = np.max(__observed[idx_bin])/np.mean(__observed[idx_bin])

        if verbose:
            if (abs(constant_bin_metric_min - 1) < 0.01) and (abs(constant_bin_metric_max - 1) < 0.01):
                print('Bin\t' + str(cl+1) + '\t| size\t ' + str(len(idx_bin)) + '\t| MSLE\t' + str(round(msle,6)) + '\t| ' + colored('This bin has almost constant values.', 'red'))
            else:
                print('Bin\t' + str(cl+1) + '\t| size\t ' + str(len(idx_bin)) + '\t| MSLE\t' + str(round(msle,6)))

        msle_in_bins.append(msle)

    return msle_in_bins

# ------------------------------------------------------------------------------

def root_mean_squared_error(observed, predicted):
    """
    Computes the root mean squared error (RMSE):

    .. math::

        \\mathrm{RMSE} = \\sqrt{\\frac{1}{N} \\sum_{i=1}^N (\\phi_{o,i} - \\phi_{p,i}) ^2}

    where :math:`N` is the number of observations, :math:`\\phi_o` is the observed and
    :math:`\\phi_p` is the predicted dependent variable.

    **Example:**

    .. code:: python

        from PCAfold import PCA, root_mean_squared_error
        import numpy as np

        # Generate dummy data set:
        X = np.random.rand(100,3)

        # Instantiate PCA class object:
        pca_X = PCA(X, scaling='auto', n_components=2)

        # Approximate the data set:
        X_rec = pca_X.reconstruct(pca_X.transform(X))

        # Compute the root mean squared error for the first variable:
        rmse = root_mean_squared_error(X[:,0], X_rec[:,0])

    :param observed:
        ``numpy.ndarray`` specifying the observed values of a single dependent variable, :math:`\\phi_o`. It should be of size ``(n_observations,)`` or ``(n_observations, 1)``.
    :param predicted:
        ``numpy.ndarray`` specifying the predicted values of a single dependent variable, :math:`\\phi_p`. It should be of size ``(n_observations,)`` or ``(n_observations, 1)``.

    :return:
        - **rmse** - root mean squared error (RMSE).
    """

    if not isinstance(observed, np.ndarray):
        raise ValueError("Parameter `observed` has to be of type `numpy.ndarray`.")

    try:
        (n_observed,) = np.shape(observed)
        n_var_observed = 1
    except:
        (n_observed, n_var_observed) = np.shape(observed)

    if n_var_observed != 1:
        raise ValueError("Parameter `observed` has to be a 0D or 1D vector.")

    if not isinstance(predicted, np.ndarray):
        raise ValueError("Parameter `predicted` has to be of type `numpy.ndarray`.")

    try:
        (n_predicted,) = np.shape(predicted)
        n_var_predicted = 1
    except:
        (n_predicted, n_var_predicted) = np.shape(predicted)

    if n_var_predicted != 1:
        raise ValueError("Parameter `predicted` has to be a 0D or 1D vector.")

    if n_observed != n_predicted:
        raise ValueError("Parameter `observed` has different number of elements than `predicted`.")

    __observed = observed.ravel()
    __predicted = predicted.ravel()

    rmse = (mean_squared_error(__observed, __predicted))**0.5

    return rmse

# ------------------------------------------------------------------------------

def stratified_root_mean_squared_error(observed, predicted, idx, verbose=False):
    """
    Computes the stratified root mean squared error (RMSE) values. Stratified RMSE is computed separately in each
    bin (cluster) of an observed dependent variable, :math:`\\phi_o`.

    RMSE in the :math:`j^{th}` bin can be computed as:

    .. math::

        \\mathrm{RMSE}_j = \\sqrt{\\frac{1}{N_j} \\sum_{i=1}^{N_j} (\\phi_{o,i}^j - \\phi_{p,i}^j) ^2}

    where :math:`N_j` is the number of observations in the :math:`j^{th}` bin, :math:`\\phi_o` is the observed and
    :math:`\\phi_p` is the predicted dependent variable.

    **Example:**

    .. code:: python

        from PCAfold import PCA, variable_bins, stratified_root_mean_squared_error
        import numpy as np

        # Generate dummy data set:
        X = np.random.rand(100,10)

        # Instantiate PCA class object:
        pca_X = PCA(X, scaling='auto', n_components=2)

        # Approximate the data set:
        X_rec = pca_X.reconstruct(pca_X.transform(X))

        # Generate bins:
        (idx, bins_borders) = variable_bins(X[:,0], k=10, verbose=False)

        # Compute stratified RMSE in 10 bins of the first variable in a data set:
        rmse_in_bins = stratified_root_mean_squared_error(X[:,0], X_rec[:,0], idx=idx, verbose=True)

    :param observed:
        ``numpy.ndarray`` specifying the observed values of a single dependent variable, :math:`\\phi_o`. It should be of size ``(n_observations,)`` or ``(n_observations, 1)``.
    :param predicted:
        ``numpy.ndarray`` specifying the predicted values of a single dependent variable, :math:`\\phi_p`. It should be of size ``(n_observations,)`` or ``(n_observations, 1)``.
    :param idx:
        ``numpy.ndarray`` of cluster classifications. It should be of size ``(n_observations,)`` or ``(n_observations,1)``.
    :param verbose: (optional)
        ``bool`` for printing sizes (number of observations) and RMSE values in each bin.

    :return:
        - **rmse_in_bins** - ``list`` specifying the mean squared error (RMSE) in each bin. It has length ``k``.
    """

    if not isinstance(observed, np.ndarray):
        raise ValueError("Parameter `observed` has to be of type `numpy.ndarray`.")

    try:
        (n_observed,) = np.shape(observed)
        n_var_observed = 1
    except:
        (n_observed, n_var_observed) = np.shape(observed)

    if n_var_observed != 1:
        raise ValueError("Parameter `observed` has to be a 0D or 1D vector.")

    if not isinstance(predicted, np.ndarray):
        raise ValueError("Parameter `predicted` has to be of type `numpy.ndarray`.")

    try:
        (n_predicted,) = np.shape(predicted)
        n_var_predicted = 1
    except:
        (n_predicted, n_var_predicted) = np.shape(predicted)

    if n_var_predicted != 1:
        raise ValueError("Parameter `predicted` has to be a 0D or 1D vector.")

    if n_observed != n_predicted:
        raise ValueError("Parameter `observed` has different number of elements than `predicted`.")

    if isinstance(idx, np.ndarray):
        if not all(isinstance(i, np.integer) for i in idx.ravel()):
            raise ValueError("Parameter `idx` can only contain integers.")
    else:
        raise ValueError("Parameter `idx` has to be of type `numpy.ndarray`.")

    try:
        (n_observations_idx, ) = np.shape(idx)
        n_idx = 1
    except:
        (n_observations_idx, n_idx) = np.shape(idx)

    if n_idx != 1:
        raise ValueError("Parameter `idx` has to have size `(n_observations,)` or `(n_observations,1)`.")

    if n_observations_idx != n_observed:
        raise ValueError('Vector of cluster classifications `idx` has different number of observations than the original data set `X`.')

    if not isinstance(verbose, bool):
        raise ValueError("Parameter `verbose` has to be a boolean.")

    __observed = observed.ravel()
    __predicted = predicted.ravel()

    rmse_in_bins = []

    for cl in np.unique(idx):

        (idx_bin,) = np.where(idx==cl)

        rmse = root_mean_squared_error(__observed[idx_bin], __predicted[idx_bin])

        constant_bin_metric_min = np.min(__observed[idx_bin])/np.mean(__observed[idx_bin])
        constant_bin_metric_max = np.max(__observed[idx_bin])/np.mean(__observed[idx_bin])

        if verbose:
            if (abs(constant_bin_metric_min - 1) < 0.01) and (abs(constant_bin_metric_max - 1) < 0.01):
                print('Bin\t' + str(cl+1) + '\t| size\t ' + str(len(idx_bin)) + '\t| RMSE\t' + str(round(rmse,6)) + '\t| ' + colored('This bin has almost constant values.', 'red'))
            else:
                print('Bin\t' + str(cl+1) + '\t| size\t ' + str(len(idx_bin)) + '\t| RMSE\t' + str(round(rmse,6)))

        rmse_in_bins.append(rmse)

    return rmse_in_bins

# ------------------------------------------------------------------------------

def normalized_root_mean_squared_error(observed, predicted, norm='std'):
    """
    Computes the normalized root mean squared error (NRMSE):

    .. math::

        \\mathrm{NRMSE} = \\frac{1}{d_{norm}} \\sqrt{\\frac{1}{N} \\sum_{i=1}^N (\\phi_{o,i} - \\phi_{p,i}) ^2}

    where :math:`d_{norm}` is the normalization factor, :math:`N` is the number of observations, :math:`\\phi_o` is the observed and
    :math:`\\phi_p` is the predicted dependent variable.

    Various normalizations are available:

    +----------------------------+--------------------------+------------------------------------------------------------------------------+
    | Normalization              | ``norm``                 | Normalization factor :math:`d_{norm}`                                        |
    +============================+==========================+==============================================================================+
    | Root square mean           | ``'root_square_mean'``   | :math:`d_{norm} = \sqrt{\mathrm{mean}(\phi_o^2)}`                            |
    +----------------------------+--------------------------+------------------------------------------------------------------------------+
    | Std                        | ``'std'``                | :math:`d_{norm} = \mathrm{std}(\phi_o)`                                      |
    +----------------------------+--------------------------+------------------------------------------------------------------------------+
    | Range                      | ``'range'``              | :math:`d_{norm} = \mathrm{max}(\phi_o) - \mathrm{min}(\phi_o)`               |
    +----------------------------+--------------------------+------------------------------------------------------------------------------+
    | Root square range          | ``'root_square_range'``  | :math:`d_{norm} = \sqrt{\mathrm{max}(\phi_o^2) - \mathrm{min}(\phi_o^2)}``   |
    +----------------------------+--------------------------+------------------------------------------------------------------------------+
    | Root square std            | ``'root_square_std'``    | :math:`d_{norm} = \sqrt{\mathrm{std}(\phi_o^2)}`                             |
    +----------------------------+--------------------------+------------------------------------------------------------------------------+
    | Absolute mean              | ``'abs_mean'``           | :math:`d_{norm} = | \mathrm{mean}(\phi_o) |`                                 |
    +----------------------------+--------------------------+------------------------------------------------------------------------------+

    **Example:**

    .. code:: python

        from PCAfold import PCA, normalized_root_mean_squared_error
        import numpy as np

        # Generate dummy data set:
        X = np.random.rand(100,3)

        # Instantiate PCA class object:
        pca_X = PCA(X, scaling='auto', n_components=2)

        # Approximate the data set:
        X_rec = pca_X.reconstruct(pca_X.transform(X))

        # Compute the root mean squared error for the first variable:
        nrmse = normalized_root_mean_squared_error(X[:,0], X_rec[:,0], norm='std')

    :param observed:
        ``numpy.ndarray`` specifying the observed values of a single dependent variable, :math:`\\phi_o`. It should be of size ``(n_observations,)`` or ``(n_observations, 1)``.
    :param predicted:
        ``numpy.ndarray`` specifying the predicted values of a single dependent variable, :math:`\\phi_p`. It should be of size ``(n_observations,)`` or ``(n_observations, 1)``.
    :param norm:
        ``str`` specifying the normalization, :math:`d_{norm}`. It can be one of the following: ``std``, ``range``, ``root_square_mean``, ``root_square_range``, ``root_square_std``, ``abs_mean``.

    :return:
        - **nrmse** - normalized root mean squared error (NRMSE).
    """

    __norms = ['root_square_mean', 'std', 'range', 'root_square_range', 'root_square_std', 'abs_mean']

    if not isinstance(observed, np.ndarray):
        raise ValueError("Parameter `observed` has to be of type `numpy.ndarray`.")

    try:
        (n_observed,) = np.shape(observed)
        n_var_observed = 1
    except:
        (n_observed, n_var_observed) = np.shape(observed)

    if n_var_observed != 1:
        raise ValueError("Parameter `observed` has to be a 0D or 1D vector.")

    if not isinstance(predicted, np.ndarray):
        raise ValueError("Parameter `predicted` has to be of type `numpy.ndarray`.")

    try:
        (n_predicted,) = np.shape(predicted)
        n_var_predicted = 1
    except:
        (n_predicted, n_var_predicted) = np.shape(predicted)

    if n_var_predicted != 1:
        raise ValueError("Parameter `predicted` has to be a 0D or 1D vector.")

    if n_observed != n_predicted:
        raise ValueError("Parameter `observed` has different number of elements than `predicted`.")

    if norm not in __norms:
        raise ValueError("Parameter `norm` can be one of the following: ``std``, ``range``, ``root_square_mean``, ``root_square_range``, ``root_square_std``, ``abs_mean``.")

    __observed = observed.ravel()
    __predicted = predicted.ravel()

    rmse = root_mean_squared_error(__observed, __predicted)

    if norm == 'root_square_mean':
        nrmse = rmse/np.sqrt(np.mean(__observed**2))
    elif norm == 'std':
        nrmse = rmse/(np.std(__observed))
    elif norm == 'range':
        nrmse = rmse/(np.max(__observed) - np.min(__observed))
    elif norm == 'root_square_range':
        nrmse = rmse/np.sqrt(np.max(__observed**2) - np.min(__observed**2))
    elif norm == 'root_square_std':
        nrmse = rmse/np.sqrt(np.std(__observed**2))
    elif norm == 'abs_mean':
        nrmse = rmse/abs(np.mean(__observed))

    return nrmse

# ------------------------------------------------------------------------------

def stratified_normalized_root_mean_squared_error(observed, predicted, idx, norm='std', use_global_norm=False, verbose=False):
    """
    Computes the stratified normalized root mean squared error (NRMSE) values. Stratified NRMSE is computed separately in each
    bin (cluster) of an observed dependent variable, :math:`\\phi_o`.

    NRMSE in the :math:`j^{th}` bin can be computed as:

    .. math::

        \\mathrm{NRMSE}_j = \\frac{1}{d_{norm}} \\sqrt{\\frac{1}{N_j} \\sum_{i=1}^{N_j} (\\phi_{o,i}^j - \\phi_{p,i}^j) ^2}

    where :math:`N_j` is the number of observations in the :math:`j^{th}` bin, :math:`\\phi_o` is the observed and
    :math:`\\phi_p` is the predicted dependent variable.

    **Example:**

    .. code:: python

        from PCAfold import PCA, variable_bins, stratified_normalized_root_mean_squared_error
        import numpy as np

        # Generate dummy data set:
        X = np.random.rand(100,10)

        # Instantiate PCA class object:
        pca_X = PCA(X, scaling='auto', n_components=2)

        # Approximate the data set:
        X_rec = pca_X.reconstruct(pca_X.transform(X))

        # Generate bins:
        (idx, bins_borders) = variable_bins(X[:,0], k=10, verbose=False)

        # Compute stratified NRMSE in 10 bins of the first variable in a data set:
        nrmse_in_bins = stratified_normalized_root_mean_squared_error(X[:,0],
                                                                      X_rec[:,0],
                                                                      idx=idx,
                                                                      norm='std',
                                                                      use_global_norm=True,
                                                                      verbose=True)

    :param observed:
        ``numpy.ndarray`` specifying the observed values of a single dependent variable, :math:`\\phi_o`. It should be of size ``(n_observations,)`` or ``(n_observations, 1)``.
    :param predicted:
        ``numpy.ndarray`` specifying the predicted values of a single dependent variable, :math:`\\phi_p`. It should be of size ``(n_observations,)`` or ``(n_observations, 1)``.
    :param idx:
        ``numpy.ndarray`` of cluster classifications. It should be of size ``(n_observations,)`` or ``(n_observations,1)``.
    :param norm:
        ``str`` specifying the normalization, :math:`d_{norm}`. It can be one of the following: ``std``, ``range``, ``root_square_mean``, ``root_square_range``, ``root_square_std``, ``abs_mean``.
    :param use_global_norm: (optional)
            ``bool`` specifying if global norm of the observed variable should be used in NRMSE calculation. If set to ``False``, norms are computed on samples from the the corresponding bin.
    :param verbose: (optional)
        ``bool`` for printing sizes (number of observations) and NRMSE values in each bin.

    :return:
        - **nrmse_in_bins** - ``list`` specifying the mean squared error (NRMSE) in each bin. It has length ``k``.
    """

    if not isinstance(observed, np.ndarray):
        raise ValueError("Parameter `observed` has to be of type `numpy.ndarray`.")

    try:
        (n_observed,) = np.shape(observed)
        n_var_observed = 1
    except:
        (n_observed, n_var_observed) = np.shape(observed)

    if n_var_observed != 1:
        raise ValueError("Parameter `observed` has to be a 0D or 1D vector.")

    if not isinstance(predicted, np.ndarray):
        raise ValueError("Parameter `predicted` has to be of type `numpy.ndarray`.")

    try:
        (n_predicted,) = np.shape(predicted)
        n_var_predicted = 1
    except:
        (n_predicted, n_var_predicted) = np.shape(predicted)

    if n_var_predicted != 1:
        raise ValueError("Parameter `predicted` has to be a 0D or 1D vector.")

    if n_observed != n_predicted:
        raise ValueError("Parameter `observed` has different number of elements than `predicted`.")

    if isinstance(idx, np.ndarray):
        if not all(isinstance(i, np.integer) for i in idx.ravel()):
            raise ValueError("Parameter `idx` can only contain integers.")
    else:
        raise ValueError("Parameter `idx` has to be of type `numpy.ndarray`.")

    try:
        (n_observations_idx, ) = np.shape(idx)
        n_idx = 1
    except:
        (n_observations_idx, n_idx) = np.shape(idx)

    if n_idx != 1:
        raise ValueError("Parameter `idx` has to have size `(n_observations,)` or `(n_observations,1)`.")

    if n_observations_idx != n_observed:
        raise ValueError('Vector of cluster classifications `idx` has different number of observations than the original data set `X`.')

    if not isinstance(verbose, bool):
        raise ValueError("Parameter `verbose` has to be a boolean.")

    __observed = observed.ravel()
    __predicted = predicted.ravel()

    nrmse_in_bins = []

    for cl in np.unique(idx):

        (idx_bin,) = np.where(idx==cl)

        if use_global_norm:

            rmse = root_mean_squared_error(__observed[idx_bin], __predicted[idx_bin])

            if norm == 'root_square_mean':
                nrmse = rmse/np.sqrt(np.mean(__observed**2))
            elif norm == 'std':
                nrmse = rmse/(np.std(__observed))
            elif norm == 'range':
                nrmse = rmse/(np.max(__observed) - np.min(__observed))
            elif norm == 'root_square_range':
                nrmse = rmse/np.sqrt(np.max(__observed**2) - np.min(__observed**2))
            elif norm == 'root_square_std':
                nrmse = rmse/np.sqrt(np.std(__observed**2))
            elif norm == 'abs_mean':
                nrmse = rmse/abs(np.mean(__observed))

        else:

            nrmse = normalized_root_mean_squared_error(__observed[idx_bin], __predicted[idx_bin], norm=norm)

        constant_bin_metric_min = np.min(__observed[idx_bin])/np.mean(__observed[idx_bin])
        constant_bin_metric_max = np.max(__observed[idx_bin])/np.mean(__observed[idx_bin])

        if verbose:
            if (abs(constant_bin_metric_min - 1) < 0.01) and (abs(constant_bin_metric_max - 1) < 0.01):
                print('Bin\t' + str(cl+1) + '\t| size\t ' + str(len(idx_bin)) + '\t| NRMSE\t' + str(round(nrmse,6)) + '\t| ' + colored('This bin has almost constant values.', 'red'))
            else:
                print('Bin\t' + str(cl+1) + '\t| size\t ' + str(len(idx_bin)) + '\t| NRMSE\t' + str(round(nrmse,6)))

        nrmse_in_bins.append(nrmse)

    return nrmse_in_bins

# ------------------------------------------------------------------------------

def turning_points(observed, predicted):
    """
    Computes the turning points percentage - the percentage of predicted outputs
    that have the opposite growth tendency to the corresponding observed growth tendency.

    .. warning::

        This function is under construction.

    :return:
        - **turning_points** - turning points percentage in %.
    """

    return turning_points

# ------------------------------------------------------------------------------

def good_estimate(observed, predicted, tolerance=0.05):
    """
    Computes the good estimate (GE) - the percentage of predicted values that
    are within the specified tolerance from the corresponding observed values.

    .. warning::

        This function is under construction.

    :param observed:
        ``numpy.ndarray`` specifying the observed values of a single dependent variable. It should be of size ``(n_observations,)`` or ``(n_observations, 1)``.
    :param predicted:
        ``numpy.ndarray`` specifying the predicted values of a single dependent variable. It should be of size ``(n_observations,)`` or ``(n_observations, 1)``.
    :parm tolerance:
        ``float`` specifying the tolerance.

    :return:
        - **good_estimate** - good estimate (GE) in %.
    """

    return good_estimate

# ------------------------------------------------------------------------------

def good_direction_estimate(observed, predicted, tolerance=0.05):
    """
    Computes the good direction (GD) and the good direction estimate (GDE).

    GD for observation :math:`i`, is computed as:

    .. math::

        GD_i = \\frac{\\vec{\\phi}_{o,i}}{|| \\vec{\\phi}_{o,i} ||} \\cdot \\frac{\\vec{\\phi}_{p,i}}{|| \\vec{\\phi}_{p,i} ||}

    where :math:`\\vec{\\phi}_o` is the observed vector quantity and :math:`\\vec{\\phi}_p` is the
    predicted vector quantity.

    GDE is computed as the percentage of predicted vector observations whose
    direction is within the specified tolerance from the direction of the
    corresponding observed vector.

    **Example:**

    .. code:: python

        from PCAfold import PCA, good_direction_estimate
        import numpy as np

        # Generate dummy data set:
        X = np.random.rand(100,3)

        # Instantiate PCA class object:
        pca_X = PCA(X, scaling='auto', n_components=2)

        # Approximate the data set:
        X_rec = pca_X.reconstruct(pca_X.transform(X))

        # Compute the vector of good direction and good direction estimate:
        (good_direction, good_direction_estimate) = good_direction_estimate(X, X_rec, tolerance=0.01)

    :param observed:
        ``numpy.ndarray`` specifying the observed vector quantity, :math:`\\vec{\\phi}_o`. It should be of size ``(n_observations,n_dimensions)``.
    :param predicted:
        ``numpy.ndarray`` specifying the predicted vector quantity, :math:`\\vec{\\phi}_p`. It should be of size ``(n_observations,n_dimensions)``.
    :param tolerance:
        ``float`` specifying the tolerance.

    :return:
        - **good_direction** - ``numpy.ndarray`` specifying the vector of good direction (GD). It has size ``(n_observations,)``.
        - **good_direction_estimate** - good direction estimate (GDE) in %.
    """

    if not isinstance(observed, np.ndarray):
        raise ValueError("Parameter `observed` has to be of type `numpy.ndarray`.")

    try:
        (n_observed, n_dimensions_1) = np.shape(observed)
    except:
        raise ValueError("Parameter `observed` should be a matrix.")

    if n_dimensions_1 < 2:
        raise ValueError("Parameter `observed` has to have at least two dimensions.")

    if not isinstance(predicted, np.ndarray):
        raise ValueError("Parameter `predicted` has to be of type `numpy.ndarray`.")

    try:
        (n_predicted, n_dimensions_2) = np.shape(predicted)
    except:
        raise ValueError("Parameter `predicted` should be a matrix.")

    if n_dimensions_2 < 2:
        raise ValueError("Parameter `predicted` has to have at least two dimensions.")

    if n_observed != n_predicted:
        raise ValueError("Parameter `observed` has different number of elements than `predicted`.")

    if n_dimensions_1 != n_dimensions_2:
        raise ValueError("Parameter `observed` has different number of dimensions than `predicted`.")

    if not isinstance(tolerance, float):
        raise ValueError("Parameter `tolerance` has to be of type `float`.")

    good_direction = np.zeros((n_observed,))

    for i in range(0,n_observed):
        good_direction[i] = np.dot(observed[i,:]/np.linalg.norm(observed[i,:]), predicted[i,:]/np.linalg.norm(predicted[i,:]))

    (idx_good_direction, ) = np.where(good_direction >= 1.0 - tolerance)

    good_direction_estimate = len(idx_good_direction)/n_observed * 100.0

    return (good_direction, good_direction_estimate)

# ------------------------------------------------------------------------------

def generate_tex_table(data_frame_table, float_format='.2f', caption='', label=''):
    """
    Generates ``tex`` code for a table stored in a ``pandas.DataFrame``. This function
    can be useful e.g. for printing regression results.

    **Example:**

    .. code:: python

        from PCAfold import PCA, generate_tex_table
        import numpy as np
        import pandas as pd

        # Generate dummy data set:
        X = np.random.rand(100,5)

        # Generate dummy variables names:
        variable_names = ['A1', 'A2', 'A3', 'A4', 'A5']

        # Instantiate PCA class object:
        pca_q2 = PCA(X, scaling='auto', n_components=2, use_eigendec=True, nocenter=False)
        pca_q3 = PCA(X, scaling='auto', n_components=3, use_eigendec=True, nocenter=False)

        # Calculate the R2 values:
        r2_q2 = pca_q2.calculate_r2(X)[None,:]
        r2_q3 = pca_q3.calculate_r2(X)[None,:]

        # Generate pandas.DataFrame from the R2 values:
        r2_table = pd.DataFrame(np.vstack((r2_q2, r2_q3)), columns=variable_names, index=['PCA, $q=2$', 'PCA, $q=3$'])

        # Generate tex code for the table:
        generate_tex_table(r2_table, float_format=".3f", caption='$R^2$ values.', label='r2-values')

    .. note::

        The code above will produce ``tex`` code:

        .. code-block:: text

            \\begin{table}[h!]
            \\begin{center}
            \\begin{tabular}{llllll} \\toprule
             & \\textit{A1} & \\textit{A2} & \\textit{A3} & \\textit{A4} & \\textit{A5} \\\\ \\midrule
            PCA, $q=2$ & 0.507 & 0.461 & 0.485 & 0.437 & 0.611 \\\\
            PCA, $q=3$ & 0.618 & 0.658 & 0.916 & 0.439 & 0.778 \\\\
            \\end{tabular}
            \\caption{$R^2$ values.}\\label{r2-values}
            \\end{center}
            \\end{table}

        Which, when compiled, will result in a table:

        .. image:: ../images/generate-tex-table.png
            :width: 450
            :align: center

    :param data_frame_table:
        ``pandas.DataFrame`` specifying the table to convert to ``tex`` code. It can include column names and
        index names.
    :param float_format:
        ``str`` specifying the display format for the numerical entries inside the
        table. By default it is set to ``'.2f'``.
    :param caption:
        ``str`` specifying caption for the table.
    :param label:
        ``str`` specifying label for the table.
    """

    (n_rows, n_columns) = np.shape(data_frame_table)
    rows_labels = data_frame_table.index.values
    columns_labels = data_frame_table.columns.values

    print('')
    print(r'\begin{table}[h!]')
    print(r'\begin{center}')
    print(r'\begin{tabular}{' + ''.join(['l' for i in range(0, n_columns+1)]) + r'} \toprule')
    print(' & ' + ' & '.join([r'\textit{' + name + '}' for name in columns_labels]) + r' \\ \midrule')

    for row_i, row_label in enumerate(rows_labels):

        row_values = list(data_frame_table.iloc[row_i,:])
        print(row_label + r' & '+  ' & '.join([str(('%' + float_format) % value) for value in row_values]) + r' \\')

    print(r'\end{tabular}')
    print(r'\caption{' + caption + r'}\label{' + label + '}')
    print(r'\end{center}')
    print(r'\end{table}')
    print('')

# ------------------------------------------------------------------------------

################################################################################
#
# Artificial neural network (ANN) regression
#
################################################################################

class ANN:
    """
    Enables reconstruction of quantities of interest (QoIs) using artificial neural network (ANN).

    **Example:**

    .. code:: python

        from PCAfold import ANN
        import numpy as np

        # Generate dummy dataset:
        input_data = np.random.rand(100,8)
        output_data = np.random.rand(100,3)

        # Instantiate ANN class object:
        ann_model = ANN(input_data,
                        output_data,
                        interior_architecture=(5,4),
                        activation_functions=('tanh', 'tanh', 'linear'),
                        weights_init='glorot_uniform',
                        biases_init='zeros',
                        loss='MSE',
                        optimizer='Adam',
                        batch_size=100,
                        n_epochs=1000,
                        learning_rate=0.001,
                        validation_perc=10,
                        random_seed=100,
                        verbose=True)

        # Begin model training:
        ann_model.train()

    A summary of the current ANN model and its hyperparameter settings can be printed using the ``summary()`` function:

    .. code:: python

        # Print the ANN model summary
        qoi_aware.summary()

    .. code-block:: text

        ANN model summary...

    :param input_data:
        ``numpy.ndarray`` specifying the data set used as the input (regressors) to the ANN. It should be of size ``(n_observations,n_input_variables)``.
    :param output_data:
        ``numpy.ndarray`` specifying the data set used as the output (predictors) to the ANN. It should be of size ``(n_observations,n_output_variables)``.
    :param interior_architecture: (optional)
        ``tuple`` of ``int`` specifying the number of neurons in the interior network architecture.
        For example, if ``interior_architecture=(4,5)``, two interior layers will be created and the overal network architecture will be ``(Input)-(4)-(5)-(Output)``.
        If set to an empty tuple, ``interior_architecture=()``, the overal network architecture will be ``(Input)-(Output)``.
        Keep in mind that if you'd like to create just one interior layer, you should use a comma after the integer: ``interior_architecture=(4,)``.
    :param activation_functions: (optional)
        ``str`` or ``tuple`` specifying activation functions in all layers. If set to ``str``, the same activation function is used in all layers.
        If set to a ``tuple`` of ``str``, a different activation function can be set at different layers. The number of elements in the ``tuple`` should match the number of layers!
        ``str`` and ``str`` elements of the ``tuple`` can only be ``'linear'``, ``'sigmoid'``, or ``'tanh'``.
    :param weights_init: (optional)
        ``str`` specifying the initialization of weights in the network. If set to ``None``, weights will be initialized using the Glorot uniform distribution.
    :param biases_init: (optional)
        ``str`` specifying the initialization of biases in the network. If set to ``None``, biases will be initialized as zeros.
    :param loss: (optional)
        ``str`` specifying the loss function. It can be ``'MAE'`` or ``'MSE'``.
    :param optimizer: (optional)
        ``str`` specifying the optimizer used during training. It can be ``'Adam'`` or ``'Nadam'``.
    :param batch_size: (optional)
        ``int`` specifying the batch size.
    :param n_epochs: (optional)
        ``int`` specifying the number of epochs.
    :param learning_rate: (optional)
        ``float`` specifying the learning rate passed to the optimizer.
    :param validation_perc: (optional)
        ``int`` specifying the percentage of the input data to be used as validation data during training. It should be a number larger than or equal to 0 and smaller than 100. Note, that if it is set above 0, not all of the input data will be used as training data. Note, that validation data does not impact model training!
    :param random_seed: (optional)
        ``int`` specifying the random seed to be used for any random operations. It is highly recommended to set a fixed random seed, as this allows for complete reproducibility of the results.
    :param verbose: (optional)
        ``bool`` for printing verbose details.

    **Attributes:**

    - **input_data** - (read only) ``numpy.ndarray`` specifying the data set used as the input to the ANN.
    - **output_data** - (read only) ``numpy.ndarray`` specifying the data set used as the output to the ANN.
    - **architecture** - (read only) ``str`` specifying the ANN architecture.
    - **ann_model** - (read only) object of ``Keras.models.Sequential`` class that stores the artificial neural network model.
    - **weights_and_biases_init** - (read only) ``list`` of ``numpy.ndarray`` specifying weights and biases with which the ANN was intialized.
    - **weights_and_biases_trained** - (read only) ``list`` of ``numpy.ndarray`` specifying weights and biases after training the ANN. Only available after calling ``ANN.train()``.
    - **training_loss** - (read only) ``list`` of losses computed on the training data. Only available after calling ``ANN.train()``.
    - **validation_loss** - (read only) ``list`` of losses computed on the validation data. Only available after calling ``ANN.train()`` and only when ``validation_perc`` is not equal to 0.
    """

    def __init__(self,
                input_data,
                output_data,
                interior_architecture=(),
                activation_functions='tanh',
                weights_init='glorot_uniform',
                biases_init='zeros',
                loss='MSE',
                optimizer='Adam',
                batch_size=200,
                n_epochs=1000,
                learning_rate=0.001,
                validation_perc=10,
                random_seed=None,
                verbose=False):

        import tensorflow as tf
        from tensorflow.keras import layers, models

        __weights_inits = ['glorot_uniform']
        __biases_inits = ['zeros']
        __activations = ['linear', 'sigmoid', 'tanh']
        __optimizers = ['Adam', 'Nadam']
        __losses = ['MSE', 'MAE']

        if not isinstance(input_data, np.ndarray):
            raise ValueError("Parameter `input_data` has to be of type `numpy.ndarray`.")

        (n_input_observations, n_input_variables) = np.shape(input_data)

        if not isinstance(output_data, np.ndarray):
            raise ValueError("Parameter `output_data` has to be of type `numpy.ndarray`.")

        (n_output_observations, n_output_variables) = np.shape(output_data)

        if n_input_observations != n_output_observations:
            raise ValueError("Parameters `input_data` and `output_data` have to have the same number of observations.")

        if not isinstance(activation_functions, str) and not isinstance(activation_functions, tuple):
            raise ValueError("Parameter `activation_functions` has to be of type `str` or `tuple`.")

        if isinstance(activation_functions, str):
            if activation_functions not in __activations:
                raise ValueError("Parameter `activation_functions` can only be 'linear' 'sigmoid' or 'tanh'.")

        if not isinstance(interior_architecture, tuple):
            raise ValueError("Parameter `interior_architecture` has to be of type `tuple`.")

        if isinstance(activation_functions, tuple):
            for i in activation_functions:
                if not isinstance(i, str):
                    raise ValueError("Parameter `activation_functions` has to be a tuple of `str`.")
                if i not in __activations:
                    raise ValueError("Elements of the parameter `activation_functions` can only be 'linear' 'sigmoid' or 'tanh'.")
            if len(activation_functions) != len(interior_architecture) + 1:
                raise ValueError("Parameter `activation_functions` has to have as many elements as there are layers in the neural network.")

        # Evaluate the architecture string:
        if len(interior_architecture)==0:
            architecture = str(n_input_variables) + '-' + str(n_output_variables)
            neuron_count = [n_input_variables, n_output_variables]
        else:
            architecture = str(n_input_variables) + '-' + '-'.join([str(i) for i in interior_architecture]) + '-' + str(n_output_variables)
            neuron_count = [n_input_variables] + [i for i in interior_architecture] + [n_output_variables]

        self.__neuron_count = neuron_count

        # Set the loss:
        if not isinstance(loss, str):
            raise ValueError("Parameter `loss` has to be of type `str`.")

        if loss not in __losses:
            raise ValueError("Parameter `loss` has to be 'MAE' or 'MSE'.")

        if loss == 'MSE':
            model_loss = tf.keras.losses.MeanSquaredError()
        elif loss == 'MAE':
            model_loss = tf.keras.losses.MeanAbsoluteError()

        # Set the optimizer:
        if not isinstance(optimizer, str):
            raise ValueError("Parameter `optimizer` has to be of type `str`.")

        if optimizer not in __optimizers:
            raise ValueError("Parameter `optimizer` has to be 'Adam' or 'Nadam'.")

        if optimizer == 'Adam':
            model_optimizer = tf.optimizers.Adam(learning_rate)
        elif optimizer == 'Nadam':
            model_optimizer = tf.optimizers.Nadam(learning_rate)

        if not isinstance(batch_size, int):
            raise ValueError("Parameter `batch_size` has to be of type `int`.")

        if not isinstance(n_epochs, int):
            raise ValueError("Parameter `n_epochs` has to be of type `int`.")

        if not isinstance(learning_rate, float):
            raise ValueError("Parameter `learning_rate` has to be of type `float`.")

        if not isinstance(validation_perc, int):
            raise ValueError("Parameter `validation_perc` has to be of type `int`.")

        if (validation_perc < 0) or (validation_perc >= 100):
            raise ValueError("Parameter `validation_perc` has to be an integer between 0 and 100`.")

        # Set random seed for neural network training reproducibility:
        if random_seed is not None:
            if not isinstance(random_seed, int):
                raise ValueError("Parameter `random_seed` has to be of type `int`.")
            tf.random.set_seed(random_seed)

        if not isinstance(verbose, bool):
            raise ValueError("Parameter `verbose` has to be a boolean.")

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        # Create an artificial neural network with a given architecture:
        ann_model = models.Sequential()

        if isinstance(activation_functions, str):
            ann_model.add(layers.Dense(self.__neuron_count[1], input_dim=self.__neuron_count[0], activation=activation_functions, kernel_initializer=weights_init, bias_initializer=biases_init))
        elif isinstance(activation_functions, tuple):
            ann_model.add(layers.Dense(self.__neuron_count[1], input_dim=self.__neuron_count[0], activation=activation_functions[0], kernel_initializer=weights_init, bias_initializer=biases_init))

        for i, n_neurons in enumerate(self.__neuron_count[2::]):
            if isinstance(activation_functions, str):
                    ann_model.add(layers.Dense(n_neurons, activation=activation_functions, kernel_initializer=weights_init, bias_initializer=biases_init))
            elif isinstance(activation_functions, tuple):
                ann_model.add(layers.Dense(n_neurons, activation=activation_functions[i+1], kernel_initializer=weights_init, bias_initializer=biases_init))

        # Compile the neural network model:
        ann_model.compile(model_optimizer, loss=model_loss)

        # Attributes coming from user inputs:
        self.__input_data = input_data
        self.__output_data = output_data
        self.__activation_functions = activation_functions
        self.__interior_architecture = interior_architecture
        self.__weights_init = weights_init
        self.__biases_init = biases_init
        self.__loss = loss
        self.__loss_function = model_loss
        self.__optimizer = optimizer
        self.__model_optimizer = model_optimizer
        self.__batch_size = batch_size
        self.__n_epochs = n_epochs
        self.__learning_rate = learning_rate
        self.__validation_perc = validation_perc
        self.__random_seed = random_seed
        self.__verbose = verbose

        # Attributes computed at class object initialization:
        self.__architecture = architecture
        self.__ann_model = ann_model
        self.__weights_and_biases_init = ann_model.get_weights()
        self.__epochs_list = [e for e in range(0, n_epochs)]
        self.__trained = False

        # Attributes available after model training:
        self.__training_loss = None
        self.__validation_loss = None
        self.__idx_min_training_loss = None
        self.__idx_min_validation_loss = None
        self.__bases_across_epochs = None
        self.__weights_and_biases_trained = None

    @property
    def input_data(self):
        return self.__input_data

    @property
    def output_data(self):
        return self.__output_data

    @property
    def architecture(self):
        return self.__architecture

    @property
    def ann_model(self):
        return self.__ann_model

    @property
    def weights_and_biases_init(self):
        return self.__weights_and_biases_init

    @property
    def weights_and_biases_trained(self):
        return self.__weights_and_biases_trained

    @property
    def training_loss(self):
        return self.__training_loss

    @property
    def validation_loss(self):
        return self.__validation_loss

# ------------------------------------------------------------------------------

    def summary(self):
        """
        Prints the ANN model summary.
        """

        print('ANN model summary...\n')
        if self.__trained:
            print('(Model has been trained)\n\n')
        else:
            print('(Model has not been trained yet)\n\n')

        print('- '*60)

        print('ANN architecture:\n')
        print('\t' + self.architecture)
        print('\n' + '- '*60)

        print('Activation functions:\n')
        activation_function_string = '(' + str(self.__neuron_count[0]) + ')'
        for i, n_neurons in enumerate(self.__neuron_count[1::]):
            if isinstance(self.__activation_functions, str):
                activation_function_string = activation_function_string + '--' + self.__activation_functions + '--'
            elif isinstance(self.__activation_functions, tuple):
                activation_function_string = activation_function_string + '--' + self.__activation_functions[i] + '--'
            activation_function_string = activation_function_string + '(' + str(n_neurons) + ')'
        print('\t' + activation_function_string)
        print('\n' + '- '*60)

        print('Model validation:\n')
        if self.__validation_perc != 0:
            print('\t- ' + 'Using ' + str(self.__validation_perc) + '% of input data as validation data')
        else:
            print('\t- ' + 'No validation data is used at model training')

        print('\t- ' + 'Model will be trained on ' + str(100 - self.__validation_perc) + '% of input data')

        print('\n' + '- '*60)

        print('Hyperparameters:\n')
        print('\t- ' + 'Batch size:\t\t' + str(self.__batch_size))
        print('\t- ' + '# of epochs:\t\t' + str(self.__n_epochs))
        print('\t- ' + 'Optimizer:\t\t' + self.__optimizer)
        print('\t- ' + 'Learning rate:\t' + str(self.__learning_rate))
        print('\t- ' + 'Loss function:\t' + self.__loss)
        print('\n' + '- '*60)

        print('Weights initialization:\n')
        if self.__weights_init is None:
            print('\t- ' + 'Glorot uniform')
        else:
            print('\t- ' + 'User-provided custom initialization of weights')
        print('\n' + '- '*60)

        print('Biases initialization:\n')
        if self.__biases_init is None:
            print('\t- ' + 'Zeros')
        else:
            print('\t- ' + 'User-provided custom initialization of biases')
        print('\n' + '- '*60)

        print('Results reproducibility:\n')
        if self.__random_seed is not None:
            print('\t- ' + 'Reproducible neural network training will be assured using random seed: ' + str(self.__random_seed))
        else:
            print('\t- ' + 'Random seed not set, neural network training results will not be reproducible!')
        print('\n' + '= '*60)

        if self.__trained:

            idx_min_training_loss, = np.where(self.__training_loss==np.min(self.__training_loss))
            idx_min_training_loss = idx_min_training_loss[0]

            print('Training results:\n')
            print('\t- ' + 'Minimum training loss:\t\t' + str(np.min(self.__training_loss)))
            print('\t- ' + 'Minimum training loss at epoch:\t' + str(idx_min_training_loss+1))
            if self.__validation_perc != 0:
                idx_min_validation_loss, = np.where(self.__validation_loss==np.min(self.__validation_loss))
                idx_min_validation_loss = idx_min_validation_loss[0]
                print('\n\t- ' + 'Minimum validation loss:\t\t' + str(np.min(self.__validation_loss)))
                print('\t- ' + 'Minimum validation loss at epoch:\t' + str(idx_min_validation_loss+1))

            print('\n' + '- '*60)

# ------------------------------------------------------------------------------

    def train(self):
        """
        Trains the artificial neural network (ANN) model.
        """

        import tensorflow as tf
        from tensorflow.keras import layers, models

        if self.__verbose: print('Starting model training...\n\n')

        if self.__random_seed is not None:
            tf.random.set_seed(self.__random_seed)

        training_losses_across_epochs = []
        validation_losses_across_epochs = []

        (n_observations, _) = np.shape(self.__input_data)

        tic = time.perf_counter()

        n_count_epochs = 0

        if self.__validation_perc != 0:
            sample_random = preprocess.DataSampler(np.zeros((n_observations,)).astype(int), random_seed=self.__random_seed, verbose=False)
            (idx_train, idx_validation) = sample_random.random(100 - self.__validation_perc)
            validation_data = (self.__input_data[idx_validation,:], self.__output_data[idx_validation,:])
            if self.__verbose: print('Using ' + str(self.__validation_perc) + '% of input data as validation data. Model will be trained on ' + str(100 - self.__validation_perc) + '% of input data.\n')
        else:
            sample_random = preprocess.DataSampler(np.zeros((n_observations,)).astype(int), random_seed=self.__random_seed, verbose=False)
            (idx_train, _) = sample_random.random(100)
            validation_data = None
            if self.__verbose: print('No validation data is used at model training. Model will be trained on 100% of input data.\n')

        for i_epoch in tqdm(self.__epochs_list):

            history = self.__ann_model.fit(self.__input_data[idx_train,:],
                                            self.__output_data[idx_train,:],
                                            epochs=1,
                                            batch_size=self.__batch_size,
                                            shuffle=True,
                                            validation_data=validation_data,
                                            verbose=0)

            training_losses_across_epochs.append(history.history['loss'][0])
            if self.__validation_perc != 0: validation_losses_across_epochs.append(history.history['val_loss'][0])

        toc = time.perf_counter()
        if self.__verbose: print(f'Time it took: {(toc - tic)/60:0.1f} minutes.\n')

        self.__training_loss = training_losses_across_epochs
        self.__validation_loss = validation_losses_across_epochs
        self.__weights_and_biases_trained = self.__ann_model.get_weights()
        self.__trained = True

        idx_min_training_loss, = np.where(self.__training_loss==np.min(self.__training_loss))
        self.__idx_min_training_loss = idx_min_training_loss[0]

        if self.__validation_perc != 0:
            idx_min_validation_loss, = np.where(self.__validation_loss==np.min(self.__validation_loss))
            self.__idx_min_validation_loss = idx_min_validation_loss[0]

# ------------------------------------------------------------------------------

    def predict(self, input_regressors):

        pass

# ------------------------------------------------------------------------------

    def print_weights_and_biases_init(self):
        """
        Prints initial weights and biases from all layers of the QoI-aware encoder-decoder.
        """

        for i in range(0,len(self.weights_and_biases_init)):
            if i%2==0: print('Layers ' + str(int(i/2) + 1) + ' -- ' + str(int(i/2) + 2) + ': ' + '- '*20)
            if i%2==0:
                print('\nWeight:')
            else:
                print('Bias:')
            print(self.weights_and_biases_init[i])
            print()

# ------------------------------------------------------------------------------

    def print_weights_and_biases_trained(self):
        """
        Prints trained weights and biases from all layers of the QoI-aware encoder-decoder.
        """

        if self.__trained:

            for i in range(0,len(self.weights_and_biases_trained)):
                if i%2==0: print('Layers ' + str(int(i/2) + 1) + ' -- ' + str(int(i/2) + 2) + ': ' + '- '*20)
                if i%2==0:
                    print('\nWeight:')
                else:
                    print('Bias:')
                print(self.weights_and_biases_trained[i])
                print()

        else:

            print('Model has not been trained yet!')

# ------------------------------------------------------------------------------

    def plot_losses(self, markevery=100, figure_size=(15,5), save_filename=None):
        """
        Plots training and validation losses.

        :param figure_size: (optional)
            ``tuple`` specifying figure size.
        :param save_filename: (optional)
            ``str`` specifying plot save location/filename. If set to ``None``
            plot will not be saved. You can also set a desired file extension,
            for instance ``.pdf``. If the file extension is not specified, the default
            is ``.png``.

        :return:
            - **plt** - ``matplotlib.pyplot`` plot handle.
        """

        if not isinstance(markevery, int):
            raise ValueError("Parameter `markevery` has to be of type `int`.")

        if not isinstance(figure_size, tuple):
            raise ValueError("Parameter `figure_size` has to be of type `tuple`.")

        if save_filename is not None:
            if not isinstance(save_filename, str):
                raise ValueError("Parameter `save_filename` has to be of type `str`.")

        if self.__trained:

            x_axis = [i for i in range(1,self.__n_epochs+1)]
            x_ticks = [1] + [i for i in range(1,self.__n_epochs) if i%markevery==0] + [self.__n_epochs]

            plt.figure(figsize=figure_size)
            plt.semilogy(x_axis, self.__training_loss, 'k', lw=3, label='Training loss')
            plt.scatter(self.__idx_min_training_loss+1, np.min(self.__training_loss), c='k', s=200, label='Min training loss', zorder=10)

            if self.__validation_perc != 0:
                plt.semilogy(x_axis, self.__validation_loss, 'r--', lw=2, label='Validation loss')
                plt.scatter(self.__idx_min_validation_loss+1, np.min(self.__validation_loss), c='r', s=100, label='Min validation loss', zorder=20)

            plt.xlabel('Epoch #', fontsize=font_labels)
            plt.xticks(x_ticks, rotation=90)
            plt.ylabel(self.__loss + ' loss', fontsize=font_labels)
            plt.legend(frameon=False, ncol=1, fontsize=font_legend)
            plt.grid(alpha=grid_opacity, zorder=1)

            if save_filename is not None: plt.savefig(save_filename, dpi=save_dpi, bbox_inches='tight')

            return plt

        else:

            print('Model has not been trained yet!')

################################################################################
#
# Partition of Unity Networks (POUNets) regression
#
################################################################################

_ndim_write = 'ndim'
_npartition_write = 'npartition'
_nbasis_write = 'nbasis'
_floattype_write = 'float_type'
_tpower_write = 'transform_power'
_tshift_write = 'transform_shift'
_tsignshift_write = 'transform_sign_shift'
_ivarcenter_write = 'ivar_center'
_ivarscale_write = 'ivar_scale'
_pcenters_write = 'centers'
_pshapes_write = 'shapes'
_coeffs_write = 'coeffs'


def init_uniform_partitions(list_npartitions, ivars, width_factor=0.5, verbose=False):
    """
    Computes parameters for initializing partition locations near training data with uniform spacing in each dimension.

    **Example:**

    .. code:: python

        from PCAfold import init_uniform_partitions
        import numpy as np

        # Generate dummy data set:
        ivars = np.random.rand(100,2)

        # compute partition parameters for an initial 5x7 grid:
        init_data = init_uniform_partitions([5, 7], ivars)

    :param list_npartitions:
        list of integers specifying the number of partitions to try initializing in each dimension. Only partitions near the provided ivars are kept.
    :param ivars:
        array of independent variables used for determining which partitions to keep
    :param width_factor:
        (optional, default 0.5) the factor multiplying the spacing between partitions for initializing the partitions' RBF widths
    :param verbose:
        (optional, default False) when True, prints the number of partitions retained compared to the initial grid

    :return:
        a dictionary of partition parameters to be used in initializing a ``PartitionOfUnityNetwork``
    """
    ndim = len(list_npartitions)
    if ndim != ivars.shape[1]:
        raise ValueError("length of partition list must match dimensionality of ivars.")
    ivars_cs, center, scale = center_scale(ivars, '0to1')

    init_parameters = {}
    init_parameters['ivar_center'] = np.array([center])
    init_parameters['ivar_scale'] = np.array([scale])

    boundaries = []
    widths = np.zeros(ndim)
    for i in range(ndim):
        full_boundaries = np.linspace(0,1,list_npartitions[i]+1)
        boundaries.append(full_boundaries[:-1])
        widths[i] = full_boundaries[1] - full_boundaries[0]
    boundaries = np.vstack([b.ravel() for b in np.meshgrid(*boundaries)]).T

    partition_centers = []
    for i in range(boundaries.shape[0]):
        subivar = ivars_cs.copy()
        for j in range(boundaries.shape[1]):
            window = np.argwhere((subivar[:, j]>=boundaries[i,j])&(subivar[:, j]<=boundaries[i,j]+widths[j]))[:, 0]
            subivar = subivar[window,:]
            if len(window) == 0:
                break
            else:
                if j == ndim-1:
                    partition_centers.append(boundaries[i,:]+0.5*widths)
    partition_centers = np.array(partition_centers)
    init_parameters['partition_centers'] = partition_centers
    if verbose:
        print('kept', partition_centers.shape[0], 'partitions out of',boundaries.shape[0])

    partition_shapes = np.zeros_like(partition_centers)
    for i in range(ndim):
        partition_shapes[:,i] = width_factor * widths[i]
    init_parameters['partition_shapes'] = 1./partition_shapes

    return init_parameters

class PartitionOfUnityNetwork:
    """
    A class for reconstruction (regression) of QoIs using POUnets.

    The POUnets are constructed with a single-layer network of normalized radial basis functions (RBFs) whose neurons each own and weight a polynomial basis.
    For independent variable inputs :math:`\\vec{x}` of dimensionality :math:`d`, the :math:`i^{\\text{th}}` partition or neuron is computed as

    .. math::

        \\Phi_i(\\vec{x};\\vec{h}_i,K_i) = \\phi^{{\\rm RBF}}_i(\\vec{x};\\vec{h}_i,K_i)/\\sum_j \\phi^{{\\rm RBF}}_j(\\vec{x};\\vec{h}_i,K_i)

    where

    .. math::

        \\phi_i^{{\\rm RBF}}(\\vec{x};\\vec{h}_i,K_i) = \\exp\\left(-(\\vec{x}-\\vec{h}_i)^\\mathsf{T}K_i(\\vec{x}-\\vec{h}_i)\\right) \\nonumber

    with vector :math:`\\vec{h}_i` and diagonal matrix :math:`K_i` defining the :math:`d` center and :math:`d` shape parameters, respectively, for training.

    The final output of a POUnet is then obtained through

    .. math::

        g(\\vec{x};\\vec{h},K,c) = \\sum_{i=1}^{p}\\left(\\Phi_i(\\vec{x};\\vec{h}_i,K_i)\\sum_{k=1}^{b}c_{i,k}m_k(\\vec{x})\\right)

    where the polynomial basis is represented as a sum of :math:`b` Taylor monomials,
    with the :math:`k^{\\text{th}}` monomial written as :math:`m_k(\\vec{x})`,
    that are multiplied by trainable basis coefficients :math:`c`.
    The number of basis monomials is determined by the ``basis_type`` for the polynomial.
    For example, in two-dimensional space, a quadratic polynomial basis contains :math:`b=6`
    monomial functions :math:`\\{1, x_1, x_2, x_1^2, x_2^2, x_1x_2\\}`.
    The combination of the partitions and polynomial basis functions creates localized polynomial fits for a QoI.

    More information can be found in :cite:`Armstrong2022`.

    The ``PartitionOfUnityNetwork`` class also provides a nonlinear transformation for the dependent variable(s) during training,
    which can be beneficial if the variable changes over orders of magnitude, for example.
    The equation for the transformation of variable :math:`f` is

    .. math::

        (|f + s_1|)^\\alpha \\text{sign}(f + s_1) + s_2 \\text{sign}(f + s_1)

    where :math:`\\alpha` is the ``transform_power``, :math:`s_1` is the ``transform_shift``, and :math:`s_2` is the ``transform_sign_shift``.

    **Example:**

    .. code:: python

        from PCAfold import init_uniform_partitions, PartitionOfUnityNetwork
        import numpy as np

        # Generate dummy data set:
        ivars = np.random.rand(100,2)
        dvars = 2.*ivars[:,0] + 3.*ivars[:,1]

        # Initialize the POUnet parameters
        net = PartitionOfUnityNetwork(**init_uniform_partitions([5,7], ivars), basis_type='linear')

        # Build the training graph with provided training data
        net.build_training_graph(ivars, dvars)

        # (optional) update the learning rate (default is 1.e-3)
        net.update_lr(1.e-4)

        # (optional) update the least-squares regularization (default is 1.e-10)
        net.update_l2reg(1.e-10)

        # Train the POUnet
        net.train(1000)

        # Evaluate the POUnet
        pred = net(ivars)

        # Evaluate the POUnet derivatives
        der = net.derivatives(ivars)

        # Save the POUnet to a file
        net.write_data_to_file('filename.pkl')

        # Load a POUnet from file
        net2 = PartitionOfUnityNetwork.load_from_file('filename.pkl')

        # Evaluate the loaded POUnet (without needing to call build_training_graph)
        pred2 = net2(ivars)

    :param partition_centers:
        array size (number of partitions) x (number of ivar inputs) for partition locations
    :param partition_shapes:
        array size (number of partitions) x (number of ivar inputs) for partition shapes influencing the RBF widths
    :param basis_type:
        string (``'constant'``, ``'linear'``, or ``'quadratic'``) for the degree of polynomial basis
    :param ivar_center:
        (optional, default ``None``) array for centering the ivar inputs before evaluating the POUnet, if ``None`` centers with zeros
    :param ivar_scale:
        (optional, default ``None``) array for scaling the ivar inputs before evaluating the POUnet, if ``None`` scales with ones
    :param basis_coeffs:
        (optional, default ``None``) if the array of polynomial basis coefficients is known, it may be provided here,
        otherwise it will be initialized with ``build_training_graph`` and trained with ``train``
    :param transform_power:
        (optional, default 1.) the power parameter used in the transformation equation during training
    :param transform_shift:
        (optional, default 0.) the shift parameter used in the transformation equation during training
    :param transform_sign_shift:
        (optional, default 0.) the signed shift parameter used in the transformation equation during training
    :param dtype:
        (optional, default ``'float64'``) string specifying either float type ``'float64'`` or ``'float32'``

    **Attributes:**

    - **partition_centers** - (read only) array of the current partition centers
    - **partition_shapes** - (read only) array of the current partition shape parameters
    - **basis_type** - (read only) string relaying the basis degree
    - **basis_coeffs** - (read only) array of the current basis coefficients
    - **ivar_center** - (read only) array of the centering parameters for the ivar inputs
    - **ivar_scale** - (read only) array of the scaling parameters for the ivar inputs
    - **dtype** - (read only) string relaying the data type (``'float64'`` or ``'float32'``)
    - **training_archive** - (read only) dictionary of the errors and POUnet states archived during training
    - **iterations** - (read only) array of the iterations archived during training
    """

    def __init__(self,
                 partition_centers,
                 partition_shapes,
                 basis_type,
                 ivar_center=None,
                 ivar_scale=None,
                 basis_coeffs=None,
                 transform_power=1.,
                 transform_shift=0.,
                 transform_sign_shift=0.,
                 dtype='float64'
                ):
        self._sess = tf.Session()
        if partition_centers.shape != partition_shapes.shape:
            raise ValueError("Size of partition centers and shapes must match")
        self._partition_centers = partition_centers.copy()
        self._partition_shapes = partition_shapes.copy()
        if basis_type not in ['constant','linear','quadratic']:
            raise ValueError("Supported basis_type includes constant, linear, or quadratic")
        self._basis_type = basis_type
        self._np = self._partition_centers.shape[0]
        self._nd = self._partition_centers.shape[1]

        if dtype != 'float64' and dtype != 'float32':
            raise ValueError("Only float32 and float64 dtype supported")
        self._dtype_str = dtype
        self._dtype = tf.float64 if self._dtype_str == 'float64' else tf.float32

        l2reg_f = 1.e-10 if self._dtype_str == 'float64' else 1.e-6
        self._l2reg = tf.Variable(l2reg_f, name='l2reg', dtype=self._dtype)
        self._lr = tf.Variable(1.e-3, name='lr', dtype=self._dtype)

        self._ivar_center = ivar_center if ivar_center is not None else np.zeros((1,partition_centers.shape[1]))
        self._inv_ivar_scale = 1./ivar_scale if ivar_scale is not None else np.ones((1,partition_centers.shape[1]))
        if len(self._ivar_center.shape) == 1:
            self._ivar_center = np.array([self._ivar_center])
        if len(self._inv_ivar_scale.shape) == 1:
            self._inv_ivar_scale = np.array([self._inv_ivar_scale])
        if self._ivar_center.shape[1] != partition_centers.shape[1]:
            raise ValueError("ivar_center dimensionality", self._ivar_center.shape[1],"does not match partition parameters",partition_centers.shape[1])
        if self._inv_ivar_scale.shape[1] != partition_centers.shape[1]:
            raise ValueError("ivar_scale dimensionality", self._inv_ivar_scale.shape[1],"does not match partition parameters",partition_centers.shape[1])

        self._transform_power = transform_power
        self._transform_shift = transform_shift
        self._transform_sign_shift = transform_sign_shift

        if basis_coeffs is not None:
            self._basis_coeffs = basis_coeffs.copy()
            self._isready = True
            self._t_basis_coeffs = tf.Variable(self._basis_coeffs, name='basis_coeffs', dtype=self._dtype)
        else:
            self._basis_coeffs = None
            self._isready = False

        self._t_ivar_center = tf.constant(np.expand_dims(self._ivar_center, axis=2), name='centers', dtype=self._dtype)
        self._t_inv_ivar_scale = tf.constant(np.expand_dims(self._inv_ivar_scale, axis=2), name='scales', dtype=self._dtype)
        self._t_xp = tf.Variable(self._partition_centers, name='partition_centers', dtype=self._dtype)
        self._t_sp = tf.Variable(self._partition_shapes, name='partition_scales', dtype=self._dtype)
        self._sess.run(tf.global_variables_initializer())
        self._built_graph = False

    @classmethod
    def load_data_from_file(cls, filename):
        """
        Load data from a specified ``filename`` with pickle (following ``write_data_to_file``)

        :param filename:
            string

        :return:
            dictionary of the POUnet data
        """
        with open(filename, 'rb') as file_input:
            pickled_data = pickle.load(file_input)
        return pickled_data

    @classmethod
    def load_from_file(cls, filename):
        """Load class from a specified ``filename`` with pickle (following ``write_data_to_file``)

        :param filename:
            string

        :return:
            ``PartitionOfUnityNetwork``
        """
        return cls(**cls.load_data_from_file(filename))

    @classmethod
    def load_data_from_txt(cls, filename, verbose=False):
        """
        Load data from a specified txt ``filename`` (following ``write_data_to_txt``)

        :param filename:
            string
        :param verbose:
            (optional, default False) print out the data as it is read

        :return:
            dictionary of the POUnet data
        """
        out_data = {}
        with open(filename) as file:
            content = file.readlines()

            if content[1].strip('\n')!=_ndim_write:
                raise ValueError('inconsistent read',content[1].strip('\n'),'and expected',_ndim_write)
            ndim = int(content[2])
            if verbose:
                print(content[1].strip('\n'), ndim)

            if content[3].strip('\n')!=_npartition_write:
                raise ValueError('inconsistent read',content[3].strip('\n'),'and expected',_npartition_write)
            npart = int(content[4])
            if verbose:
                print(content[3].strip('\n'), npart)

            if content[5].strip('\n')!=_nbasis_write:
                raise ValueError('inconsistent read',content[5].strip('\n'),'and expected',_nbasis_write)
            nbasis = int(content[6])
            if verbose:
                print(content[5].strip('\n'), nbasis)
            if nbasis==0:
                out_data['basis_type'] = 'constant'
            elif nbasis==1:
                out_data['basis_type'] = 'linear'
            elif nbasis==2:
                out_data['basis_type'] = 'quadratic'

            if content[7].strip('\n')!=_floattype_write:
                raise ValueError('inconsistent read',content[7].strip('\n'),'and expected',_floattype_write)
            float_str = content[8].strip('\n')
            out_data['dtype'] = float_str
            if verbose:
                print(content[7].strip('\n'), float_str)

            if content[9].strip('\n')!=_tpower_write:
                raise ValueError('inconsistent read',content[9].strip('\n'),'and expected',_tpower_write)
            transform_power = float(content[10])
            out_data['transform_power'] = transform_power
            if verbose:
                print(content[9].strip('\n'), transform_power)

            if content[11].strip('\n')!=_tshift_write:
                raise ValueError('inconsistent read',content[11].strip('\n'),'and expected',_tshift_write)
            transform_shift = float(content[12])
            out_data['transform_shift'] = transform_shift
            if verbose:
                print(content[11].strip('\n'), transform_shift)

            if content[13].strip('\n')!=_tsignshift_write:
                raise ValueError('inconsistent read',content[13].strip('\n'),'and expected',_tsignshift_write)
            transform_sign_shift = float(content[14])
            out_data['transform_sign_shift'] = transform_sign_shift
            if verbose:
                print(content[13].strip('\n'), transform_sign_shift)

            if content[15].strip('\n')!=_ivarcenter_write:
                raise ValueError('inconsistent read',content[15].strip('\n'),'and expected',_ivarcenter_write)
            if verbose:
                print(content[15].strip('\n'))
            istart = 16
            ivar_centers = np.zeros((1,ndim))
            for i in range(ndim):
                ivar_centers[0,i] = float(content[istart])
                istart += 1
            if verbose:
                print(ivar_centers)
            out_data['ivar_center'] = ivar_centers

            if content[istart].strip('\n')!=_ivarscale_write:
                raise ValueError('inconsistent read',content[istart].strip('\n'),'and expected',_ivarscale_write)
            ivar_scale = np.zeros((1,ndim))
            if verbose:
                print(content[istart].strip('\n'))
            istart += 1
            for i in range(ndim):
                ivar_scale[0,i] = float(content[istart])
                istart += 1
            if verbose:
                print(ivar_scale)
            out_data['ivar_scale'] = ivar_scale


            if content[istart].strip('\n')!=_pcenters_write:
                raise ValueError('inconsistent read',content[istart].strip('\n'),'and expected',_pcenters_write)
            ncoef = int(ndim*npart)
            centers = np.zeros(ncoef)
            if verbose:
                print(content[istart].strip('\n'))
            istart += 1
            for i in range(ncoef):
                centers[i] = float(content[istart])
                istart += 1
            centers = centers.reshape(npart,ndim)
            if verbose:
                print(centers)
            out_data['partition_centers'] = centers

            if content[istart].strip('\n')!=_pshapes_write:
                raise ValueError('inconsistent read',content[istart].strip('\n'),'and expected',_pshapes_write)
            shapes = np.zeros(ncoef)
            if verbose:
                print(content[istart].strip('\n'))
            istart += 1
            for i in range(ncoef):
                shapes[i] = float(content[istart])
                istart += 1
            shapes = shapes.reshape(npart,ndim)
            if verbose:
                print(shapes)
            out_data['partition_shapes'] = shapes

            if content[istart].strip('\n')!=_coeffs_write:
                raise ValueError('inconsistent read',content[istart].strip('\n'),'and expected',_coeffs_write)
            ncoef = nbasis * ndim + 1
            if nbasis>1:
                ncoef += ndim*(ndim+1)*0.5-ndim
            ncoef *= npart
            ncoef = int(ncoef)
            coeffs = np.zeros(ncoef)
            if verbose:
                print(content[istart].strip('\n'), ncoef)
            istart += 1
            for i in range(ncoef):
                coeffs[i] = float(content[istart])
                istart += 1
            coeffs = coeffs.reshape(npart,ncoef//npart).T
            coeffs = coeffs.ravel()
            if verbose:
                print(coeffs)
            out_data['basis_coeffs'] = np.array([coeffs])
        return out_data

    @tf.function
    def tf_transform(self, x):
        inter = x + self._transform_shift
        return tf.math.pow(tf.cast(tf.abs(inter), dtype=self._dtype), self._transform_power) * tf.math.sign(inter) + self._transform_sign_shift*tf.math.sign(inter)

    @tf.function
    def tf_untransform(self, x):
        o = self._transform_sign_shift*tf.math.sign(x)
        inter = tf.math.sign(x-o) * tf.math.pow(tf.cast(tf.abs(x-o), dtype=self._dtype), 1./self._transform_power)
        return inter - self._transform_shift

    @tf.function
    def tf_partitions_prenorm(self, x):
        # evaluate non-normalized partitions
        self._t_xmxp_scaled = (x - tf.transpose(self._t_xp)) * tf.transpose(self._t_sp)
        return tf.math.exp(-tf.reduce_sum(self._t_xmxp_scaled * self._t_xmxp_scaled, axis=1))

    @tf.function
    def tf_partitions(self, x):
        # evaluate normalized partitions
        self._t_nnp = self.tf_partitions_prenorm(x)
        return tf.transpose(tf.transpose(self._t_nnp) / tf.reduce_sum(self._t_nnp, axis=1))

    @tf.function
    def tf_predict(self, x, t_p):
        # evaluate basis
        for ib in range(self._t_basis_coeffs.shape[0]):
            if self._basis_type == 'constant':
                t_basis = self._t_basis_coeffs[ib, :self._np]
            elif self._basis_type == 'linear':
                t_basis = self._t_basis_coeffs[ib, :self._np]
                for i in range(self._nd):
                    t_basis += self._t_basis_coeffs[ib, self._np*(i+1):self._np*(i+2)] * x[:, i:i+1, 0]
            elif self._basis_type == 'quadratic':
                if self._nd == 1:
                    t_basis = self._t_basis_coeffs[ib, :self._np] + \
                            self._t_basis_coeffs[ib, self._np:self._np*2] * x[:, :1, 0] + \
                            self._t_basis_coeffs[ib, self._np*2:self._np*3] * x[:, :1, 0] * x[:, :1, 0]
                elif self._nd == 2:
                    t_basis = self._t_basis_coeffs[ib, :self._np] + \
                                    self._t_basis_coeffs[ib, self._np:self._np*2] * x[:, :1, 0] + \
                                    self._t_basis_coeffs[ib, self._np*2:self._np*3] * x[:, 1:2, 0] + \
                                    self._t_basis_coeffs[ib, self._np*3:self._np*4] * x[:, :1, 0] * x[:, :1, 0] + \
                                    self._t_basis_coeffs[ib, self._np*4:self._np*5] * x[:, 1:2, 0] * x[:, 1:2, 0]
                    t_basis += self._t_basis_coeffs[ib, self._np*5:self._np*6] * x[:, :1, 0] * x[:, 1:2, 0] # crossterm
                elif self._nd == 3:
                    t_basis = self._t_basis_coeffs[ib, :self._np] + \
                                    self._t_basis_coeffs[ib, self._np:self._np*2] * x[:, :1, 0] + \
                                    self._t_basis_coeffs[ib, self._np*2:self._np*3] * x[:, 1:2, 0] + \
                                    self._t_basis_coeffs[ib, self._np*3:self._np*4] * x[:, 2:3, 0] + \
                                    self._t_basis_coeffs[ib, self._np*4:self._np*5] * x[:, :1, 0] * x[:, :1, 0] + \
                                    self._t_basis_coeffs[ib, self._np*5:self._np*6] * x[:, 1:2, 0] * x[:, 1:2, 0] + \
                                    self._t_basis_coeffs[ib, self._np*6:self._np*7] * x[:, 2:3, 0] * x[:, 2:3, 0]
                    t_basis += self._t_basis_coeffs[ib, self._np*7:self._np*8] * x[:, :1, 0] * x[:, 1:2, 0] # crossterm
                    t_basis += self._t_basis_coeffs[ib, self._np*8:self._np*9] * x[:, :1, 0] * x[:, 2:3, 0] # crossterm
                    t_basis += self._t_basis_coeffs[ib, self._np*9:self._np*10] * x[:, 2:3, 0] * x[:, 1:2, 0] # crossterm
                else:
                    raise ValueError("unsupported dimensionality + degree combo")
            else:
                raise ValueError("unsupported dimensionality + degree combo")
            # evaluate full network
            if ib==0:
                output = tf.reduce_sum(t_p * t_basis, axis=1)
            else:
                if ib==1:
                    output = tf.expand_dims(output, axis=1)
                temp = tf.expand_dims(tf.reduce_sum(t_p * t_basis, axis=1), axis=1)
                output = tf.concat([output, temp], 1)
        return output

    def build_training_graph(self, ivars, dvars, error_type='abs', constrain_positivity=False, istensor=False, verbose=False):
        """
        Construct the graph used during training (including defining the training errors) with the provided training data

        :param ivars:
            array of independent variables for training
        :param dvars:
            array of dependent variable(s) for training
        :param error_type:
            (optional, default ``'abs'``) the type of training error: relative ``'rel'`` or absolute ``'abs'``
        :param constrain_positivity:
            (optional, default False) when True, it penalizes the training error with :math:`f - |f|` for dependent variables :math:`f`. This can be useful when used in ``QoIAwareProjectionPOUnet``
        :param istensor:
            (optional, default False) whether to evaluate ivars and dvars as tensorflow Tensors or numpy arrays
        :param verbose:
            (options, default False) print out the number of the partition and basis parameters when True
        """
        if error_type != 'rel' and error_type != 'abs':
            raise ValueError("Supported error_type include rel and abs.")

        if istensor:
            if len(ivars.shape)!=3:
                raise ValueError("Expected ivars with 3 tensor dimensions but recieved",len(ivars.shape))
            self._t_xyt_prescale = ivars
            self._t_ft = dvars
        else:
            if len(ivars.shape)!=2:
                raise ValueError("Expected ivars with 2 array dimensions but recieved",len(ivars.shape))
            self._t_xyt_prescale = tf.Variable(np.expand_dims(ivars, axis=2), name='eval_pts', dtype=self._dtype)
            self._t_ft = tf.Variable(dvars, name='eval_qoi', dtype=self._dtype)

        if ivars.shape[1] != self._nd:
            raise ValueError("Dimensionality of ivars",ivars.shape[1],'does not match expected dimensionality',self._nd)
        self._t_ft = self.tf_transform(self._t_ft)
        self._t_xyt = (self._t_xyt_prescale - self._t_ivar_center)*self._t_inv_ivar_scale

        self._t_penalty = self._t_ft - tf.cast(tf.abs(self._t_ft), dtype=self._dtype)

        ftdim = len(self._t_ft.shape)
        self._nft = float(self._t_ft.get_shape().as_list()[1]) if ftdim>1 else 1.

        if self._basis_coeffs is None:
            if self._basis_type == 'constant':
                self._basis_coeffs = np.ones((1, self._np)) / self._np
            elif self._basis_type == 'linear':
                self._basis_coeffs = np.hstack((np.ones((1, self._np)) / self._np, np.zeros((1, self._np*self._nd))))
            elif self._basis_type == 'quadratic':
                if self._nd == 1:
                    self._basis_coeffs = np.hstack((np.ones((1, self._np)) / self._np, np.zeros((1, self._np*2))))
                elif self._nd == 2:
                    self._basis_coeffs = np.hstack((np.ones((1, self._np)) / self._np, np.zeros((1, self._np*5))))
                elif self._nd == 3:
                    self._basis_coeffs = np.hstack((np.ones((1, self._np)) / self._np, np.zeros((1, self._np*9))))
                else:
                    raise ValueError('unsupported dimensionality + degree combo')
            else:
                raise ValueError('unsupported dimensionality + degree combo')

            if ftdim>1:
                orig_coeffs = self._basis_coeffs.copy()
                for i in range(1,self._t_ft.shape[1]):
                    self._basis_coeffs = np.vstack((self._basis_coeffs, orig_coeffs))
            self._isready = True
        else:
            if ftdim>1:
                if self._basis_coeffs.shape[0] != self._t_ft.shape[1]:
                    raise ValueError("Expected",self._basis_coeffs.shape[0],"dependent variables but received",self._t_ft.shape[1])
            else:
                if self._basis_coeffs.shape[0] != 1:
                    raise ValueError("Expected",self._basis_coeffs.shape[0],"dependent variables but received 1")

        self._t_basis_coeffs = tf.Variable(self._basis_coeffs, name='basis_coeffs', dtype=self._dtype)
        if verbose:
            print('Constructing a graph for',int(self._nft),'dependent variables with',self._t_xp.shape[0],'partitions and',self._t_basis_coeffs.shape[1],'coefficients per variable.')

        # predictions
        self._t_p = self.tf_partitions(self._t_xyt)
        self._t_pred = self.tf_predict(self._t_xyt, self._t_p)

        # evaluate error
        self._rel_err = (self._t_pred - self._t_ft) # start with absolute error
        if error_type == 'rel':
            # divide by appropriate factor if want relative error
            self._rel_err /= (tf.cast(tf.abs(self._t_ft), dtype=self._dtype) + tf.constant(1.e-4, name='rel_den', dtype=self._dtype))

        self._t_msre = tf.reduce_mean(tf.math.square(self._rel_err))
        self._t_infre = tf.reduce_max(tf.cast(tf.abs(self._rel_err), dtype=self._dtype))
        self._t_ssre = tf.reduce_sum(tf.math.square(self._rel_err))/self._nft

        # set training error
        self._train_err = self._t_ssre
        if constrain_positivity:
            self._train_err += tf.cast(tf.abs(tf.reduce_sum(self._t_penalty)/self._nft), dtype=self._dtype)

        # optimizers
        self._part_l2_opt = tf.train.AdamOptimizer(learning_rate=self._lr).minimize(self._train_err, var_list=[self._t_xp, self._t_sp])

        # set up lstsq
        if self._basis_type == 'constant':
            self._Amat = tf.identity(self._t_p)
        elif self._basis_type == 'linear':
            self._Amat = tf.concat([self._t_p, self._t_p*self._t_xyt[:, :1, 0]], 1)
            for i in range(1, self._nd):
                self._Amat = tf.concat([self._Amat, self._t_p*self._t_xyt[:, i:i+1, 0]], 1)
        elif self._basis_type == 'quadratic':
            if self._nd==1:
                self._Amat = tf.concat([self._t_p, self._t_p*self._t_xyt[:, :1, 0]], 1)
                self._Amat = tf.concat([self._Amat, self._t_p*self._t_xyt[:, :1, 0]*self._t_xyt[:, :1, 0]], 1)
            elif self._nd==2:
                self._Amat = tf.concat([tf.concat([self._t_p, self._t_p*self._t_xyt[:, :1, 0]], 1), self._t_p*self._t_xyt[:, 1:2, 0]], 1)
                self._Amat = tf.concat([self._Amat, self._t_p*self._t_xyt[:, :1, 0]*self._t_xyt[:, :1, 0]], 1)
                self._Amat = tf.concat([self._Amat, self._t_p*self._t_xyt[:, 1:2, 0]*self._t_xyt[:, 1:2, 0]], 1)
                self._Amat = tf.concat([self._Amat, self._t_p*self._t_xyt[:, :1, 0]*self._t_xyt[:, 1:2, 0]], 1) # crossterm
            elif self._nd==3:
                self._Amat = tf.concat([tf.concat([self._t_p, self._t_p*self._t_xyt[:, :1, 0]], 1), self._t_p*self._t_xyt[:, 1:2, 0]], 1)
                self._Amat = tf.concat([self._Amat, self._t_p*self._t_xyt[:, 2:3, 0]], 1)
                self._Amat = tf.concat([self._Amat, self._t_p*self._t_xyt[:, :1, 0]*self._t_xyt[:, :1, 0]], 1)
                self._Amat = tf.concat([self._Amat, self._t_p*self._t_xyt[:, 1:2, 0]*self._t_xyt[:, 1:2, 0]], 1)
                self._Amat = tf.concat([self._Amat, self._t_p*self._t_xyt[:, 2:3, 0]*self._t_xyt[:, 2:3, 0]], 1)
                self._Amat = tf.concat([self._Amat, self._t_p*self._t_xyt[:, :1, 0]*self._t_xyt[:, 1:2, 0]], 1) # crossterm
                self._Amat = tf.concat([self._Amat, self._t_p*self._t_xyt[:, :1, 0]*self._t_xyt[:, 2:3, 0]], 1) # crossterm
                self._Amat = tf.concat([self._Amat, self._t_p*self._t_xyt[:, 2:3, 0]*self._t_xyt[:, 1:2, 0]], 1) # crossterm
            else:
                raise ValueError("unsupported dimensionality + degree combo")
        else:
            raise ValueError("unsupported dimensionality + degree combo")

        if ftdim>1:
            rhs = self._t_ft
        else:
            rhs = tf.expand_dims(self._t_ft, axis=1)

        if error_type == 'rel':
            # if training error is relative
            if self._nft>1:
                raise ValueError("Cannot support relative error lstsqr for more than 1 variable")
            else:
                lstsq_rel_coeff = 1. / (tf.abs(rhs) + 1.e-4) # must be same format as error for partitions
                self._Amat *= lstsq_rel_coeff
                rhs *= lstsq_rel_coeff

        # perform lstsq
        self._lstsq = self._t_basis_coeffs.assign(tf.transpose(tf.linalg.lstsq(self._Amat, rhs, l2_regularizer=self._l2reg)))

        self._sess.run(tf.global_variables_initializer())

        msre, infre, ssre = self._sess.run(self._t_msre), self._sess.run(self._t_infre), self._sess.run(self._t_ssre)
        self._iterations = list([0])
        self._training_archive = {'mse': list([msre]), 'inf': list([infre]), 'sse': list([ssre]), 'data':list([self.__getstate__()])}
        self._built_graph = True

    def update_lr(self, lr):
        """
        update the learning rate for training

        :param lr:
            float for the learning rate
        """
        print('updating lr:', lr)
        self._sess.run(self._lr.assign(lr))

    def update_l2reg(self, l2reg):
        """
        update the least-squares regularization for training

        :param l2reg:
            float for the least-squares regularization
        """
        print('updating l2reg:', l2reg)
        self._sess.run(self._l2reg.assign(l2reg))

    def lstsq(self, verbose=True):
        """
        update the basis coefficients with least-squares regression

        :param verbose:
            (optional, default True) prints when least-squares solve is performed when True
        """
        if not self._built_graph:
            raise ValueError("Need to call build_training_graph before lstsq.")
        if verbose:
            print('performing least-squares solve')
        self._sess.run(self._lstsq)

    def train(self, iterations, archive_rate=100, use_best_archive_sse=True, verbose=False):
        """
        Performs training using a least-squares gradient descent block coordinate descent strategy.
        This alternates between updating the partition parameters with gradient descent and updating the basis coefficients with least-squares.

        :param iterations:
            integer for number of training iterations to perform
        :param archive_rate:
            (optional, default 100) the rate at which the errors and parameters are archived during training. These can be accessed with the ``training_archive`` attribute
        :param use_best_archive_sse:
            (optional, default True) when True will set the POUnet parameters to those with the lowest error observed during training,
            otherwise the parameters from the last iteration are used
        :param verbose:
            (optional, default False) when True will print progress
        """
        if not self._built_graph:
            raise ValueError("Need to call build_training_graph before train.")
        if use_best_archive_sse and archive_rate>iterations:
            raise ValueError("Cannot archive the best parameters with archive_rate", archive_rate,'over',iterations,'iterations.')

        if verbose:
            print('-' * 60)
            print(f'  {"iteration":>10} | {"mean sqr":>10} | {"% max":>10}  | {"sum sqr":>10}')
            print('-' * 60)

        if use_best_archive_sse:
            best_error = self._sess.run(self._t_ssre).copy()
            best_centers = self._sess.run(self._t_xp).copy()
            best_shapes = self._sess.run(self._t_sp).copy()
            best_coeffs = self._sess.run(self._t_basis_coeffs).copy()

        for i in range(iterations):
            self._sess.run(self._part_l2_opt) # update partitions
            self._sess.run(self._lstsq) # update basis coefficients

            if not (i + 1) % archive_rate:
                msre, infre, ssre = self._sess.run(self._t_msre), self._sess.run(self._t_infre), self._sess.run(self._t_ssre)
                if verbose:
                    print(f'  {i + 1:10} | {msre:10.2e} | {100. * infre:10.2f}% | {ssre:10.2e}')
                self._iterations.append(self._iterations[-1] + archive_rate)
                self._training_archive['mse'].append(msre)
                self._training_archive['inf'].append(infre)
                self._training_archive['sse'].append(ssre)
                self._training_archive['data'].append(self.__getstate__())

                if use_best_archive_sse:
                    if ssre < best_error:
                        if verbose:
                            print('resetting best error')
                        best_error = ssre.copy()
                        best_centers = self._sess.run(self._t_xp).copy()
                        best_shapes = self._sess.run(self._t_sp).copy()
                        best_coeffs = self._sess.run(self._t_basis_coeffs).copy()
        if use_best_archive_sse:
            self._sess.run(self._t_xp.assign(best_centers))
            self._sess.run(self._t_sp.assign(best_shapes))
            self._sess.run(self._t_basis_coeffs.assign(best_coeffs))

    @tf.function
    def tf_call(self, xeval):
        if self._isready:
            xeval_tf = (xeval - self._t_ivar_center)*self._t_inv_ivar_scale
            t_p = self.tf_partitions(xeval_tf)
            return self.tf_untransform(self.tf_predict(xeval_tf, t_p))
        else:
            raise ValueError("basis coefficients have not been set.")

    def __call__(self, xeval):
        """
        evaluate the POUnet

        :param xeval:
            array of independent variable query points

        :return:
            array of POUnet predictions
        """
        if xeval.shape[1] != self._nd:
            raise ValueError("Dimensionality of inputs",xeval.shape[1],'does not match expected dimensionality',self._nd)
        xeval_tf_prescale = tf.Variable(np.expand_dims(xeval, axis=2), name='eval_pts', dtype=self._dtype)
        self._sess.run(tf.variables_initializer([xeval_tf_prescale]))
        pred = self.tf_call(xeval_tf_prescale)
        return self._sess.run(pred)

    def __getstate__(self):
        """dictionary of current POUnet parameters"""
        return dict(partition_centers=self.partition_centers,
                    partition_shapes=self.partition_shapes,
                    basis_type=self.basis_type,
                    ivar_center=self.ivar_center,
                    ivar_scale=self.ivar_scale,
                    basis_coeffs=self.basis_coeffs,
                    transform_power=self._transform_power,
                    transform_shift=self._transform_shift,
                    transform_sign_shift=self._transform_sign_shift,
                    dtype=self.dtype
                   )

    def derivatives(self, xeval, dvar_idx=0):
        """
        evaluate the POUnet derivatives

        :param xeval:
            array of independent variable query points
        :param dvar_idx:
            (optional, default 0) index for the dependent variable whose derivatives are being evaluated

        :return:
            array of POUnet derivative evaluations
        """
        if xeval.shape[1] != self._nd:
            raise ValueError("Dimensionality of inputs",xeval.shape[1],'does not match expected dimensionality',self._nd)
        xeval_tf_prescale = tf.Variable(np.expand_dims(xeval, axis=2), name='eval_pts', dtype=self._dtype)
        self._sess.run(tf.variables_initializer([xeval_tf_prescale]))
        allpreds = []
        with tf.GradientTape() as tape:
            pred = self.tf_call(xeval_tf_prescale)
            if len(pred.shape)>1:
                pred = pred[:,dvar_idx]
        der = tape.gradient(pred, xeval_tf_prescale)
        return self._sess.run(der)[:,:,0]


    def partition_prenorm(self, xeval):
        """
        evaluate the POUnet partitions prior to normalization

        :param xeval:
            array of independent variable query points

        :return:
            array of POUnet RBF partition evaluations before normalization
        """
        if xeval.shape[1] != self._nd:
            raise ValueError("Dimensionality of inputs",xeval.shape[1],'does not match expected dimensionality',self._nd)
        xeval_tf_prescale = tf.Variable(np.expand_dims(xeval, axis=2), name='eval_pts', dtype=self._dtype)
        self._sess.run(tf.variables_initializer([xeval_tf_prescale]))
        xeval_tf = (xeval_tf_prescale - self._t_ivar_center)*self._t_inv_ivar_scale
        t_nnp = self.tf_partitions_prenorm(xeval_tf)
        return self._sess.run(t_nnp)

    def write_data_to_file(self, filename):
        """
        Save class data to a specified file using pickle. This does not include the archived data from training,
        which can be separately accessed with training_archive and saved outside of ``PartitionOfUnityNetwork``.

        :param filename:
            string
        """
        with open(filename, 'wb') as file_output:
            pickle.dump(self.__getstate__(), file_output)

    def write_data_to_txt(self, filename, nformat='%.14e'):
        """
        Save data to a specified txt file. This may be used to read POUnet parameters into other languages such as C++

        :param filename:
            string
        """
        if self._basis_coeffs.shape[0]>1:
            raise ValueError("Cannot read/write to txt for multiple dependent variables. Try ``write_data_to_pkl`` instead.")
        with open(filename, 'w') as f:
            f.write("POUNET" + '\n')

            f.write(_ndim_write + '\n')
            f.write(str(self._nd) + '\n')

            f.write(_npartition_write + '\n')
            f.write(str(self._np) + '\n')

            f.write(_nbasis_write + '\n')
            if self._basis_type == 'constant':
                nbasis=0
            elif self._basis_type == 'linear':
                nbasis=1
            elif self._basis_type == 'quadratic':
                nbasis=2
            else:
                raise ValueError("Unsupported basis type.")
            f.write(str(nbasis) + '\n')

            f.write(_floattype_write + '\n')
            f.write(self._dtype_str + '\n')

            f.write(_tpower_write + '\n')
            f.write(str(self._transform_power) + '\n')

            f.write(_tshift_write + '\n')
            f.write(str(self._transform_shift) + '\n')

            f.write(_tsignshift_write + '\n')
            f.write(str(self._transform_sign_shift) + '\n')

            f.write(_ivarcenter_write + '\n')
            np.savetxt(f, self._ivar_center.ravel(), fmt=nformat)
            f.write(_ivarscale_write + '\n')
            np.savetxt(f, 1./self._inv_ivar_scale.ravel(), fmt=nformat)

            f.write(_pcenters_write + '\n')
            pcenters = self.partition_centers.T
            np.savetxt(f, pcenters.ravel(order='F'), fmt=nformat)

            f.write(_pshapes_write + '\n')
            pshapes = self.partition_shapes.T
            np.savetxt(f, pshapes.ravel(order='F'), fmt=nformat)

            f.write(_coeffs_write + '\n')
            coeffs = self.basis_coeffs[0,:]
            totalbasis = coeffs.size//self._np
            coeffs = coeffs.reshape(totalbasis, self._np)
            np.savetxt(f, coeffs.ravel(order='F'), fmt=nformat)

    @property
    def iterations(self):
        return np.array(self._iterations)

    @property
    def training_archive(self):
        return {e: np.array(self._training_archive[e]) if e != 'data' else self._training_archive[e] for e in self._training_archive}

    def get_partition_sum(self, xeval):
        nnp = self.partition_prenorm(xeval)
        return np.sum(nnp, axis=1)

    @property
    def partition_centers(self):
        return self._sess.run(self._t_xp)

    @property
    def partition_shapes(self):
        return self._sess.run(self._t_sp)

    @property
    def basis_type(self):
        return self._basis_type

    @property
    def dtype(self):
        return self._dtype_str

    @property
    def basis_coeffs(self):
        return self._sess.run(self._t_basis_coeffs)

    @property
    def ivar_center(self):
        return self._ivar_center

    @property
    def ivar_scale(self):
        return 1./self._inv_ivar_scale

################################################################################
#
# Plotting
#
################################################################################

def plot_2d_regression(x, observed, predicted, x_label=None, y_label=None, color_observed=None, color_predicted=None, figure_size=(7,7), title=None, save_filename=None):
    """
    Plots the result of regression of a dependent variable on top
    of a one-dimensional manifold defined by a single independent variable ``x``.

    **Example:**

    .. code:: python

        from PCAfold import PCA, plot_2d_regression
        import numpy as np

        # Generate dummy data set:
        X = np.random.rand(100,10)

        # Obtain two-dimensional manifold from PCA:
        pca_X = PCA(X)
        PCs = pca_X.transform(X)
        X_rec = pca_X.reconstruct(PCs)

        # Plot the manifold:
        plt = plot_2d_regression(X[:,0],
                                 X[:,0],
                                 X_rec[:,0],
                                 x_label='$x$',
                                 y_label='$y$',
                                 color_observed='k',
                                 color_predicted='r',
                                 figure_size=(10,10),
                                 title='2D regression',
                                 save_filename='2d-regression.pdf')
        plt.close()

    :param x:
        ``numpy.ndarray`` specifying the variable on the :math:`x`-axis. It should be of size ``(n_observations,)`` or ``(n_observations,1)``.
    :param observed:
        ``numpy.ndarray`` specifying the observed values of a single dependent variable.
        It should be of size ``(n_observations,)`` or ``(n_observations, 1)``.
    :param predicted:
        ``numpy.ndarray`` specifying the predicted values of a single dependent variable.
        It should be of size ``(n_observations,)`` or ``(n_observations, 1)``.
    :param x_label: (optional)
        ``str`` specifying :math:`x`-axis label annotation. If set to ``None``
        label will not be plotted.
    :param y_label: (optional)
        ``str`` specifying :math:`y`-axis label annotation. If set to ``None``
        label will not be plotted.
    :param color_observed: (optional)
        ``str`` specifying the color of the plotted observed variable.
    :param color_predicted: (optional)
        ``str`` specifying the color of the plotted predicted variable.
    :param figure_size: (optional)
        ``tuple`` specifying figure size.
    :param title: (optional)
        ``str`` specifying plot title. If set to ``None`` title will not be
        plotted.
    :param save_filename: (optional)
        ``str`` specifying plot save location/filename. If set to ``None``
        plot will not be saved. You can also set a desired file extension,
        for instance ``.pdf``. If the file extension is not specified, the default
        is ``.png``.

    :return:
        - **plt** - ``matplotlib.pyplot`` plot handle.
    """

    if not isinstance(x, np.ndarray):
        raise ValueError("Parameter `x` has to be of type `numpy.ndarray`.")

    try:
        (n_x,) = np.shape(x)
        n_var_x = 1
    except:
        (n_x, n_var_x) = np.shape(x)

    if n_var_x != 1:
        raise ValueError("Parameter `x` has to be a 0D or 1D vector.")

    if not isinstance(observed, np.ndarray):
        raise ValueError("Parameter `observed` has to be of type `numpy.ndarray`.")

    try:
        (n_observed,) = np.shape(observed)
        n_var_observed = 1
    except:
        (n_observed, n_var_observed) = np.shape(observed)

    if n_var_observed != 1:
        raise ValueError("Parameter `observed` has to be a 0D or 1D vector.")

    if not isinstance(predicted, np.ndarray):
        raise ValueError("Parameter `predicted` has to be of type `numpy.ndarray`.")

    try:
        (n_predicted,) = np.shape(predicted)
        n_var_predicted = 1
    except:
        (n_predicted, n_var_predicted) = np.shape(predicted)

    if n_var_predicted != 1:
        raise ValueError("Parameter `predicted` has to be a 0D or 1D vector.")

    if n_observed != n_predicted:
        raise ValueError("Parameter `observed` has different number of elements than `predicted`.")

    if n_x != n_observed:
        raise ValueError("Parameter `observed` has different number of elements than `x`.")

    if n_x != n_predicted:
        raise ValueError("Parameter `predicted` has different number of elements than `x`.")

    if x_label is not None:
        if not isinstance(x_label, str):
            raise ValueError("Parameter `x_label` has to be of type `str`.")

    if y_label is not None:
        if not isinstance(y_label, str):
            raise ValueError("Parameter `y_label` has to be of type `str`.")

    if color_observed is not None:
        if not isinstance(color_observed, str):
            raise ValueError("Parameter `color_observed` has to be of type `str`.")
    else:
        color_observed = '#191b27'

    if color_predicted is not None:
        if not isinstance(color_predicted, str):
            raise ValueError("Parameter `color_predicted` has to be of type `str`.")
    else:
        color_predicted = '#C7254E'

    if not isinstance(figure_size, tuple):
        raise ValueError("Parameter `figure_size` has to be of type `tuple`.")

    if title is not None:
        if not isinstance(title, str):
            raise ValueError("Parameter `title` has to be of type `str`.")

    if save_filename is not None:
        if not isinstance(save_filename, str):
            raise ValueError("Parameter `save_filename` has to be of type `str`.")

    fig = plt.figure(figsize=figure_size)

    scat = plt.scatter(x.ravel(), observed.ravel(), c=color_observed, marker='o', s=scatter_point_size, alpha=0.1)
    scat = plt.scatter(x.ravel(), predicted.ravel(), c=color_predicted, marker='o', s=scatter_point_size, alpha=0.4)

    if x_label != None: plt.xlabel(x_label, **csfont, fontsize=font_labels)
    if y_label != None: plt.ylabel(y_label, **csfont, fontsize=font_labels)
    plt.xticks(fontsize=font_axes, **csfont)
    plt.yticks(fontsize=font_axes, **csfont)
    plt.grid(alpha=grid_opacity)
    lgnd = plt.legend(['Observed', 'Predicted'], fontsize=font_legend, loc="best")
    lgnd.legendHandles[0]._sizes = [marker_size*5]
    lgnd.legendHandles[1]._sizes = [marker_size*5]

    if title != None: plt.title(title, **csfont, fontsize=font_title)
    if save_filename != None: plt.savefig(save_filename, dpi=save_dpi, bbox_inches='tight')

    return plt

# ------------------------------------------------------------------------------

def plot_2d_regression_scalar_field(grid_bounds, regression_model, x=None, y=None, resolution=(10,10), extension=(0,0), x_label=None, y_label=None, s_field=None, s_manifold=None, manifold_color=None, colorbar_label=None, color_map='viridis', colorbar_range=None, manifold_alpha=1, grid_on=True, figure_size=(7,7), title=None, save_filename=None):
    """
    Plots a 2D field of a regressed scalar dependent variable.
    A two-dimensional manifold can be additionally plotted on top of the field.

    **Example:**

    .. code:: python

        from PCAfold import PCA, KReg, plot_2d_regression_scalar_field
        import numpy as np

        # Generate dummy data set:
        X = np.random.rand(100,2)
        Z = np.random.rand(100,1)

        # Train the kernel regression model:
        model = KReg(X, Z)

        # Define the regression model:
        def regression_model(query):

            predicted = model.predict(query, 'nearest_neighbors_isotropic', n_neighbors=1)[:,0]

            return predicted

        # Define the bounds for the scalar field:
        grid_bounds = ([np.min(X[:,0]),np.max(X[:,0])],[np.min(X[:,1]),np.max(X[:,1])])

        # Plot the regressed scalar field:
        plt = plot_2d_regression_scalar_field(grid_bounds,
                                            regression_model,
                                            x=X[:,0],
                                            y=X[:,1],
                                            resolution=(100,100),
                                            extension=(10,10),
                                            x_label='$X_1$',
                                            y_label='$X_2$',
                                            s_field=4,
                                            s_manifold=60,
                                            manifold_color=Z,
                                            colorbar_label='$Z_1$',
                                            color_map='inferno',
                                            colorbar_range=(0,1),
                                            manifold_alpha=1,
                                            grid_on=False,
                                            figure_size=(10,6),
                                            title='2D regressed scalar field',
                                            save_filename='2D-regressed-scalar-field.pdf')
        plt.close()

    :param grid_bounds:
        ``tuple`` of ``list`` specifying the bounds of the dependent variable on the :math:`x` and :math:`y` axis.
    :param regression_model:
        ``function`` that outputs the predicted vector using the regression model.
        It should take as input a ``numpy.ndarray`` of size ``(1,2)``, where the two
        elements specify the first and second independent variable values. It should output
        a ``float`` specifying the regressed scalar value at that input.
    :param x: (optional)
        ``numpy.ndarray`` specifying the variable on the :math:`x`-axis.
        It should be of size ``(n_observations,)`` or ``(n_observations,1)``.
        It can be used to plot a 2D manifold on top of the streamplot.
    :param y: (optional)
        ``numpy.ndarray`` specifying the variable on the :math:`y`-axis.
        It should be of size ``(n_observations,)`` or ``(n_observations,1)``.
        It can be used to plot a 2D manifold on top of the streamplot.
    :param resolution: (optional)
        ``tuple`` of ``int`` specifying the resolution of the streamplot grid on the :math:`x` and :math:`y` axis.
    :param extension: (optional)
        ``tuple`` of ``float`` or ``int`` specifying a percentage by which the
        dependent variable should be extended beyond on the :math:`x` and :math:`y` axis, beyond what has been specified by the ``grid_bounds`` parameter.
    :param x_label: (optional)
        ``str`` specifying :math:`x`-axis label annotation. If set to ``None``
        label will not be plotted.
    :param y_label: (optional)
        ``str`` specifying :math:`y`-axis label annotation. If set to ``None``
        label will not be plotted.
    :param s_field: (optional)
        ``int`` or ``float`` specifying the scatter point size for the scalar field.
    :param s_manifold: (optional)
        ``int`` or ``float`` specifying the scatter point size for the manifold.
    :param manifold_color: (optional)
        vector or string specifying color for the manifold. If it is a
        vector, it has to have length consistent with the number of observations
        in ``x`` and ``y`` vectors. It should be of type ``numpy.ndarray`` and size
        ``(n_observations,)`` or ``(n_observations,1)``.
        It can also be set to a string specifying the color directly, for
        instance ``'r'`` or ``'#006778'``.
        If not specified, manifold will be plotted in black.
    :param colorbar_label: (optional)
        ``str`` specifying colorbar label annotation.
    :param color_map: (optional)
        ``str`` or ``matplotlib.colors.ListedColormap`` specifying the colormap to use as per ``matplotlib.cm``. Default is ``'viridis'``.
    :param colorbar_range: (optional)
        ``tuple`` specifying the lower and the upper bound for the colorbar range.
    :param manifold_alpha: (optional)
        ``float`` or ``int`` specifying the opacity of the plotted manifold.
    :param grid_on:
        ``bool`` specifying whether grid should be plotted.
    :param figure_size: (optional)
        ``tuple`` specifying figure size.
    :param title: (optional)
        ``str`` specifying plot title. If set to ``None`` title will not be
        plotted.
    :param save_filename: (optional)
        ``str`` specifying plot save location/filename. If set to ``None``
        plot will not be saved. You can also set a desired file extension,
        for instance ``.pdf``. If the file extension is not specified, the default
        is ``.png``.

    :return:
        - **plt** - ``matplotlib.pyplot`` plot handle.
    """

    if not isinstance(grid_bounds, tuple):
        raise ValueError("Parameter `grid_bounds` has to be of type `tuple`.")

    if not callable(regression_model):
        raise ValueError("Parameter `regression_model` has to be of type `function`.")

    if x is not None:

        if not isinstance(x, np.ndarray):
            raise ValueError("Parameter `x` has to be of type `numpy.ndarray`.")

        try:
            (n_x,) = np.shape(x)
            n_var_x = 1
        except:
            (n_x, n_var_x) = np.shape(x)

        if n_var_x != 1:
            raise ValueError("Parameter `x` has to be a 0D or 1D vector.")

    if y is not None:

        if not isinstance(y, np.ndarray):
            raise ValueError("Parameter `y` has to be of type `numpy.ndarray`.")

        try:
            (n_y,) = np.shape(y)
            n_var_y = 1
        except:
            (n_y, n_var_y) = np.shape(y)

        if n_var_y != 1:
            raise ValueError("Parameter `y` has to be a 0D or 1D vector.")

        if n_x != n_y:
            raise ValueError("Parameter `x` has different number of elements than `y`.")

    if not isinstance(resolution, tuple):
        raise ValueError("Parameter `resolution` has to be of type `tuple`.")

    if not isinstance(extension, tuple):
        raise ValueError("Parameter `extension` has to be of type `tuple`.")

    if x_label is not None:
        if not isinstance(x_label, str):
            raise ValueError("Parameter `x_label` has to be of type `str`.")

    if y_label is not None:
        if not isinstance(y_label, str):
            raise ValueError("Parameter `y_label` has to be of type `str`.")

    if s_field is None:
        s_field = scatter_point_size
    else:
        if not isinstance(s_field, int) and not isinstance(s_field, float):
            raise ValueError("Parameter `s_field` has to be of type `int` or `float`.")

    if s_manifold is None:
        s_manifold = scatter_point_size
    else:
        if not isinstance(s_manifold, int) and not isinstance(s_manifold, float):
            raise ValueError("Parameter `s_manifold` has to be of type `int` or `float`.")

    if manifold_color is not None:
        if not isinstance(manifold_color, str):
            if not isinstance(manifold_color, np.ndarray):
                raise ValueError("Parameter `manifold_color` has to be `None`, or of type `str` or `numpy.ndarray`.")

    if isinstance(manifold_color, np.ndarray):

        try:
            (n_color,) = np.shape(manifold_color)
            n_var_color = 1
        except:
            (n_color, n_var_color) = np.shape(manifold_color)

        if n_var_color != 1:
            raise ValueError("Parameter `manifold_color` has to be a 0D or 1D vector.")

        if n_color != n_x:
            raise ValueError("Parameter `manifold_color` has different number of elements than `x` and `y`.")

    if colorbar_label is not None:
        if not isinstance(colorbar_label, str):
            raise ValueError("Parameter `colorbar_label` has to be of type `str`.")

    if not isinstance(color_map, str):
        if not isinstance(color_map, ListedColormap):
            raise ValueError("Parameter `color_map` has to be of type `str` or `matplotlib.colors.ListedColormap`.")

    if colorbar_range is not None:
        if not isinstance(colorbar_range, tuple):
            raise ValueError("Parameter `colorbar_range` has to be of type `tuple`.")
        else:
            (cbar_min, cbar_max) = colorbar_range

    if manifold_alpha is not None:
        if not isinstance(manifold_alpha, float) and not isinstance(manifold_alpha, int):
            raise ValueError("Parameter `manifold_alpha` has to be of type `float`.")

    if not isinstance(grid_on, bool):
        raise ValueError("Parameter `grid_on` has to be of type `bool`.")

    if not isinstance(figure_size, tuple):
        raise ValueError("Parameter `figure_size` has to be of type `tuple`.")

    if title is not None:
        if not isinstance(title, str):
            raise ValueError("Parameter `title` has to be of type `str`.")

    if save_filename is not None:
        if not isinstance(save_filename, str):
            raise ValueError("Parameter `save_filename` has to be of type `str`.")

    (x_extend, y_extend) = extension
    (x_resolution, y_resolution) = resolution
    ([x_minimum, x_maximum], [y_minimum, y_maximum]) = grid_bounds

    # Create extension in both dimensions:
    x_extension = x_extend/100.0 * abs(x_minimum - x_maximum)
    y_extension = y_extend/100.0 * abs(y_minimum - y_maximum)

    # Create grid points for the independent variables where regression will be applied:
    x_grid = np.linspace(x_minimum-x_extension, x_maximum+x_extension, x_resolution)
    y_grid = np.linspace(y_minimum-y_extension, y_maximum+y_extension, y_resolution)
    xy_mesh = np.meshgrid(x_grid, y_grid, indexing='xy')

    # Evaluate the predicted scalar using the regression model:
    regressed_scalar = np.zeros((y_grid.size*x_grid.size,))

    for i in range(0,y_grid.size*x_grid.size):

                regression_input = np.reshape(np.array([xy_mesh[0].ravel()[i], xy_mesh[1].ravel()[i]]), [1,2])
                regressed_scalar[i] = regression_model(regression_input)

    fig = plt.figure(figsize=figure_size)

    if colorbar_range is not None:
        scat_field = plt.scatter(xy_mesh[0].ravel(), xy_mesh[1].ravel(), c=regressed_scalar, cmap=color_map, s=s_field, vmin=cbar_min, vmax=cbar_max)
    else:
        scat_field = plt.scatter(xy_mesh[0].ravel(), xy_mesh[1].ravel(), c=regressed_scalar, cmap=color_map, s=s_field)

    if (x is not None) and (y is not None):

        if manifold_color is None:
            scat = plt.scatter(x.ravel(), y.ravel(), c='k', marker='o', s=s_manifold, edgecolor='none', alpha=manifold_alpha)
        elif isinstance(manifold_color, str):
            scat = plt.scatter(x.ravel(), y.ravel(), c=manifold_color, cmap=color_map, marker='o', s=s_manifold, edgecolor='none', alpha=manifold_alpha)
        elif isinstance(manifold_color, np.ndarray):
            if colorbar_range is not None:
                scat = plt.scatter(x.ravel(), y.ravel(), c=manifold_color.ravel(), cmap=color_map, marker='o', s=s_manifold, edgecolor='none', vmin=cbar_min, vmax=cbar_max, alpha=manifold_alpha)
            else:
                scat = plt.scatter(x.ravel(), y.ravel(), c=manifold_color.ravel(), cmap=color_map, marker='o', s=s_manifold, edgecolor='none', vmin=np.min(regressed_scalar), vmax=np.max(regressed_scalar), alpha=manifold_alpha)

    cb = fig.colorbar(scat_field)
    cb.ax.tick_params(labelsize=font_colorbar_axes)
    if colorbar_label is not None: cb.set_label(colorbar_label, fontsize=font_colorbar, rotation=0, horizontalalignment='left')
    if colorbar_range is not None: plt.clim(cbar_min, cbar_max)

    if x_label is not None: plt.xlabel(x_label, **csfont, fontsize=font_labels)
    if y_label is not None: plt.ylabel(y_label, **csfont, fontsize=font_labels)
    plt.xlim([np.min(x_grid),np.max(x_grid)])
    plt.ylim([np.min(y_grid),np.max(y_grid)])
    plt.xticks(fontsize=font_axes, **csfont)
    plt.yticks(fontsize=font_axes, **csfont)
    if grid_on: plt.grid(alpha=grid_opacity)

    if title is not None: plt.title(title, **csfont, fontsize=font_title)
    if save_filename is not None: plt.savefig(save_filename, dpi=save_dpi, bbox_inches='tight')

    return plt

# ------------------------------------------------------------------------------

def plot_2d_regression_streamplot(grid_bounds, regression_model, x=None, y=None, resolution=(10,10), extension=(0,0), color='k', x_label=None, y_label=None, s_manifold=None, manifold_color=None, colorbar_label=None, color_map='viridis', colorbar_range=None, manifold_alpha=1, grid_on=True, figure_size=(7,7), title=None, save_filename=None):
    """
    Plots a streamplot of a regressed vector field of a dependent variable.
    A two-dimensional manifold can be additionally plotted on top of the streamplot.

    **Example:**

    .. code:: python

        from PCAfold import PCA, KReg, plot_2d_regression_streamplot
        import numpy as np

        # Generate dummy data set:
        X = np.random.rand(100,5)
        S_X = np.random.rand(100,5)

        # Obtain two-dimensional manifold from PCA:
        pca_X = PCA(X, n_components=2)
        PCs = pca_X.transform(X)
        S_Z = pca_X.transform(S_X, nocenter=True)

        # Train the kernel regression model:
        model = KReg(PCs, S_Z)

        # Define the regression model:
        def regression_model(query):

            predicted = model.predict(query, 'nearest_neighbors_isotropic', n_neighbors=1)

            return predicted

        # Define the bounds for the streamplot:
        grid_bounds = ([np.min(PCs[:,0]),np.max(PCs[:,0])],[np.min(PCs[:,1]),np.max(PCs[:,1])])

        # Plot the regression streamplot:
        plt = plot_2d_regression_streamplot(grid_bounds,
                                            regression_model,
                                            x=PCs[:,0],
                                            y=PCs[:,1],
                                            resolution=(15,15),
                                            extension=(20,20),
                                            color='r',
                                            x_label='$Z_1$',
                                            y_label='$Z_2$',
                                            manifold_color=X[:,0],
                                            colorbar_label='$X_1$',
                                            color_map='plasma',
                                            colorbar_range=(0,1),
                                            manifold_alpha=1,
                                            grid_on=False,
                                            figure_size=(10,6),
                                            title='Streamplot',
                                            save_filename='streamplot.pdf')
        plt.close()

    :param grid_bounds:
        ``tuple`` of ``list`` specifying the bounds of the dependent variable on the :math:`x` and :math:`y` axis.
    :param regression_model:
        ``function`` that outputs the predicted vector using the regression model.
        It should take as input a ``numpy.ndarray`` of size ``(1,2)``, where the two
        elements specify the first and second independent variable values. It should output
        a ``numpy.ndarray`` of size ``(1,2)``, where the two elements specify
        the first and second regressed vector elements.
    :param x: (optional)
        ``numpy.ndarray`` specifying the variable on the :math:`x`-axis.
        It should be of size ``(n_observations,)`` or ``(n_observations,1)``.
        It can be used to plot a 2D manifold on top of the streamplot.
    :param y: (optional)
        ``numpy.ndarray`` specifying the variable on the :math:`y`-axis.
        It should be of size ``(n_observations,)`` or ``(n_observations,1)``.
        It can be used to plot a 2D manifold on top of the streamplot.
    :param resolution: (optional)
        ``tuple`` of ``int`` specifying the resolution of the streamplot grid on the :math:`x` and :math:`y` axis.
    :param extension: (optional)
        ``tuple`` of ``float`` or ``int`` specifying a percentage by which the
        dependent variable should be extended beyond on the :math:`x` and :math:`y` axis, beyond what has been specified by the ``grid_bounds`` parameter.
    :param color: (optional)
        ``str`` specifying the streamlines color.
    :param x_label: (optional)
        ``str`` specifying :math:`x`-axis label annotation. If set to ``None``
        label will not be plotted.
    :param y_label: (optional)
        ``str`` specifying :math:`y`-axis label annotation. If set to ``None``
        label will not be plotted.
    :param s_manifold: (optional)
        ``int`` or ``float`` specifying the scatter point size for the manifold.
    :param manifold_color: (optional)
        vector or string specifying color for the manifold. If it is a
        vector, it has to have length consistent with the number of observations
        in ``x`` and ``y`` vectors. It should be of type ``numpy.ndarray`` and size
        ``(n_observations,)`` or ``(n_observations,1)``.
        It can also be set to a string specifying the color directly, for
        instance ``'r'`` or ``'#006778'``.
        If not specified, manifold will be plotted in black.
    :param colorbar_label: (optional)
        ``str`` specifying colorbar label annotation.
    :param color_map: (optional)
        ``str`` or ``matplotlib.colors.ListedColormap`` specifying the colormap to use as per ``matplotlib.cm``. Default is ``'viridis'``.
    :param colorbar_range: (optional)
        ``tuple`` specifying the lower and the upper bound for the colorbar range.
    :param manifold_alpha: (optional)
        ``float`` or ``int`` specifying the opacity of the plotted manifold.
    :param grid_on:
        ``bool`` specifying whether grid should be plotted.
    :param figure_size: (optional)
        ``tuple`` specifying figure size.
    :param title: (optional)
        ``str`` specifying plot title. If set to ``None`` title will not be
        plotted.
    :param save_filename: (optional)
        ``str`` specifying plot save location/filename. If set to ``None``
        plot will not be saved. You can also set a desired file extension,
        for instance ``.pdf``. If the file extension is not specified, the default
        is ``.png``.

    :return:
        - **plt** - ``matplotlib.pyplot`` plot handle.
    """

    if not isinstance(grid_bounds, tuple):
        raise ValueError("Parameter `grid_bounds` has to be of type `tuple`.")

    if not callable(regression_model):
        raise ValueError("Parameter `regression_model` has to be of type `function`.")

    if x is not None:

        if not isinstance(x, np.ndarray):
            raise ValueError("Parameter `x` has to be of type `numpy.ndarray`.")

        try:
            (n_x,) = np.shape(x)
            n_var_x = 1
        except:
            (n_x, n_var_x) = np.shape(x)

        if n_var_x != 1:
            raise ValueError("Parameter `x` has to be a 0D or 1D vector.")

    if y is not None:

        if not isinstance(y, np.ndarray):
            raise ValueError("Parameter `y` has to be of type `numpy.ndarray`.")

        try:
            (n_y,) = np.shape(y)
            n_var_y = 1
        except:
            (n_y, n_var_y) = np.shape(y)

        if n_var_y != 1:
            raise ValueError("Parameter `y` has to be a 0D or 1D vector.")

        if n_x != n_y:
            raise ValueError("Parameter `x` has different number of elements than `y`.")

    if not isinstance(resolution, tuple):
        raise ValueError("Parameter `resolution` has to be of type `tuple`.")

    if not isinstance(extension, tuple):
        raise ValueError("Parameter `extension` has to be of type `tuple`.")

    if color is not None:
        if not isinstance(color, str):
            raise ValueError("Parameter `color` has to be of type `str`.")

    if x_label is not None:
        if not isinstance(x_label, str):
            raise ValueError("Parameter `x_label` has to be of type `str`.")

    if y_label is not None:
        if not isinstance(y_label, str):
            raise ValueError("Parameter `y_label` has to be of type `str`.")

    if s_manifold is None:
        s_manifold = scatter_point_size
    else:
        if not isinstance(s_manifold, int) and not isinstance(s_manifold, float):
            raise ValueError("Parameter `s_manifold` has to be of type `int` or `float`.")

    if manifold_color is not None:
        if not isinstance(manifold_color, str):
            if not isinstance(manifold_color, np.ndarray):
                raise ValueError("Parameter `manifold_color` has to be `None`, or of type `str` or `numpy.ndarray`.")

    if isinstance(manifold_color, np.ndarray):

        try:
            (n_color,) = np.shape(manifold_color)
            n_var_color = 1
        except:
            (n_color, n_var_color) = np.shape(manifold_color)

        if n_var_color != 1:
            raise ValueError("Parameter `manifold_color` has to be a 0D or 1D vector.")

        if n_color != n_x:
            raise ValueError("Parameter `manifold_color` has different number of elements than `x` and `y`.")

    if colorbar_label is not None:
        if not isinstance(colorbar_label, str):
            raise ValueError("Parameter `colorbar_label` has to be of type `str`.")

    if not isinstance(color_map, str):
        if not isinstance(color_map, ListedColormap):
            raise ValueError("Parameter `color_map` has to be of type `str` or `matplotlib.colors.ListedColormap`.")

    if colorbar_range is not None:
        if not isinstance(colorbar_range, tuple):
            raise ValueError("Parameter `colorbar_range` has to be of type `tuple`.")
        else:
            (cbar_min, cbar_max) = colorbar_range

    if manifold_alpha is not None:
        if not isinstance(manifold_alpha, float) and not isinstance(manifold_alpha, int):
            raise ValueError("Parameter `manifold_alpha` has to be of type `float`.")

    if not isinstance(grid_on, bool):
        raise ValueError("Parameter `grid_on` has to be of type `bool`.")

    if not isinstance(figure_size, tuple):
        raise ValueError("Parameter `figure_size` has to be of type `tuple`.")

    if title is not None:
        if not isinstance(title, str):
            raise ValueError("Parameter `title` has to be of type `str`.")

    if save_filename is not None:
        if not isinstance(save_filename, str):
            raise ValueError("Parameter `save_filename` has to be of type `str`.")

    (x_extend, y_extend) = extension
    (x_resolution, y_resolution) = resolution
    ([x_minimum, x_maximum], [y_minimum, y_maximum]) = grid_bounds

    # Create extension in both dimensions:
    x_extension = x_extend/100.0 * abs(x_minimum - x_maximum)
    y_extension = y_extend/100.0 * abs(y_minimum - y_maximum)

    # Create grid points for the independent variables where regression will be applied:
    x_grid = np.linspace(x_minimum-x_extension, x_maximum+x_extension, x_resolution)
    y_grid = np.linspace(y_minimum-y_extension, y_maximum+y_extension, y_resolution)
    xy_mesh = np.meshgrid(x_grid, y_grid, indexing='xy')

    # Evaluate the predicted vectors using the regression model:
    x_vector = np.zeros((y_grid.size, x_grid.size))
    y_vector = np.zeros((y_grid.size, x_grid.size))

    for j, x_variable in enumerate(x_grid):
        for i, y_variable in enumerate(y_grid):

                regression_input = np.reshape(np.array([x_variable, y_variable]), [1,2])

                regressed_vector = regression_model(regression_input)

                x_vector[i,j] = regressed_vector[0,0]
                y_vector[i,j] = regressed_vector[0,1]

    fig = plt.figure(figsize=figure_size)

    if (x is not None) and (y is not None):

        if manifold_color is None:
            scat = plt.scatter(x.ravel(), y.ravel(), c='k', marker='o', s=s_manifold, edgecolor='none', alpha=manifold_alpha)
        elif isinstance(manifold_color, str):
            scat = plt.scatter(x.ravel(), y.ravel(), c=manifold_color, cmap=color_map, marker='o', s=s_manifold, edgecolor='none', alpha=manifold_alpha)
        elif isinstance(manifold_color, np.ndarray):
            scat = plt.scatter(x.ravel(), y.ravel(), c=manifold_color.ravel(), cmap=color_map, marker='o', s=s_manifold, edgecolor='none', alpha=manifold_alpha)

    if isinstance(manifold_color, np.ndarray):
        if manifold_color is not None:
            cb = fig.colorbar(scat)
            cb.ax.tick_params(labelsize=font_colorbar_axes)
            if colorbar_label is not None: cb.set_label(colorbar_label, fontsize=font_colorbar, rotation=0, horizontalalignment='left')
            if colorbar_range is not None: plt.clim(cbar_min, cbar_max)

    plt.streamplot(xy_mesh[0], xy_mesh[1], x_vector, y_vector, color=color, density=3, linewidth=1, arrowsize=1)

    if x_label is not None: plt.xlabel(x_label, **csfont, fontsize=font_labels)
    if y_label is not None: plt.ylabel(y_label, **csfont, fontsize=font_labels)
    plt.xlim([np.min(x_grid),np.max(x_grid)])
    plt.ylim([np.min(y_grid),np.max(y_grid)])
    plt.xticks(fontsize=font_axes, **csfont)
    plt.yticks(fontsize=font_axes, **csfont)
    if grid_on: plt.grid(alpha=grid_opacity)

    if title is not None: plt.title(title, **csfont, fontsize=font_title)
    if save_filename is not None: plt.savefig(save_filename, dpi=save_dpi, bbox_inches='tight')

    return plt

# ------------------------------------------------------------------------------

def plot_3d_regression(x, y, observed, predicted, elev=45, azim=-45, clean=False, x_label=None, y_label=None, z_label=None, color_observed=None, color_predicted=None, s_observed=None, s_predicted=None, alpha_observed=None, alpha_predicted=None, figure_size=(7,7), title=None, save_filename=None):
    """
    Plots the result of regression of a dependent variable on top
    of a two-dimensional manifold defined by two independent variables ``x`` and ``y``.

    **Example:**

    .. code:: python

        from PCAfold import PCA, plot_3d_regression
        import numpy as np

        # Generate dummy data set:
        X = np.random.rand(100,10)

        # Obtain three-dimensional manifold from PCA:
        pca_X = PCA(X)
        PCs = pca_X.transform(X)
        X_rec = pca_X.reconstruct(PCs)

        # Plot the manifold:
        plt = plot_3d_regression(X[:,0],
                                 X[:,1],
                                 X[:,0],
                                 X_rec[:,0],
                                 elev=45,
                                 azim=-45,
                                 x_label='$x$',
                                 y_label='$y$',
                                 z_label='$z$',
                                 color_observed='k',
                                 color_predicted='r',
                                 figure_size=(10,10),
                                 title='3D regression',
                                 save_filename='3d-regression.pdf')
        plt.close()

    :param x:
        ``numpy.ndarray`` specifying the variable on the :math:`x`-axis. It should be of size ``(n_observations,)`` or ``(n_observations,1)``.
    :param y:
        ``numpy.ndarray`` specifying the variable on the :math:`y`-axis. It should be of size ``(n_observations,)`` or ``(n_observations,1)``.
    :param observed:
        ``numpy.ndarray`` specifying the observed values of a single dependent variable.
        It should be of size ``(n_observations,)`` or ``(n_observations, 1)``.
    :param predicted:
        ``numpy.ndarray`` specifying the predicted values of a single dependent variable.
        It should be of size ``(n_observations,)`` or ``(n_observations, 1)``.
    :param elev: (optional)
        ``float`` or ``int`` specifying the elevation angle.
    :param azim: (optional)
        ``float`` or ``int`` specifying the azimuth angle.
    :param clean: (optional)
        ``bool`` specifying if a clean plot should be made. If set to ``True``, nothing else but the data points and the 3D axes is plotted.
    :param x_label: (optional)
        ``str`` specifying :math:`x`-axis label annotation. If set to ``None``
        label will not be plotted.
    :param y_label: (optional)
        ``str`` specifying :math:`y`-axis label annotation. If set to ``None``
        label will not be plotted.
    :param z_label: (optional)
        ``str`` specifying :math:`z`-axis label annotation. If set to ``None``
        label will not be plotted.
    :param color_observed: (optional)
        ``str`` specifying the color of the plotted observed variable.
    :param color_predicted: (optional)
        ``str`` specifying the color of the plotted predicted variable.
    :param s_observed: (optional)
        ``int`` or ``float`` specifying the scatter point size for the observed variable.
    :param s_predicted: (optional)
        ``int`` or ``float`` specifying the scatter point size for the predicted variable.
    :param alpha_observed: (optional)
        ``int`` or ``float`` specifying the point opacity for the observed variable.
    :param alpha_predicted: (optional)
        ``int`` or ``float`` specifying the point opacity for the predicted variable.
    :param figure_size: (optional)
        ``tuple`` specifying figure size.
    :param title: (optional)
        ``str`` specifying plot title. If set to ``None`` title will not be
        plotted.
    :param save_filename: (optional)
        ``str`` specifying plot save location/filename. If set to ``None``
        plot will not be saved. You can also set a desired file extension,
        for instance ``.pdf``. If the file extension is not specified, the default
        is ``.png``.

    :return:
        - **plt** - ``matplotlib.pyplot`` plot handle.
    """

    from mpl_toolkits.mplot3d import Axes3D

    if not isinstance(x, np.ndarray):
        raise ValueError("Parameter `x` has to be of type `numpy.ndarray`.")

    try:
        (n_x,) = np.shape(x)
        n_var_x = 1
    except:
        (n_x, n_var_x) = np.shape(x)

    if n_var_x != 1:
        raise ValueError("Parameter `x` has to be a 0D or 1D vector.")

    if not isinstance(y, np.ndarray):
        raise ValueError("Parameter `y` has to be of type `numpy.ndarray`.")

    try:
        (n_y,) = np.shape(y)
        n_var_y = 1
    except:
        (n_y, n_var_y) = np.shape(y)

    if n_var_y != 1:
        raise ValueError("Parameter `y` has to be a 0D or 1D vector.")

    if not isinstance(observed, np.ndarray):
        raise ValueError("Parameter `observed` has to be of type `numpy.ndarray`.")

    try:
        (n_observed,) = np.shape(observed)
        n_var_observed = 1
    except:
        (n_observed, n_var_observed) = np.shape(observed)

    if n_var_observed != 1:
        raise ValueError("Parameter `observed` has to be a 0D or 1D vector.")

    if not isinstance(predicted, np.ndarray):
        raise ValueError("Parameter `predicted` has to be of type `numpy.ndarray`.")

    try:
        (n_predicted,) = np.shape(predicted)
        n_var_predicted = 1
    except:
        (n_predicted, n_var_predicted) = np.shape(predicted)

    if n_var_predicted != 1:
        raise ValueError("Parameter `predicted` has to be a 0D or 1D vector.")

    if n_observed != n_predicted:
        raise ValueError("Parameter `observed` has different number of elements than `predicted`.")

    if n_x != n_observed:
        raise ValueError("Parameter `observed` has different number of elements than `x`, `y` and `z`.")

    if n_x != n_predicted:
        raise ValueError("Parameter `predicted` has different number of elements than `x`, `y` and `z`.")

    if not isinstance(elev, float) and not isinstance(elev, int):
        raise ValueError("Parameter `elev` has to be of type `int` or `float`.")

    if not isinstance(azim, float) and not isinstance(azim, int):
        raise ValueError("Parameter `azim` has to be of type `int` or `float`.")

    if not isinstance(clean, bool):
        raise ValueError("Parameter `clean` has to be of type `bool`.")

    if x_label is not None:
        if not isinstance(x_label, str):
            raise ValueError("Parameter `x_label` has to be of type `str`.")

    if y_label is not None:
        if not isinstance(y_label, str):
            raise ValueError("Parameter `y_label` has to be of type `str`.")

    if z_label is not None:
        if not isinstance(z_label, str):
            raise ValueError("Parameter `z_label` has to be of type `str`.")

    if color_observed is not None:
        if not isinstance(color_observed, str):
            raise ValueError("Parameter `color_observed` has to be of type `str`.")
    else:
        color_observed = '#191b27'

    if color_predicted is not None:
        if not isinstance(color_predicted, str):
            raise ValueError("Parameter `color_predicted` has to be of type `str`.")
    else:
        color_predicted = '#C7254E'

    if s_observed is None:
        s_observed = scatter_point_size
    else:
        if not isinstance(s_observed, int) and not isinstance(s_observed, float):
            raise ValueError("Parameter `s_observed` has to be of type `int` or `float`.")

    if s_predicted is None:
        s_predicted = scatter_point_size
    else:
        if not isinstance(s_predicted, int) and not isinstance(s_predicted, float):
            raise ValueError("Parameter `s_predicted` has to be of type `int` or `float`.")

    if alpha_observed is None:
        alpha_observed = 0.1
    else:
        if not isinstance(alpha_observed, int) and not isinstance(alpha_observed, float):
            raise ValueError("Parameter `alpha_observed` has to be of type `int` or `float`.")

    if alpha_predicted is None:
        alpha_predicted = 0.4
    else:
        if not isinstance(alpha_predicted, int) and not isinstance(alpha_predicted, float):
            raise ValueError("Parameter `alpha_predicted` has to be of type `int` or `float`.")

    if not isinstance(figure_size, tuple):
        raise ValueError("Parameter `figure_size` has to be of type `tuple`.")

    if title is not None:
        if not isinstance(title, str):
            raise ValueError("Parameter `title` has to be of type `str`.")

    if save_filename is not None:
        if not isinstance(save_filename, str):
            raise ValueError("Parameter `save_filename` has to be of type `str`.")

    fig = plt.figure(figsize=figure_size)
    ax = fig.add_subplot(111, projection='3d')

    scat = ax.scatter(x.ravel(), y.ravel(), observed.ravel(), c=color_observed, marker='o', s=s_observed, alpha=alpha_observed)
    scat = ax.scatter(x.ravel(), y.ravel(), predicted.ravel(), c=color_predicted, marker='o', s=s_predicted, alpha=alpha_predicted)

    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('w')
    ax.yaxis.pane.set_edgecolor('w')
    ax.zaxis.pane.set_edgecolor('w')
    ax.view_init(elev=elev, azim=azim)

    if clean:

        plt.xticks([])
        plt.yticks([])
        ax.set_zticks([])
        ax.spines["top"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)

    else:

        if x_label != None: ax.set_xlabel(x_label, **csfont, fontsize=font_labels, rotation=0, labelpad=20)
        if y_label != None: ax.set_ylabel(y_label, **csfont, fontsize=font_labels, rotation=0, labelpad=20)
        if z_label != None: ax.set_zlabel(z_label, **csfont, fontsize=font_labels, rotation=0, labelpad=20)

        ax.tick_params(pad=5)
        ax.grid(alpha=grid_opacity)

        for label in (ax.get_xticklabels()):
            label.set_fontsize(font_axes)
        for label in (ax.get_yticklabels()):
            label.set_fontsize(font_axes)
        for label in (ax.get_zticklabels()):
            label.set_fontsize(font_axes)

    lgnd = plt.legend(['Observed', 'Predicted'], fontsize=font_legend, bbox_to_anchor=(0.9,0.9), loc="upper left")
    lgnd.legendHandles[0]._sizes = [marker_size*5]
    lgnd.legendHandles[1]._sizes = [marker_size*5]

    if title != None: ax.set_title(title, **csfont, fontsize=font_title)
    if save_filename != None: plt.savefig(save_filename, dpi=save_dpi, bbox_inches='tight')

    return plt

# ------------------------------------------------------------------------------

def plot_stratified_metric(metric_in_bins, bins_borders, variable_name=None, metric_name=None, yscale='linear', ylim=None, figure_size=(10,5), title=None, save_filename=None):
    """
    This function plots a stratified metric across bins of a dependent variable.

    **Example:**

    .. code:: python

        from PCAfold import PCA, variable_bins, stratified_coefficient_of_determination, plot_stratified_metric
        import numpy as np

        # Generate dummy data set:
        X = np.random.rand(100,10)

        # Instantiate PCA class object:
        pca_X = PCA(X, scaling='auto', n_components=2)

        # Approximate the data set:
        X_rec = pca_X.reconstruct(pca_X.transform(X))

        # Generate bins:
        (idx, bins_borders) = variable_bins(X[:,0], k=10, verbose=False)

        # Compute stratified R2 in 10 bins of the first variable in a data set:
        r2_in_bins = stratified_coefficient_of_determination(X[:,0], X_rec[:,0], idx=idx, use_global_mean=True, verbose=True)

        # Visualize how R2 changes across bins:
        plt = plot_stratified_metric(r2_in_bins,
                                      bins_borders,
                                      variable_name='$X_1$',
                                      metric_name='$R^2$',
                                      yscale='log',
                                      figure_size=(10,5),
                                      title='Stratified $R^2$',
                                      save_filename='r2.pdf')
        plt.close()

    :param metric_in_bins:
        ``list`` of metric values in each bin.
    :param bins_borders:
        ``list`` of bins borders that were created to stratify the dependent variable.
    :param variable_name: (optional)
        ``str`` specifying the name of the variable for which the metric was computed. If set to ``None``
        label on the x-axis will not be plotted.
    :param metric_name: (optional)
        ``str`` specifying the name of the metric to be plotted on the y-axis. If set to ``None``
        label on the x-axis will not be plotted.
    :param yscale: (optional)
        ``str`` specifying the scale for the y-axis.
    :param figure_size: (optional)
        ``tuple`` specifying figure size.
    :param title: (optional)
        ``str`` specifying plot title. If set to ``None`` title will not be
        plotted.
    :param save_filename: (optional)
        ``str`` specifying plot save location/filename. If set to ``None``
        plot will not be saved. You can also set a desired file extension,
        for instance ``.pdf``. If the file extension is not specified, the default
        is ``.png``.

    :return:
        - **plt** - ``matplotlib.pyplot`` plot handle.
    """

    if not isinstance(figure_size, tuple):
        raise ValueError("Parameter `figure_size` has to be of type `tuple`.")

    if title is not None:
        if not isinstance(title, str):
            raise ValueError("Parameter `title` has to be of type `str`.")

    if save_filename is not None:
        if not isinstance(save_filename, str):
            raise ValueError("Parameter `save_filename` has to be of type `str`.")

    bin_centers = []
    for i in range(0,len(bins_borders)-1):
        bin_length = bins_borders[i+1] - bins_borders[i]
        bin_centers.append(bins_borders[i] + bin_length/2)

    figure = plt.figure(figsize=figure_size)
    plt.scatter(bin_centers, metric_in_bins, c='#191b27')
    plt.grid(alpha=grid_opacity)
    plt.xlim([bins_borders[0], bins_borders[-1]])
    plt.yscale(yscale)
    if ylim is not None: plt.ylim(ylim)
    if variable_name is not None: plt.xlabel(variable_name, **csfont, fontsize=font_labels)
    if metric_name is not None: plt.ylabel(metric_name, **csfont, fontsize=font_labels)

    if title is not None: plt.title(title, fontsize=font_title, **csfont)
    if save_filename is not None: plt.savefig(save_filename, dpi=save_dpi, bbox_inches='tight')

    return plt
