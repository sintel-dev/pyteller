import numpy as np
from sklearn.metrics import mean_squared_error,mean_absolute_error
# TODO break up metrics between normalized and not normalized


def MASE(training_series, prediction_series, testing_series):
    """
    # Source: https://github.com/CamDavidsonPilon/Python-Numerics/blob/master/TimeSeries/MASE.py
    Computes Mean absolute scaled error forcast error for univariate time series prediction.

    See "Another look at measures of forecast accuracy", Rob J Hyndman

    Args:
        training_series: the series used to train the model, 1d numpy array
        testing_series: the test series to predict, 1d numpy array or float
        prediction_series: the prediction of testing_series, 1d numpy array (same size as testing_series) or float
        absolute: "squares" to use sum of squares and root the result, "absolute" to use absolute values.
    Returns: float of MASE
    """
    n = training_series.shape[0]
    d = np.abs(np.diff(np.array(training_series,dtype=float))).sum() / (n - 1)

    errors = np.abs(np.array(testing_series,dtype=float) - np.array(prediction_series,dtype=float)).mean()
    if np.isnan(d) or errors == 0 or d == 0:
        return 0
    else:
        return errors.mean() / d


def sMAPE(prediction_series, testing_series):
    """

    Computes the Symmetric mean absolute percentage error forcast error for univariate time series prediction.

    Args:
        training_series: the series used to train the model, 1d numpy array
        testing_series: the test series to predict, 1d numpy array or float
        prediction_series: the prediction of testing_series, 1d numpy array (same size as testing_series) or float
        absolute: "squares" to use sum of squares and root the result, "absolute" to use absolute values.
    Returns: float of sMAPE
    """
    prediction_series = np.array(prediction_series,dtype=float)
    testing_series = np.array(testing_series,dtype=float)
    zero = prediction_series[testing_series == 0]  # Check if two time steps are both zero
    if zero.size != 0:  # If there are indexes where the zeros align return zero, there is no error
        return 0
    else:
        return 100 / len(testing_series) * np.sum(2 * np.abs(prediction_series
                                                             - testing_series) / (np.abs(testing_series) + np.abs(prediction_series)))
