import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


def root_mean_square_error(testing_series, prediction_series):
    return mean_squared_error(testing_series, prediction_series, squared=False)


def MASE(training_series, y_true, y_pred):
    """
    # Source: https://github.com/CamDavidsonPilon/Python-Numerics/blob/master/TimeSeries/MASE.py
    Computes Mean absolute scaled error forecast error

    See "Another look at measures of forecast accuracy", Rob J Hyndman

    Args:
        training_series (numpy.ndarray):
            the series used to train the model
        y_true (numpy.ndarray):
            ``numpy.ndarray`` of the known values for the given predictions
        y_pred: (numpy.ndarray):
            ``numpy.ndarray`` of the generated predictions

    Returns:
        float of MASE
    """
    n = training_series.shape[0]
    d = np.abs(np.diff(np.array(training_series, dtype=float))).sum() / (n - 1)

    errors = np.abs(
        np.array(y_true, dtype=float) - np.array(y_pred, dtype=float)).mean()
    if np.isnan(d) or errors == 0 or d == 0:
        return 0
    else:
        return errors.mean() / d


def sMAPE(testing_series, prediction_series):
    """

    Computes the Symmetric mean absolute percentage error forcast error for univariate time series
    prediction.

    Args:
        training_series (numpy.ndarray):
            the series used to train the model
        y_true (numpy.ndarray):
            ``numpy.ndarray`` of the known values for the given predictions
        y_pred: (numpy.ndarray):
            ``numpy.ndarray`` of the generated predictions

    Returns:
        float of sMAPE
    """
    undefined = prediction_series != testing_series
    prediction_series = np.array(prediction_series[undefined], dtype=float)
    testing_series = np.array(testing_series[undefined], dtype=float)

    return 100 / len(testing_series) \
        * np.sum(2 * np.abs(prediction_series - testing_series)
                 / (np.abs(testing_series) + np.abs(prediction_series)))


def MAPE(testing_series, prediction_series):
    undefined = prediction_series != testing_series
    prediction_series = np.array(prediction_series[undefined], dtype=float)
    testing_series = np.array(testing_series[undefined], dtype=float)
    return 1 / len(testing_series) * \
        np.sum(abs((testing_series - prediction_series) / testing_series)) * 100


def under_pred(y, y_hat):
    return np.sum(np.where(y >= y_hat, y - y_hat, 0))


def over_pred(y, y_hat):
    return np.sum(np.where(y <= y_hat, y_hat - y, 0))


METRICS = {
    'MAE': mean_absolute_error,
    'MSE': mean_squared_error,
    'RMSE': root_mean_square_error,
    'MASE': MASE,
    'sMAPE': sMAPE,
    'MAPE': MAPE,

}
