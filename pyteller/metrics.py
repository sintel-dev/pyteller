import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


def root_mean_square_error(testing_series, prediction_series):
    """

    Computes the Root Mean  Square Error for univariate time series
    prediction.

    Args:
        y_true (numpy.ndarray):
            ``numpy.ndarray`` of the known values for the given predictions
        y_pred: (numpy.ndarray):
            ``numpy.ndarray`` of the generated predictions

    Returns:
        float of RMSE
    """

    return mean_squared_error(testing_series, prediction_series, squared=False)


def _naive_forecasting(actual, seasonality=1):
    """ Naive forecasting method which just repeats previous samples """
    return actual[:-seasonality]


def MASE(actual: np.ndarray, predicted: np.ndarray, seasonality: int = 1):
    """

    Computes the Mean Absolute Scaled Error for univariate time series
    prediction.

    Args:
        y_true (numpy.ndarray):
            ``numpy.ndarray`` of the known values for the given predictions
        y_pred: (numpy.ndarray):
            ``numpy.ndarray`` of the generated predictions

    Returns:
        float of MASE
    """

    """
    Mean Absolute Scaled Error
    Baseline (benchmark) is computed with naive forecasting (shifted by @seasonality)
    """
    return mean_absolute_error(actual, predicted) / mean_absolute_error(actual[seasonality:],
                                                                        _naive_forecasting(actual, seasonality))


def sMAPE(testing_series, prediction_series):
    """

    Computes the Symmetric mean absolute percentage error for univariate time series
    prediction.

    Args:
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
    """

    Computes the  mean absolute percentage error for univariate time series
    prediction.

    Args:
        y_true (numpy.ndarray):
            ``numpy.ndarray`` of the known values for the given predictions
        y_pred: (numpy.ndarray):
            ``numpy.ndarray`` of the generated predictions

    Returns:
        float of MAPE
    """

    undefined = prediction_series != testing_series
    prediction_series = np.array(prediction_series[undefined], dtype=float)
    testing_series = np.array(testing_series[undefined], dtype=float)
    return 1 / len(testing_series) * \
        np.sum(abs((testing_series - prediction_series) / testing_series)) * 100


def under_pred(y, y_hat):
    """

    Computes the summation of where the forecast was less than the actual values

    Args:
        y_true (numpy.ndarray):
            ``numpy.ndarray`` of the known values for the given predictions
        y_pred: (numpy.ndarray):
            ``numpy.ndarray`` of the generated predictions

    Returns:
        float of the sum of the underpredicted values
    """

    return np.sum(np.where(y >= y_hat, y - y_hat, 0))


def over_pred(y, y_hat):
    """

    Computes the summation of where the forecast was greater than the actual values

    Args:
        y_true (numpy.ndarray):
            ``numpy.ndarray`` of the known values for the given predictions
        y_pred: (numpy.ndarray):
            ``numpy.ndarray`` of the generated predictions

    Returns:
        float of the sum of the overpredicted values
    """

    return np.sum(np.where(y <= y_hat, y_hat - y, 0))


METRICS = {
    'MAE': mean_absolute_error,
    'MSE': mean_squared_error,
    'RMSE': root_mean_square_error,
    'MASE': MASE,
    'sMAPE': sMAPE,
    'MAPE': MAPE,
    'under': under_pred,
    'over': over_pred

}
