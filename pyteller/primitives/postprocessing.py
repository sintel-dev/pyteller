import numpy as np
import pandas as pd


def flatten(X,index, freq, pred_length):
    """flattens predictions and averages duplicate predicted values

    The function takes in a an array and averages the predictions that are for the same timestep

    Args:
        X (ndarray):
            N-dimensional sequence to iterate over.
        pred_length (int):
            Length of the target sequence
        index (ndarray):
            Array containing the index values of X.
        columns (list):
            List with the names of the columns of the original dataset
        freq (int):
            Length of the target sequences.


    Returns:
        ndarray, ndarray, ndarray, ndarray:
            * input sequences.
            * target sequences.
            * first index value of each input sequence.
            * first index value of each target sequence.
    """

    index = np.reshape(index, [index.size, 1])
    index = np.insert(index, 1, np.full((pred_length - 1, 1), freq), axis=1)
    index = np.cumsum(index, axis=1).flatten()
    if X.ndim == 3:
        df = pd.DataFrame(data=X.reshape(-1, X.shape[-1]), index=index)
    else:
        df = pd.DataFrame(data=X.flatten(), index=index)
    df = df.groupby(df.index).mean()

    return df


def reformat_data(X, _X):
    forecast = X
    test = _X
    if forecast.index.dtype == 'float' or forecast.index.dtype == 'int':
        forecast.index = pd.to_datetime(forecast.index.values * 1e9)
    else:
        forecast.index = pd.to_datetime(forecast.index)

    if test['timestamp'].dtypes == 'float' or test['timestamp'].dtypes == 'int':
        test['timestamp'] = pd.to_datetime(test['timestamp'] * 1e9)
    else:
        test['timestamp'] = pd.to_datetime(test['timestamp'])

    actual = test.set_index('timestamp')
    actual = actual[actual.index.isin(forecast.index)]
    forecast.columns = actual.columns

    return actual, forecast, _X
