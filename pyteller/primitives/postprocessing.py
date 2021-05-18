import numpy as np
import pandas as pd

def flatten(X, index, pred_length):
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

    Returns:
        ndarray, ndarray, ndarray, ndarray:
            * input sequences.
            * target sequences.
            * first index value of each input sequence.
            * first index value of each target sequence.
    """
    # freq = index[1] - index[0]
    index_epoch = pd.to_datetime(index).astype(np.int64) // 1e9
    freq = (index_epoch[1] - index_epoch[0])/pred_length
    index = np.reshape(index_epoch, [index_epoch.size, 1])
    index = np.insert(index, 1, np.full((pred_length - 1, 1), freq), axis=1)
    index = np.cumsum(index, axis=1).flatten()
    df = pd.DataFrame(data=X.flatten(), index=index)
    df = df.groupby(df.index).mean()
    df.index = pd.to_datetime(df.index.values * 1e9)
    return df.values, df.index


def flatten2(X, index, freq, pred_length):
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
    freq = df['timestamp'].iloc[1] - df['timestamp'].iloc[0]
    index = np.reshape(index, [index.size, 1])
    index = np.insert(index, 1, np.full((pred_length - 1, 1), freq), axis=1)
    index = np.cumsum(index, axis=1).flatten()
    if X.ndim == 3:
        df = pd.DataFrame(data=X.reshape(-1, X.shape[-1]), index=index)
    else:
        df = pd.DataFrame(data=X.flatten(), index=index)
    df = df.groupby(df.index).mean()

    return df


def reformat_data(X, index, actuals, time_column):
    #convert index to datetime
    # if index.dtype == 'float' or index.dtype == 'int':
    #     index = pd.to_datetime(index.values * 1e9)
    # else:
    #     index = pd.to_datetime(index)
    #
    # if actuals[time_column].dtypes == 'float' or actuals[time_column].dtypes == 'int':
    #     actuals[time_column] = pd.to_datetime(actuals[time_column] * 1e9)
    # else:
    #     actuals[time_column] = pd.to_datetime(actuals[time_column])

    actuals = actuals.set_index(time_column)
    forecasts = pd.DataFrame(data=X, index=index)
    actuals = actuals[actuals.index.isin(forecasts.index)]
    forecasts.columns = actuals.columns

    return actuals, forecasts


def reformat_data2(X, index, actuals):
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

    actuals = test.set_index('timestamp')
    actuals = actuals[actuals.index.isin(forecast.index)]
    forecast.columns = actuals.columns

    return actuals, forecast, _X
