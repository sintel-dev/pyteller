import numpy as np
import pandas as pd


def flatten(X, index, pred_length, freq, type='average'):
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

    if index.dtype == int:  # if the index was made with make_index=True
        freq = 1
    else:
        index = pd.to_datetime(index).astype(np.int64) // 1e9

    if type == 'horizon':
        X = X[-1]
        index = index[-1]

    index = np.reshape(index, [index.size, 1])
    index = np.insert(index, 1, np.full((pred_length - 1, 1), freq, dtype=float), axis=1)
    index = np.cumsum(index, axis=1).flatten()
    df = pd.DataFrame(data=X.flatten(), index=index)
    df = df.groupby(df.index).mean()
    if index.dtype != int:
        df.index = pd.to_datetime(df.index.values * 1e9)
    X = df.values
    index = df.index
    return X, index


def reformat_data(X, index, actuals, time_column,targets):
    # convert index to datetime
    # if index.dtype == 'float' or index.dtype == 'int':
    #     index = pd.to_datetime(index.values * 1e9)
    # else:
    #     index = pd.to_datetime(index)
    #
    # if actuals[time_column].dtypes == 'float' or actuals[time_column].dtypes == 'int':
    #     actuals[time_column] = pd.to_datetime(actuals[time_column] * 1e9)
    # else:
    #     actuals[time_column] = pd.to_datetime(actuals[time_column])

    actuals = actuals.set_index(time_column.lower())
    forecasts = pd.DataFrame(data=X, index=index)
    actuals = actuals[actuals.index.isin(forecasts.index)]
    forecasts.columns =[targets]

    return actuals[[targets]], forecasts
