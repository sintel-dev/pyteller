import numpy as np
import pandas as pd


def get_index(X, time_column='timestamp'):
    """Stores the index of an input time series in the context
    Args:
        X (pandas.DataFrame):
            N-dimensional sequence of values.
        time_column (int):
            Column of X that contains time values.

    Returns:
        ndarray, ndarray:
            * Input sequence
            * Index of input sequence
    """

    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X)

    X = X.sort_values(time_column).set_index(time_column)

    return np.asarray(X.values), np.asarray(X.index)


def rolling_window_sequences(X, index, window_size, target_size, step_size, target_column=None, offset=0,
                             drop=None, drop_windows=False):
    """Create rolling window sequences out of time series data.

    The function creates an array of input sequences and an array of target sequences by rolling
    over the input sequence with a specified window.
    Optionally, certain values can be dropped from the sequences.

    Args:
        X (ndarray):
            N-dimensional sequence to iterate over.
        index (ndarray):
            Array containing the index values of X.
        window_size (int):
            Length of the input sequences.
        target_size (int):
            Length of the target sequences.
        step_size (int):
            Indicating the number of steps to move the window forward each round.
        target_column (int):
            Indicating which column of X is the target.
        offset (int):
            Indicating the number of steps between the input and the target sequence.
        drop (ndarray or None or str or float or bool):
            Optional. Array of boolean values indicating which values of X are invalid, or value
            indicating which value should be dropped. If not given, `None` is used.
        drop_windows (bool):
            Optional. Indicates whether the dropping functionality should be enabled. If not
            given, `False` is used.

    Returns:
        ndarray, ndarray, ndarray, ndarray:
            * input sequences.
            * target sequences.
            * first index value of each input sequence.
            * first index value of each target sequence.
    """
    # if offset!=0:
    #     step_size=target_size

    out_X = list()
    out_y = list()
    X_index = list()
    y_index = list()
    # target = np.squeeze(X[:, target_column])
    target = X[:, target_column]

    if drop_windows:
        if hasattr(drop, '__len__') and (not isinstance(drop, str)):
            if len(drop) != len(X):
                raise Exception('Arrays `drop` and `X` must be of the same length.')
        else:
            if isinstance(drop, float) and np.isnan(drop):
                drop = np.isnan(X)
            else:
                drop = X == drop

    start = 0
    max_start = len(X) - window_size - target_size - offset + 1
    while start < max_start:
        end = start + window_size

        if drop_windows:
            drop_window = drop[start:end + target_size]
            to_drop = np.where(drop_window)[0]
            if to_drop.size:
                start += to_drop[-1] + 1
                continue

        out_X.append(X[start:end])
        out_y.append(target[end + offset:end + offset + target_size])
        X_index.append(index[start])
        y_index.append(index[end + offset])
        start = start + step_size

    return np.asarray((out_X)), np.asarray((out_y)), np.asarray(X_index), np.asarray(y_index)
    # return np.asarray(np.squeeze(out_X)), np.asarray(np.squeeze(out_y)), np.asarray(X_index), np.asarray(y_index)
