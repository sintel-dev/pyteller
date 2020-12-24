import numpy as np
import pandas as pd


def flatten(X, pred_length, index, columns, freq):
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
    df = pd.DataFrame(data=X.flatten(), index=index, columns=columns)
    df = df.groupby(df.index).mean()
    return df
