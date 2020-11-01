
class MeanEstimator:

    """Mean Estimator.

    This is a persistence estimator that always returns a constant value,
    which consist on the mean value from the given input.

    This estimator is here only to serve as reference of what
    an estimator primitive looks like, and is not intended to be
    used in real scenarios.
    """

    def __init__(self, value_column='value'):
        self._value_column = value_column

    def fit(self, X):
        import numpy as np
        values = X[self._value_column]
        self._mean = np.mean(values)

    def predict(self, X):
        import numpy as np
        return np.full(len(X), self._mean)


class Persistence:

    """Persistence Estimator.

    This is a persistence estimator that always returns last value as the prediction,

    This estimator is here only to serve as reference of what
    an estimator primitive looks like, and is not intended to be
    used in real scenarios.
    """

    def __init__(self):
        self.values_col = 'target'

    # def fit(self, X):
    #     self.values = X[self._value_column]
    def predict(self, X, pred_length):
        import numpy as np
        import pandas as pd
        signals = [col for col in X if col.startswith('signal')]
        self.values = X[signals]
        time = pd.DataFrame(data=X['timestamp'][-pred_length:].values, columns=['timestamp'])
        # preds = self.values.iloc[-pred_length:, :]
        preds = self.values.iloc[-pred_length - 1:-pred_length, :]
        preds = pd.concat([preds]*pred_length)
        preds['timestamp'] =time.values
        return preds


def split_sequence(X, target_column, sequence_size, overlap_size):
    """Split sequences of time series data.

    The function creates a list of input sequences by splitting the input sequence
    into partitions with a specified size and pads it with values from previous
    sequence according to the overlap size.

    Args:
        X (ndarray):
            N-dimensional value sequence to iterate over.
        index (ndarray):
            N-dimensional index sequence to iterate over.
        target_column (int):
            Indicating which column of X is the target.
        sequence_size (int):
            Length of the input sequences.
        overlap_size (int):
            Length of the values from previous window.

    Returns:
        tuple:
            * List of sliced value as ndarray.
            * List of sliced index as ndarray.
    """
    X_ = list()
    index_ = list()

    overlap = 0
    start = 0
    max_start = len(X) - 1

    target = X[target_column]
    # TODO : target column is in hyperparams, but index is right now hard coded
    index = X['timestamp']
    while start < max_start:
        end = start + sequence_size

        X_.append(target[start - overlap:end])
        index_.append(index[start - overlap:end])

        start = end
        overlap = overlap_size

    return X_, index_
