import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from dateutil.parser import parse
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


class persistence:

    """Persistence Estimator.

    This is a persistence estimator that always returns last value as the prediction,

    This estimator is here only to serve as reference of what
    an estimator primitive looks like, and is not intended to be
    used in real scenarios.
    """

    def __init__(self,pred_length,offset):
        self.pred_length = pred_length
        self.offset = offset

    def fit(self, X, y):
        preds=np.repeat(X[:,-1], self.pred_length,axis=1)
        #Validation
        val = mean_absolute_error(y, preds)
        print('training MAE: ' ,val)



    # def fit(self, X):
    #     self.values = X[self._value_column]
    def predict(self, X):
        preds = np.repeat(X[:, -1], self.pred_length, axis=1)

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


def lagged(X, index, shift):
    X = pd.DataFrame(data=X, index=index)
    y = X.shift(shift)
    y_hat=y.iloc[shift:, :]
    target_index=y_hat.index
    return np.asarray(y_hat), np.asarray(target_index)

import math
def make_cylinder_volume_func(r):
    def volume(h):
        return math.pi * r * r * h
    return volume

def rho_loss():
    import tensorflow as tf
    def loss(q, y_p, y):
        e = y_p - y
        return tf.keras.backend.mean(tf.keras.backend.maximum(q * e, (q - 1) * e))
    return rho_loss

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from dateutil.parser import parse
# class persistence:
#
#     def __init__(self, offset=1):
#         super(NaiveForecaster, self).__init__()
#         self.offset = offset
#
#     def fit(self, y, X=None, fh=None):
#
#         return self
