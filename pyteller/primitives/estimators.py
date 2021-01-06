import numpy as np
from sklearn.metrics import mean_absolute_error


class persistence:

    """Persistence Estimator.

    This is a persistence estimator that always returns last value as the prediction,

    This estimator is here only to serve as reference of what
    an estimator primitive looks like, and is not intended to be
    used in real scenarios.
    """

    def __init__(self, pred_length, offset):
        self.pred_length = pred_length
        self.offset = offset

    def fit(self, X, y):
        val = 0
        preds = np.repeat(X[:, [-1]], self.pred_length, axis=1)
        for i in range(X.shape[2]):
            pred, y_ = preds[:, :, i], y[:, :, i]
            val += mean_absolute_error(y_, pred)
        print('training MAE: ', val / X.shape[2])

    def predict(self, X):
        preds = np.repeat(X[:, [-1]], self.pred_length, axis=1)
        return preds
    # def fit(self, X, y):
    #     val=0
    #     preds = np.repeat(X[:, [-1], :], self.pred_length, axis=1)
    #     for i in range(X.shape[2]):
    #         pred, y_ = preds[:, :, i], y[:, :, i]
    #         val += mean_absolute_error(y_, pred)
    #     print('training MAE: ', val/X.shape[2])
    #
    # def predict(self, X):
    #     preds = np.repeat(X[:, [-1], :], self.pred_length, axis=1)
    #     return preds
