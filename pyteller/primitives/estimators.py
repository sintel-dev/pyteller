import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from dateutil.parser import parse

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

    def predict(self, X):
        preds = np.repeat(X[:, -1], self.pred_length, axis=1)
        return preds

