
class MeanEstimator:

    """Mean Estimator.

    This is a dummy estimator that always returns a constant value,
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

    This is a dummy estimator that always returns last value as the prediction,

    This estimator is here only to serve as reference of what
    an estimator primitive looks like, and is not intended to be
    used in real scenarios.
    """

    def __init__(self, value_column='value',lead = 'lead'):
        self._value_column = value_column
        self.lead = lead

    def fit(self, X):
        self.values = X[self._value_column]


    def predict(self):
        import numpy as np
        return np.full(self.lead,self.values.iloc[-1])
