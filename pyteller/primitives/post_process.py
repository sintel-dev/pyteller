import numpy as np
import pandas as pd
def Flatten(X,freq,pred_length,entities,offset):
    X.index = pd.to_datetime(X.index.values * 1e9)

    X2 = X.iloc[:, -pred_length:]
    df = pd.concat([X2[col] for col in X2])

    from datetime import timedelta

    up = freq * (X.shape[1] - 1)
    start = X.index[0] + timedelta(seconds=int(offset * freq))
    end = X.index[-1] + timedelta(seconds=int(up))
    freq_s = str(int(freq)) + 's'
    index = pd.date_range(start=start, end=end, freq=freq_s)

    df.index = index

    cols =entities
    df = df.to_frame(cols)
    return df

def convert_date(X,entities):
    X.index = pd.to_datetime(X.index.values * 1e9)
    cols =entities
    X.columns=[cols]
    return X

def get_index(X):


    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X)

    X = X.sort_values('timestamp').set_index('timestamp')

    return np.asarray(X.values), np.asarray(X.index)
