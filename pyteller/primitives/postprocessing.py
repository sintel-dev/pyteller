import numpy as np
import pandas as pd
def Select_target(X,entities,target_index,pred_length): #THis os only for problems with offset
    X.index = pd.to_datetime(X.index.values * 1e9)

    X2 = X.iloc[::pred_length, :]
    df = pd.concat([X2[col] for col in X2])
    df = df.iloc[:-pred_length + 1]
    df.index = target_index

    cols = entities
    df = df.to_frame(cols)

    # make better by not subtracting last values, will need to just extend index
    return df



def flatten(X,pred_length, index,columns,freq):
    index = np.reshape(index, [index.size, 1])
    index = np.insert(index, 1, np.full((pred_length - 1, 1), freq), axis=1)
    index = np.cumsum(index, axis=1).flatten()
    df = pd.DataFrame(data=X.flatten(), index=index, columns=columns)
    df = df.groupby(df.index).mean()
    return df

def convert_date(X,entities):
    X.index = pd.to_datetime(X.index.values * 1e9)
    cols =entities
    X.columns=[cols]
    return X



def average_across(X):

    X1 = X[:, 0:5].mean(axis=1).reshape(len(X), 1)
    X2 = X[:, 5:].mean(axis=1).reshape(len(X), 1)
    XX = np.concatenate((X1, X2), axis=1)

    return(XX)
def max_across(X):
    X1 = X[:, 0:5].max(axis=1).reshape(len(X), 1)
    X2 = X[:, 5:].max(axis=1).reshape(len(X), 1)
    XX = np.concatenate((X1, X2), axis=1)

    return(XX)

def diff(X,index,interval=1):
    diff = list()
    for i in range(interval, len(X)):
        value = X[i] - X[i - interval]
        diff.append(value)
    index = index[2:]
    return np.array(diff), index

def un_diff(X,y_hat,target_index,index_orig):
    end = Xx.index.get_loc(target_index[-1])
    Xxx = Xx.iloc[:end + 1]

    start = Xx.index.get_loc(target_index[0])
    Xxxx = Xxx.iloc[start-2:]

    undiff = list()
    for i in range(len(y_hat)):
        value =y_hat.iloc[i]+Xxxx[i-2]
        undiff.append(value)
    return np.array(undiff)

def offset(X,index,y, target_index,offset):

    if offset == 0:
        return X, y, index, target_index
    X = X[:-offset]
    index = index[:-offset]
    target_index= target_index[offset:]
    y = y[offset:]
    return X,y,index ,target_index

def max_in_window(X, interval, time_column, method=['max']):
    """Aggregate values over given time span.

    Args:
        X (ndarray or pandas.DataFrame):
            N-dimensional sequence of values.
        interval (int):
            Integer denoting time span to compute aggregation of.
        time_column (int):
            Column of X that contains time values.
        method (str or list):
            Optional. String describing aggregation method or list of strings describing multiple
            aggregation methods. If not given, `mean` is used.

    Returns:
        ndarray, ndarray:
            * Sequence of aggregated values, one column for each aggregation method.
            * Sequence of index values (first index of each aggregated segment).
    """
    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X)

    X = X.sort_values(time_column).set_index(time_column)

    if isinstance(method, str):
        method = [method]

    start_ts = X.index.values[0]
    max_ts = X.index.values[-1]

    values = list()
    index = list()
    while start_ts <= max_ts:
        end_ts = start_ts + interval
        subset = X.loc[start_ts:end_ts - 1]
        aggregated = [
            getattr(subset, agg)(skipna=True).values
            for agg in method
        ]
        values.append(np.concatenate(aggregated))
        index.append(start_ts)
        start_ts = end_ts

    return np.asarray(values), np.asarray(index)


def get_index(X, time_column='timestamp'):
    """Stores the index of an input timeseries in the context
    Args:
        X (ndarray or pandas.DataFrame):
            N-dimensional sequence of values.
        time_column (int):
            Column of X that contains time values.

    Returns:
        ndarray, ndarray:
            * Sequence of averaged values.
            * Sequence of index values (first index of each averaged segment).
    """

    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X)

    X = X.sort_values(time_column).set_index(time_column)

    return np.asarray(X.values), np.asarray(X.index)


