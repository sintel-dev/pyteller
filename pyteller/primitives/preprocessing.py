import numpy as np
import pandas as pd


def format_data(X, target_signal, timestamp_col, static_variables, entity_col, entities):
    # target_signal = self.target_signal
    # timestamp_col = self.timestamp_col
    # static_variables = self.static_variables
    # entity_col = self.entity_cols
    # entities = self.entities

    # Fix if the user specified multiple targets. They should be specified as multiple entities
    entities = target_signal if isinstance(target_signal, list) else entities
    signal = None if isinstance(target_signal, list) else target_signal

    columns = {
        'signal': signal,
        'timestamp': timestamp_col,
        'entity': entity_col,
        'static_variable': static_variables
    }
    columns = {k: v for k, v in columns.items() if v is not None}

    df = pd.DataFrame()
    for key in columns:
        df[key] = X[columns[key]]

    # Scenario 1: (longform) user specifies entity column and target variable column
    if 'entity' in columns:
        df['entity'] = df['entity'].astype(str)  # Make all entities strings
        all_entities = df.entity.unique()  # Find the unique values in the entity column
        all_entities = [x for x in list(all_entities) if x != 'nan']
        entities_new = all_entities  # entities are the unique values in specified entity column

        # Make the long form into flatform by having entities as columns
        df = df.pivot(index='timestamp', columns='entity')['signal'].reset_index()

        # Scenario 1b User specifies certain entities from entity column
        if entities is not None:
            entities = [entities] if isinstance(entities, str) else entities
            to_remove = list(all_entities)
            to_remove = list(set(to_remove) - set(entities))  # Don't remove it
            entities_new = entities
            df = df.drop(to_remove, axis=1)

    # Scenario 2: (flatform) user specifies one signal
    elif signal is not None:
        entities_new = [signal]
        df = df.rename(columns={'signal': signal})
        if True in df.duplicated('timestamp'):
            raise ValueError('Multiple observations per timestamp')

    # Scenario 3: (flatform) user specifies multiple entities but there is no entity column
    else:
        entities_new = entities
        for entity in entities:
            df[entity] = X[[entity]]

    # Convert to epoch time
    if df['timestamp'].dtypes != 'float' and df['timestamp'].dtypes != 'int':
        df['timestamp'] = pd.to_datetime(df['timestamp']).values.astype(np.int64) // 1e9

    df = df.sort_values('timestamp')
    freq = df['timestamp'][1] - df['timestamp'][0]
    if isinstance(entities_new, str):
        target_column = [0]
    else:
        target_column = list(range(len(entities_new)))
    X = df
    _X = X
    entities = entities_new
    return X, _X, freq, target_column, entities


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


def rolling_window_sequences(X, index, window_size, target_size, step_size, target_column=None,
                             offset=0, drop=None, drop_windows=False):
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


def format_data2(X, timestamp_col='timestamp', target_signal_column=None, target_signal=None,
                 entity_col=None, entities=None):
    """Format data into machine readible format.
    Args:
        X (pandas.DataFrame):
            Array of input sequence.
        timestamp_col (str or int):
            Column denoting the timestamp column.
        target_signal_column (str or int):
            Column denoting the channel column.
        target_signal (list):
            Subset of channels to extract, If None, extract all channels.
        entity_col (str or int):
            Column denoting the entities column.
        entities (list):
            Subset of entities to extract. If None, extract all entities.
    Returns:
        pandas.DataFrame:
            Formatted data into machine readible (timestamp, values).
    Examples:
        Input:
            * multiple entity-instances, univariate
                +-----------+------------+--------+
                | entity_id | timestamp  |  value |
                +-----------+------------+--------+
                |   E1      | 1404165600 |  215   |
                |   E2      | 1404165600 |  400   |
                |   E1      | 1404166300 |  150   |
                +-----------+------------+--------+
            * multiple entity-instances, multivariate -- horizonal
                +-----------+------------+------+------+------+
                | entity_id | timestamp  |  v1  |  v2  |  v3  |
                +-----------+------------+------+------+------+
                |   E1      | 1404165600 |  15  |  37  |  48  |
                |   E2      | 1404165600 |  20  |  30  |  20  |
                |   E1      | 1404166300 |  34  |  97  |  23  |
                +-----------+------------+------+------+------+
            * multiple entity-instances, multivariate -- longform
                +-----------+------------+------------+--------+
                | entity_id | channel_id | timestamp  |  value |
                +-----------+------------+------------+--------+
                |   E1      |     C3     | 1404165600 |  215   |
                |   E2      |     C1     | 1404165600 |  239   |
                |   E1      |     C2     | 1404166300 |  134   |
                +-----------+------------+------------+--------+
        Output:
            * A ``pandas.DataFrame`` with timestamp column and one or many value columns.
    """
    target_signal = target_signal or X.columns.drop(timestamp_col)
    if isinstance(target_signal, str):
        target_signal = [target_signal]
    if isinstance(entities, str):
        entities = [entities]
    X = X.set_index(timestamp_col)
    # check entities
    if entity_col:
        entities = entities or set(X[entity_col])
        target_signal = [x for x in target_signal if x != entity_col]
        X = X[X[entity_col].isin(entities)] # filter row based
    if target_signal_column:
        X = X.pivot(index=X.index, columns=target_signal_column)
        X.columns = X.columns.droplevel()
        X.columns.name = None
    return X[target_signal].reset_index()
