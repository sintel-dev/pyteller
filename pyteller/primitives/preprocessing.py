import numpy as np
import pandas as pd

def format_data(X, time_column=None, target_column=None, targets=None,
                entity_column=None, entities=None, make_index=False):
    """Format data into machine readable format.
    Args:
        df (pandas.DataFrame):
            Array of input sequence.
        time_column (str or int):
            Column denoting the timestamp column.
        target_column (str or int):
            Column denoting the target column.
        targets (list):
            Subset of targets to extract, If None, extract all targets.
        entity_column (str or int):
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
                | entity_id | target_id | timestamp  |  value |
                +-----------+------------+------------+--------+
                |   E1      |     C3     | 1404165600 |  215   |
                |   E2      |     C1     | 1404165600 |  239   |
                |   E1      |     C2     | 1404166300 |  134   |
                +-----------+------------+------------+--------+
        Output:
            * A ``pandas.DataFrame`` with timestamp column and one or many value columns.
    """
    if make_index == True:
        X.insert(0,'timestamp', range(len(X)))

    targets = targets or X.columns.drop(time_column)

    if isinstance(targets, str):
        targets = [targets]

    if isinstance(entities, str):
        entities = [entities]

    X = X.set_index(time_column)

    # check entities
    if entity_column:
        entities = entities or set(X[entity_column])
        targets = [x for x in targets if x != entity_column]
        X = X[X[entity_column].isin(entities)]  # filter row based

    if target_column:
        X = X.pivot(index=X.index, columns=target_column)
        X.columns = X.columns.droplevel()
        X.columns.name = None

    if make_index==False:
        if X.index.dtype == 'float' or X.index.dtype == 'int':
            X.index = pd.to_datetime(X.index.values * 1e9)

        else:
            X.index = pd.to_datetime(X.index)
    else:
        X.index = range(len(X))
        X.index.name = time_column


    return X[targets].reset_index(), X[targets].reset_index()


def get_index(X, time_column):
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

    X=X.sort_values(time_column.lower()).set_index(time_column.lower())
    freq=(X.index[1:2].astype(np.int64)-X.index[0:1].astype(np.int64))// 1e9

    return np.asarray(X.values), np.asarray(X.index), freq

