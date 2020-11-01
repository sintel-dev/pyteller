import os
import logging

import pandas as pd

LOGGER = logging.getLogger(__name__)

DATA_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    'data'
)

BUCKET = 'pyteller'
S3_URL = 'https://{}.s3.amazonaws.com/{}'


def download(name, test_size=None, data_path=DATA_PATH):
    """Load the CSV with the given name from S3.

    If the CSV has never been loaded before, it will be downloaded
    from the [pyteller bucket](https://pyteller.s3.amazonaws.com) or
    the S3 bucket specified following the `s3://{bucket}/path/to/the.csv` format,
    and then cached inside the `data` folder, within the `pyteller` package
    directory, and then returned.

    Otherwise, if it has been downloaded and cached before, it will be directly
    loaded from the `pyteller/data` folder without contacting S3.

    If a `test_size` value is given, the data will be split in two parts
    without altering its order, making the second one proportionally as
    big as the given value.

    Args:
        name (str): Name of the CSV to load.
        test_size (float): Value between 0 and 1 indicating the proportional
            size of the test split. If 0 or None (default), the data is not split.

    Returns:
        If no test_size is given, a single pandas.DataFrame is returned containing all
        the data. If test_size is given, a tuple containing one pandas.DataFrame for
        the train split and another one for the test split is returned.
    """

    url = None
    if name.startswith('s3://'):
        parts = name[5:].split('/', 1)
        bucket = parts[0]
        path = parts[1]
        url = S3_URL.format(bucket, path)

        filename = os.path.join(data_path, path.split('/')[-1])
    else:
        filename = os.path.join(data_path, name + '.csv')

    if os.path.exists(filename):
        data = pd.read_csv(filename)
    else:
        url = url or S3_URL.format(BUCKET, '{}.csv'.format(name))

        LOGGER.info('Downloading CSV %s from %s', name, url)
        os.makedirs(data_path, exist_ok=True)
        data = pd.read_csv(url)
        data.to_csv(filename, index=False)

    return data


def load_csv(path):
    data = pd.read_csv(path)

    return pd.DataFrame(data)


def load_signal(data,
                train_size=.75,
                timestamp_col=None,
                entity_cols=None,
                signals=None,
                dynamic_variables=None,
                static_variables=None,
                column_dict=None):

    if os.path.isfile(data):
        data = load_csv(data)
    else:
        data = download(data)

    if column_dict is not None:
        allowed_keys = ['timestamp', 'signals', 'entity']
        columns = {k: column_dict[k] for k in allowed_keys}
        columns = {k: v for k, v in columns.items() if
                   pd.Series(v).notna().all()}  # remove Nan value keys
    else:
        columns = {
            'signals': signals,
            'timestamp': timestamp_col,
            'entity': entity_cols,

            'dynamic_variable': dynamic_variables,
            'static_variable': static_variables
        }

# TODO more than one target
    columns = {k: v for k, v in columns.items() if v is not None}
    df = pd.DataFrame()
    if 'signals' in columns:
        signals = columns['signals']
        if isinstance(signals, str):
            signals = [item.strip() for item in signals.split(',')]

        df = data[signals]
        signals_name = "".join('signal_{} '.format(x) for x in signals)  # add prefix signals
        signals_name = [item.strip() for item in signals_name.split(' ')]
        new_name = dict(zip(signals, signals_name))
        df = df.rename(columns=new_name)
    for key in columns:
        if key != 'signals':
            df[key] = data[columns[key]]

    # if column_dict is not None:
    #     is_entity = 'entity' in columns
    # if entity_cols is None and is_entity == False:
    # if is_entity == False:
    if 'entity' not in df:
        df = df.assign(entity=1)
    train_df = pd.DataFrame()
    test_df = pd.DataFrame()
    df = df.groupby('entity')
    for entity_name, entity_df in df:
        train_length = round(len(entity_df) * train_size)
        train = entity_df.iloc[:train_length]
        test = entity_df.iloc[train_length:]
        train_df = train_df.append(train)
        test_df = test_df.append(test)
    train = train_df.groupby('entity')
    test = test_df.groupby('entity')
    # for entity, train_entity in train:
    #     if train_entity["timestamp"].is_unique == False:
    #         raise ValueError(
    #             'There are multiple values for a single timestamp, please choose an entity column that will group the data to have only one value for a timestep for each entity.')
    return train, test
