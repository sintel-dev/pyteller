import os
import logging
import numpy as np

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
                targets=None,
                dynamic_variables=None,
                static_variables=None,
                column_dict=None):

    if os.path.isfile(data):
        data = load_csv(data)
    else:
        data = download(data)

    if column_dict != None:
        columns = column_dict
    else:
        columns = {
            'timestamp': timestamp_col,
            'entity': entity_cols,
            'target': targets,
            'dynamic_variable': dynamic_variables,
            'static_variable': static_variables
        }
# TODO: More than one entity or target col etc, need to umpack
        columns = {k: v for k, v in columns.items() if v != None}
    df = pd.DataFrame()
    for key in columns:
        df[key] = data[columns[key]]
    df=df.iloc[0:10000]


    # df['timestamp'] = pd.to_datetime(df['timestamp'], infer_datetime_format=True)
    train_length = round(len(df) * train_size)
    train = df.iloc[:train_length]
    test = df.iloc[train_length:]
    if entity_cols == None and column_dict['entity']==None:
        train=train.assign(entity=1)
        test = test.assign(entity=1)
    train = train.groupby('entity')
    test = test.groupby('entity')
    for entity, train_entity in train:
        if train_entity["timestamp"].is_unique==False:
            raise ValueError('There are multiple values for a single timestamp, please choose an entity column that will group the data to have only one value for a timestep for each entity.')
    return train, test

