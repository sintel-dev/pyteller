import os
import logging
import pandas as pd
import numpy as np

LOGGER = logging.getLogger(__name__)

DATA_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    'data'
)

BUCKET = 'd3-ai-orion'
S3_URL = 'https://{}.s3.amazonaws.com/{}'


def download(name, test_size=None, data_path=DATA_PATH):
    """Load the CSV with the given name from S3.

    If the CSV has never been loaded before, it will be downloaded
    from the [d3-ai-orion bucket](https://d3-ai-orion.s3.amazonaws.com) or
    the S3 bucket specified following the `s3://{bucket}/path/to/the.csv` format,
    and then cached inside the `data` folder, within the `orion` package
    directory, and then returned.

    Otherwise, if it has been downloaded and cached before, it will be directly
    loaded from the `orion/data` folder without contacting S3.

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


def load_csv(path, timestamp_column=None, value_column=None):
    header = None if timestamp_column is not None else 'infer'
    data = pd.read_csv(path, header=header)

    if timestamp_column is None:
        if value_column is not None:
            raise ValueError("If value_column is provided, timestamp_column must be as well")

        return data

    elif value_column is None:
        raise ValueError("If timestamp_column is provided, value_column must be as well")
    elif timestamp_column == value_column:
        raise ValueError("timestamp_column cannot be the same as value_column")

    timestamp_column_name = data.columns[timestamp_column]
    value_column_name = data.columns[value_column]
    columns = {
        'timestamp': data[timestamp_column_name].values,
        'value': data[value_column_name].values,
    }

    return pd.DataFrame(columns)[['timestamp', 'value']]



def load_signal(signal, test_size=None, timestamp_column=None, value_column=None):
    signal_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'data',
        signal,

    )
    if os.path.isfile(signal_path):
        data = load_csv(signal_path, timestamp_column, value_column)
    else:
        data = download(signal) #If you need to download the data

    #If the datatype of the column is object, it's the entity ID column. There can only be one.
    types = data.dtypes
    data = data.rename(columns={data.columns[0]: "timestamp", data.columns[np.where(types == 'object')[0][1]]: "entity_id"})

    if test_size is None:
        return data

    test_length = round(len(data) * test_size)
    train = data.iloc[:-test_length]
    test = data.iloc[-test_length:]

    return train, test
