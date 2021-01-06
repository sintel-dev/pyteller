import logging
import os

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


def load_data(data):

    if os.path.isfile(data):
        data = load_csv(data)
    else:
        data = download(data)
    return data


def ingest_data(self,
                data,
                timestamp_col=None,
                entity_col=None,
                entities=None,
                signal=None,
                dynamic_variables=None,
                static_variables=None,
                ):
    # Fix if the user specified multiple targets. They should be specified as multiple entities
    entities = signal if isinstance(signal, list) else entities
    signal = None if isinstance(signal, list) else signal

    columns = {
        'signal': signal,
        'timestamp': timestamp_col,
        'entity': entity_col,
        'dynamic_variable': dynamic_variables,
        'static_variable': static_variables
    }
    columns = {k: v for k, v in columns.items() if v is not None}

    df = pd.DataFrame()
    for key in columns:
        df[key] = data[columns[key]]

    # Scenario 1: (longform) user specifies entity column and target variable column
    if 'entity' in columns:
        df['entity'] = df['entity'].astype(str)  # Make all entities strings
        all_entities = df.entity.unique()  # Find the unique values in the entity column
        all_entities = [x for x in list(all_entities) if x != 'nan']
        self.entities = all_entities  # entities are the unique values in specified entity column

        # Make the long form into flatform by having entities as columns
        df = df.pivot(index='timestamp', columns='entity')['signal'].reset_index()

        # Scenario 1b User specifies certain entities from entity column
        if entities is not None:
            entities = [entities] if isinstance(entities, str) else entities
            to_remove = list(all_entities)
            to_remove = list(set(to_remove) - set(entities))  # Don't remove it
            self.entities = entities

            df = df.drop(to_remove, axis=1)

    # Scenario 2: (flatform) user specifies one signal
    elif signal is not None:
        self.entities = [signal]
        df = df.rename(columns={'signal': signal})
        if True in df.duplicated('timestamp'):
            raise ValueError('Multiple observations per timestamp')

    # Scenario 3: (flatform) user specifies multiple entities but there is no entity column
    else:
        self.entities = entities
        for entity in entities:
            df[entity] = data[[entity]]

    # Convert to epoch time
    if df['timestamp'].dtypes != 'float' and df['timestamp'].dtypes != 'int':
        df['timestamp'] = pd.to_datetime(df['timestamp']).values.astype(np.int64) // 1e9
    df = df.sort_values('timestamp')
    self.freq = df['timestamp'][1] - df['timestamp'][0]
    if isinstance(self.entities, str):
        self.target_column = [0]
    else:
        self.target_column = list(range(len(self.entities)))
    return df


def egest_data(test, prediction):
    if prediction.index.dtype == 'float' or prediction.index.dtype == 'int':
        prediction.index = pd.to_datetime(prediction.index.values * 1e9)
    else:
        prediction.index = pd.to_datetime(prediction.index)

    if test['timestamp'].dtypes == 'float' or test['timestamp'].dtypes == 'int':
        test['timestamp'] = pd.to_datetime(test['timestamp'] * 1e9)
    else:
        test['timestamp'] = pd.to_datetime(test['timestamp'])

    actual = test.set_index('timestamp')
    actual = actual[actual.index.isin(prediction.index)]
    return actual, prediction
