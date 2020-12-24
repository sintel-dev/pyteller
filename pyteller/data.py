import os
import logging

import pandas as pd
import numpy as np

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


def load_data(data,
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
    #User specified multiple targets, this should be specified as multiple entities, fix here
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

    if 'entity' in columns: #Scenario 1: user specifies entity column and target variable column
        df['entity']=df['entity'].astype(str) # Make all entities strings
        all_entities = df.entity.unique() # Find the unique values in the entity column
        all_entities = [x for x in list(all_entities) if x != 'nan']
        self.entities=all_entities #entities are the unique values in the specified entity column

        #Make the long form into flatform by having entities as columns
        df2 = df.groupby('timestamp')['signal'].apply(
            lambda group_series: group_series.tolist()).reset_index()
        df2[all_entities] = pd.DataFrame(df2.signal.tolist(), index=df2.index)
        to_remove=list(['signal'])

        if entities != None: #Scenario 1b User specifies a certain entity from the entity column
            to_remove=to_remove+list(all_entities)
            entities = [entities] if isinstance(entities, str) else entities
            to_remove=list(set(to_remove)-set(entities))#Don't remove it
            # to_remove.remove(entities)
            self.entities=entities

        df = df2.drop(to_remove, axis=1)

    elif signal is not None: #Scenario 2: user specfies one signal
        self.entities = signal
        df=df.rename(columns={'signal':signal})

    else: #Scenario 3 user specifies multiple entitiies but there is no entity column
        self.entities=entities
        for entity in entities:
            df[entity] = data[[entity]]

    if df['timestamp'].dtypes !='float' and df['timestamp'].dtypes !='int':
        df['timestamp']=pd.to_datetime(df['timestamp']).values.astype(np.int64) // 1e9
    df = df.sort_values('timestamp')
    self.freq=df['timestamp'][1]-df['timestamp'][0]

    return df


def post_process(self, prediction):
    prediction.index = pd.to_datetime(prediction.index.values * 1e9)
    df = pd.concat([prediction[col] for col in prediction])

    from datetime import timedelta

    up = self.freq * (self.pred_length - 1)
    end = prediction.index[-1] + timedelta(seconds=up)
    freq = str(int(self.freq)) + 's'
    index = pd.date_range(start=prediction.index[0], end=end, freq=freq)

    df.index=index

    cols = self.entities
    df = df.to_frame(cols)
    return df
