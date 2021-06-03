import logging
import os

import pandas as pd
from sklearn.model_selection import KFold, train_test_split

LOGGER = logging.getLogger(__name__)

DATA_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    'data'
)

BUCKET = 'pyteller'
S3_URL = 'https://{}.s3.amazonaws.com/{}'


def download(name, data_path=DATA_PATH):
    """Load the CSV with the given name from S3.

    If the CSV has never been loaded before, it will be downloaded
    from the [pyteller bucket](https://pyteller.s3.amazonaws.com) or
    the S3 bucket specified following the `s3://{bucket}/path/to/the.csv` format,
    and then cached inside the `data` folder, within the `pyteller` package
    directory, and then returned.

    Otherwise, if it has been downloaded and cached before, it will be directly
    loaded from the `pyteller/data` folder without contacting S3.

    Args:
        name (str): Name of the CSV to load.

    Returns:
        A single pandas.DataFrame is returned containing all
        the data.
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


def load_data(data, lookback=None, pred_length=None, test_size=0.2):
    if os.path.isfile(data):
        data = pd.read_csv(data)

    else:
        data = download(data)

    if test_size is None:
        return data

    if lookback and pred_length is not None:
        test_length = lookback + pred_length
    else:
        test_length = round(len(data) * test_size)

    train = data.iloc[:-test_length]
    test = data.iloc[-test_length:]

    return train, test


def _get_split(data, index):
    if hasattr(data, 'iloc'):
        return data.iloc[index]
    else:
        return data[index]


def get_splits(data, n_splits=1, random_state=None):
    """Return splits of this dataset ready for Cross Validation.

    If n_splits is 1, a tuple containing the X for train and test
    and the y for train and test is returned.
    Otherwise, if n_splits is bigger than 1, a list of such tuples
    is returned, one for each split.

    Args:
        n_splits (int): Number of times that the data needs to be splitted.
        data (array-like): Numpy array or pandas DataFrame containing all the data of
            this dataset, excluding the labels or target values.

    Returns:
        tuple or list:
            if n_splits is 1, a tuple containing the X for train and test
            and the y for train and test is returned.
            Otherwise, if n_splits is bigger than 1, a list of such tuples
            is returned, one for each split.
    """
    if n_splits == 1:

        return train_test_split(
            data,
            shuffle=False,
            stratify=False,
            random_state=random_state
        )

    else:
        cv_class = KFold
        cv = cv_class(n_splits=n_splits, shuffle=False, random_state=None)

        splits = list()
        for train, test in cv.split(data):
            X_train = _get_split(data, train)
            X_test = _get_split(data, test)
            splits.append((X_train, X_test))

        # for train, test in cv.split(self.data, self.target):
        #     X_train = self._get_split(self.data, train)
        #     y_train = self._get_split(self.target, train)
        #     X_test = self._get_split(self.data, test)
        #     y_test = self._get_split(self.target, test)
        #     splits.append((X_train, X_test, y_train, y_test))

        return splits
