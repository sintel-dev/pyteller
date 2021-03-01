# Logging
import logging

from pyteller.core import Pyteller
from pyteller.data import load_data

logging.basicConfig(level=logging.INFO)
logging.getLogger().setLevel(level=logging.ERROR)
logging.getLogger('pyteller').setLevel(level=logging.INFO)

# Load the dataset from a dataset on the s3 bucket, or in this example the local file path
current_data = load_data('pyteller/data/AL_Weather_current.csv')


pipeline = 'pyteller/pipelines/sandbox/ARIMA/arima.json'

hyperparameters = {
    "statsmodels.tsa.arima_model.Arima#1": {
        "p": 1,
        "d": 1,
        "q": 0
    }
}

pyteller = Pyteller(
    pipeline=pipeline,
    hyperparameters=hyperparameters,
    pred_length=5,
    offset=3,
    timestamp_col='valid',
    target_signal='tmpf',
    entity_col='station',
    entities='8A0',
)

# Fit the data to the pipeline.
train = pyteller.fit(current_data)

# Load the input_data
input_data = load_data('pyteller/data/AL_Weather_input.csv')

# forecast and evaluate
output = pyteller.forecast(data=input_data, visualization=False, plot=True)
scores = pyteller.evaluate(test_data=output['actual'], forecast=output['forecast'],
                           metrics=['MAPE', 'sMAPE'])
