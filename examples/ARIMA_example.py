# Load the dataset from a dataset on the s3 bucket, or in this example the local file path
from pyteller.data import load_data
current_data = load_data('pyteller/data/AL_Weather_current.csv')

from pyteller.core import Pyteller, ingest_data

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
current_data = ingest_data(pyteller,current_data)
pyteller.fit(current_data, ingested_data = True)

input_data=load_data('pyteller/data/AL_Weather_input.csv')
y,y_hat = pyteller.forecast(data=input_data)

scores= pyteller.evaluate(forecast=y_hat, test_data=y, metrics=['MAPE','sMAPE'])
