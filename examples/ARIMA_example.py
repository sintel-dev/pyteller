# Logging
import logging

from pyteller.core import Pyteller
from pyteller.data import load_data

logging.basicConfig(level=logging.INFO)
logging.getLogger().setLevel(level=logging.ERROR)
logging.getLogger('pyteller').setLevel(level=logging.INFO)

# Load the dataset from a dataset on the s3 bucket, or in this example the local file path
current_data, input_data = load_data('AL_Weather')


pipeline = 'pyteller/pipelines/pyteller/ARIMA/arima.json'

hyperparameters = {
    "statsmodels.tsa.arima_model.Arima#1": {
        "p": 2,
        "d": 1,
        "q": 2
    }
}

pyteller = Pyteller(
    pipeline=pipeline,
    pred_length=5,
    offset=3,
    time_column='valid',
    targets='tmpf',
    entity_column='station',
    entities='8A0',
)

# Fit the data to the pipeline.
train = pyteller.fit(current_data.iloc[0:1000])

# forecast and evaluate
output = pyteller.forecast(data=input_data, postprocessing=False, predictions_only=False)
scores = pyteller.evaluate(actuals=output['actuals'], forecasts=output['forecasts'],
                           metrics=['MAPE', 'sMAPE'])

pyteller.plot(output)
