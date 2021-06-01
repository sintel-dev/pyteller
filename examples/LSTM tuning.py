# Load the dataset from a dataset on the s3 bucket, or in this example the local file path
from pyteller.core import Pyteller
from pyteller.data import load_data

current_data, input_data = load_data('AL_Weather')

current_data.head()
pipeline = 'pyteller/pipelines/pyteller/LSTM/LSTM.json'

hyperparameters = {
    'keras.Sequential.LSTMTimeSeriesRegressor#1': {
        'epochs': 20
    }
}


pyteller = Pyteller(
    pipeline=pipeline,
    pred_length=12,
    time_column='valid',
    targets='tmpf',
    entity_column='station',
    entities='8A0',
    hyperparameters=hyperparameters
)

tunables = pyteller.pipeline.get_tunable_hyperparameters(flat=True)
# Fit the data to the pipeline.

pyteller.fit(current_data, tune=True)

pyteller.pipeline.get_hyperparameters()

# forecast and evaluate

output = pyteller.forecast(data=input_data, postprocessing=False, predictions_only=False)

scores = pyteller.evaluate(actuals=output['actuals'], forecasts=output['forecasts'],
                           metrics=['MAPE', 'sMAPE'])
pyteller.plot(output)
