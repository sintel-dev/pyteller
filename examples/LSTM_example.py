# Load the dataset from a dataset on the s3 bucket, or in this example the local file path
from pyteller.core import Pyteller
from pyteller.data import load_data

current_data, input_data = load_data('AL_Weather')

current_data.head()
pipeline = 'pyteller/pipelines/pyteller/LSTM/LSTM.json'
# hyperparameters = {
#     'pyteller.primitives.preprocessing.format_data#1': {
#         'make_index': False
#     },
#     'pyteller.primitives.postprocessing.flatten#1': {
#         'type': 'horizon'
#     }
# }

hyperparameters = {
    'keras.Sequential.LSTMTimeSeriesRegressor#1': {
        'epochs': 20
    }
}


pyteller = Pyteller(
    pipeline=pipeline,
    pred_length=40,
    offset=0,
    time_column='valid',
    targets='tmpf',
    # targets=['tmpf','dwpf'],
    # target_column='station',
    entity_column='station',
    entities='8A0',
    hyperparameters=hyperparameters
)

# Fit the data to the pipeline.
pyteller.fit(current_data, tune=False)


# forecast and evaluate

output = pyteller.forecast(data=input_data, postprocessing=False, predictions_only=False)

scores = pyteller.evaluate(actuals=output['actuals'], forecasts=output['forecasts'],
                           metrics=['MAPE', 'sMAPE'])
pyteller.plot(output)
