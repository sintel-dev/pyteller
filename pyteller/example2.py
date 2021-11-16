# Tuning a Pyteller Pipeline
from pyteller.data import load_data

## Load the input Data, Input data is loaded from s3 bucket

current_data,input_data = load_data('AL_Weather')
current_data.head()
pipeline = 'pyteller.LSTM.LSTM'
# current_data=current_data[['tmpf','valid','station']]
# input_data=input_data[['tmpf','valid','station']]
## Set hyperparameters
hyperparameters = {
    'keras.Sequential.LSTMTimeSeriesRegressor#1': {
        'epochs': 10,
        'verbose': False
    }
}

from mlblocks.discovery import find_pipelines
pipelines=find_pipelines('pyteller')

## Instantiate the pyteller pipeline by specifying the column names and desired prediction length

from pyteller import Pyteller

pyteller = Pyteller(
    pipeline=pipeline,
    pred_length=1,
    offset=0,
    time_column='valid',
    targets='tmpf',
    entity_column='station',
    entities='8A0',
    target_column=0
)

## Get the tunable hyperparameters
#These hyperparamteres are set in the primitive `.jsons` and also can be set in the pipelines in the `tunable` `.json` field

tunables = pyteller.pipeline.get_tunable_hyperparameters(flat=True)
print(tunables)

## Fit the pipeline
pyteller.fit(current_data, tune=False, max_evals=4)

## Check what tuner found the best hyperparameters to be
best_params=pyteller.pipeline.get_hyperparameters()
print(best_params)

## Forecast
output = pyteller.forecast(data=input_data, postprocessing=False, predictions_only=False)

## Evaluate
scores = pyteller.evaluate(actuals=output['actuals'], forecasts=output['forecasts'],
                           metrics=['MAPE', 'sMAPE'])

scores.head()

## Plot
pyteller.plot(output)
