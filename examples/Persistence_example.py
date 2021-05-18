# Load the dataset from a dataset on the s3 bucket, or in this example the local file path
from pyteller.core import Pyteller
from pyteller.data import load_data

current_data, input_data = load_data('AL_Weather')

# Specify pipeline and hyperparamters if any
pipeline = 'pyteller/pipelines/pyteller/persistence/persistence.json'

# Make instance of Pyteller, specifying where to make the prediction and the column names
pyteller = Pyteller(
    pipeline=pipeline,
    pred_length=5,
    offset=0,
    time_column='valid',
    targets='tmpf',
    entity_column='station',
    entities='8A0',
)

# Fit the data to the pipeline.
train = pyteller.fit(current_data,tune=False)


# forecast and evaluate
output = pyteller.forecast(data=input_data, postprocessing=False, predictions_only=False)
scores = pyteller.evaluate(actuals=output['actuals'], forecasts=output['forecasts'],
                           metrics=['MAPE', 'sMAPE'])
pyteller.plot(output)
