# Load the dataset from a dataset on the s3 bucket, or in this example the local file path
from pyteller.core import Pyteller
from pyteller.data import load_data

current_data, input_data = load_data('AL_Weather')

# Specify pipeline and hyperparamters if any
pipeline = 'pyteller/pipelines/pyteller/persistence/persistence.json'

# Make instance of Pyteller, specifying where to make the prediction and the column names
pyteller = Pyteller(
    pipeline=pipeline,
    pred_length=10,
    offset=5,
    timestamp_col='valid',
    target_signal='tmpf',
    entity_col='station',
    entities='8A0',
)

# Fit the data to the pipeline.
pyteller.fit(current_data)

# forecast and evaluate
output = pyteller.forecast(data=input_data, postprocessing=True, predictions_only=False)
scores = pyteller.evaluate(test_data=output['actual'], forecast=output['forecast'],
                           metrics=['MAE', 'sMAPE'])

pyteller.plot(output)
