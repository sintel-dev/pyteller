# Load the dataset from a dataset on the s3 bucket, or in this example the local file path
from pyteller.core import Pyteller
from pyteller.data import load_data

current_data = load_data('pyteller/data/AL_Weather_current.csv')

# Specify pipeline and hyperparamters if any
pipeline = 'pyteller/pipelines/pyteller/persistence/persistence.json'

# Make instance of Pyteller, specifying where to make the prediction and the column names
pyteller = Pyteller(
    pipeline=pipeline,
    pred_length=10,
    offset=5,
    timestamp_col='valid',
    target_signal='tmpf',
    # static_variables=None,
    entity_col='station',
    # entities='8A0',
)

# Fit the data to the pipeline.
train = pyteller.fit(current_data)

# Load the input_data
input_data = load_data('pyteller/data/AL_Weather_input.csv')

# forecast and evaluate
output = pyteller.forecast(data=input_data, postprocessing=True, predictions_only=False)
scores = pyteller.evaluate(test_data=output['actual'],forecast=output['forecast'],
                           metrics=['MAPE','sMAPE'])

