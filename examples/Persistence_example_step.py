# Load the dataset from a dataset on the s3 bucket, or in this example the local file path
from mlblocks.discovery import find_pipelines

from pyteller.core import Pyteller
from pyteller.data import load_data

possible_pipelines = find_pipelines()

current_data = load_data('AL_Weather')
# current_data = load_data('pyteller/data/AL_Weather_current.csv')

# pipeline='sandbox.persistence.persistence_step_through'
pipeline = 'pyteller/pipelines/sandbox/persistence/persistence_step_through.json'

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

context = pyteller.fit(current_data, output_=1)
train = context['X']
pyteller.fit(start_=2, **context)

# Load the input_data
input_data = load_data('pyteller/data/AL_Weather_input.csv')

# forecast and evaluate
output = pyteller.forecast(data=input_data, visualization=True, plot=True)
scores = pyteller.evaluate(test_data=output['actual'], forecast=output['forecast'],
                           metrics=['MAPE'])
