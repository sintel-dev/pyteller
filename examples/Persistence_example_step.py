# Load the dataset from a dataset on the s3 bucket, or in this example the local file path
from pyteller.core import Pyteller
from pyteller.data import load_data
from pyteller.utils import plot

current_data = load_data('pyteller/data/AL_Weather_current.csv')

# Specify pipeline and hyperparamters if any
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

train = pyteller.fit(current_data, output_=0)
# train = pyteller.fit(current_data, output_='training_data')
# train = pyteller.fit(current_data, output_=0)
pyteller.fit(**train)

# Load the input_data
input_data = load_data('pyteller/data/AL_Weather_input.csv')

# forecast and evaluate
output = pyteller.forecast(data=input_data, visualization=False)
scores = pyteller.evaluate(
    test_data=output['actual'],
    forecast=output['forecast'],
    metrics=[
        'MAPE',
        'sMAPE'])

# plot
plot([output['actual'].iloc[:, 0:1], output['forecast'].iloc[:, 0:1]], frequency='day')