# Load the dataset from a dataset on the s3 bucket, or in this example the local file path
from pyteller.core import Pyteller
from pyteller.data import load_data

current_data = load_data('pyteller/data/AL_Weather_current.csv')


pipeline = 'pyteller/pipelines/sandbox/LSTM/LSTM_offset.json'
pyteller = Pyteller(
    pipeline=pipeline,
    pred_length=5,
    offset=3,
    timestamp_col='valid',
    target_signal='tmpf',
    # static_variables=None,
    entity_col='station',
    entities='8A0',
)

# Fit the data to the pipeline.
train = pyteller.fit(current_data)

# Load the input_data
input_data = load_data('pyteller/data/AL_Weather_input.csv')

# forecast and evaluate
output = pyteller.forecast(data=input_data, visualization=False, plot=True)
scores = pyteller.evaluate(
    test_data=output['actual'],
    forecast=output['forecast'],
    metrics=[
        'MAPE',
        'sMAPE'])


