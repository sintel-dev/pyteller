# Load the dataset from a dataset on the s3 bucket, or in this example the local file path
from pyteller.data import load_data
current_data = load_data('pyteller/data/AL_Weather_current.csv').iloc[0:1000]


from pyteller.core import Pyteller
pipeline = 'pyteller/pipelines/sandbox/ARIMA/arima.json'
pyteller = Pyteller(
    pipeline=pipeline,
    pred_length=5,
    offset=3
)

pyteller.fit(
    data=current_data,
    timestamp_col='valid',
    target_signal='tmpf',
    # static_variables=None,
    entity_col='station',
    entities='8A0'
    )

input_data=load_data('pyteller/data/AL_Weather_input.csv').iloc[0:1000]
y,y_hat = pyteller.forecast(data=input_data)

scores= pyteller.evaluate(forecast=y_hat, test_data=y, metrics=['MAPE','sMAPE'])
