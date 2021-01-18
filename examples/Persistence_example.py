
# Load the dataset from a dataset on the s3 bucket, or in this example the local file path
from pyteller.data import load_data
current_data = load_data('pyteller/data/AL_Weather_current.csv')

from pyteller.core import Pyteller, ingest_data

pipeline = 'pyteller/pipelines/sandbox/persistence/persistence.json'
pyteller = Pyteller(
    pipeline=pipeline,
    pred_length=5,
    offset=3,
    timestamp_col='valid',
    target_signal='tmpf',
    # static_variables=None,
    entity_col='station',
    # entities='8A0',
)
current_data = ingest_data(pyteller, current_data)
pyteller.fit(current_data, ingested_data = True)

input_data=load_data('pyteller/data/AL_Weather_input.csv')
input_data = ingest_data(pyteller, input_data)
y,y_hat = pyteller.forecast(data=input_data, ingested_data = True)

scores= pyteller.evaluate(forecast=y_hat, test_data=y, metrics=['MAPE','sMAPE'])

from pyteller.utils import plot
plot([y.iloc[:,0:1],y_hat.iloc[:,0:1]],frequency='day')
