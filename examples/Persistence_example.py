# Load the dataset from a dataset on the s3 bucket, or in this example the local file path
from pyteller.data import load_data
current_data = load_data('pyteller/data/AL_Weather_current.csv')

# Specify pipeline and hyperparamters if any
pipeline = 'pyteller/pipelines/sandbox/persistence/persistence.json'

# Make instance of Pyteller, specifying where to make the prediction and the column names

from pyteller.core import Pyteller, ingest_data
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

# Ingest the data if you wish to be able to see the format that will go into the pipeline
current_data = ingest_data(pyteller, current_data)

# Fit the data to the pipeline
pyteller.fit(current_data, ingested_data = True)

# Load and ingest the input_data
input_data=load_data('pyteller/data/AL_Weather_input.csv')
input_data = ingest_data(pyteller, input_data)

# forecast and evaluate
y,y_hat = pyteller.forecast(data=input_data, ingested_data = True)
scores= pyteller.evaluate(forecast=y_hat, test_data=y, metrics=['MAPE','sMAPE'])

# plot
from pyteller.utils import plot
plot([y.iloc[:,0:1],y_hat.iloc[:,0:1]],frequency='day')
