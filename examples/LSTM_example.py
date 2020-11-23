from pyteller.data import load_data

# Load the dataset from a dataset on the s3 bucket, or in this example the local file path
from pyteller.data import load_data
current_data=load_data('../pyteller/data/taxi.csv')


# Initialization settings: user doesn't change these once new data comes in
from pyteller.core import Pyteller
pyteller = Pyteller (
    pipeline='mean_24h_lstm',
    pred_length=10,
    offset=5,
    goal=None,
    goal_window=None
)

pyteller.fit(
    data=current_data,
    timestamp_col='timestamp',
    target_signals='value',
    static_variables=None,
    # entity_col='station',
    train_size=.75)

input_data=load_data('../pyteller/data/taxi_test.csv')
forecast = pyteller.forecast(data=input_data)


scores = pyteller.evaluate(forecast=forecast, test_data=input_data, metrics='MAPE')

# pyteller.save('../fit_models/persistence')


