from pyteller.data import load_data

# Load the dataset from a dataset on the s3 bucket, or in this example the local file path
from pyteller.data import load_data
# current_data=load_data('../pyteller/data/Demand_average_train.csv').iloc[0:6000,:]
current_data=load_data('../pyteller/data/taxi.csv')#.iloc[0:6000,:]
# current_data=load_data('../pyteller/data/Demand_max_train.csv')#.iloc[0:6000,:]


# Initialization settings: user doesn't change these once new data comes in
from pyteller.core import Pyteller
pyteller = Pyteller (
    # pipeline='lagged_demand_average',
    pipeline='../pyteller/pipelines/sandbox/LSTM/lstm_demand_max.json',
    # pred_length=10,
    # offset=5,
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
# input_data=load_data('../pyteller/data/Demand_average_test.csv')
# input_data=load_data('../pyteller/data/Demand_max_test.csv')
y,y_hat = pyteller.forecast(data=input_data)


scores= pyteller.evaluate(forecast=y_hat, test_data=y, metrics=['MAPE','sMAPE'])


import json
with open('../fit_models/lstm_max_35.txt', 'w') as file:
    file.write(json.dumps(scores))

pyteller.save('../fit_models/lstm_max_35')

# from pyteller.core import Pyteller
# path='../fit_models/lstm_max_35'
# b=Pyteller.load(path)

