from pyteller.data import load_data

# Load the dataset from a dataset on the s3 bucket, or in this example the local file path
current_data=load_data('../pyteller/data/AL_Weather_current.csv')

#Make a Pyteller object
from pyteller.core import Pyteller
#Initialization settings: user doesn't change these once new data comes in
pyteller = Pyteller (
    data=current_data,
    timestamp_col='valid',
    signals=['tmpf', 'dwpf'],
    static_variables=None,
    entity_cols='station',
    # train_size=.75
)

#forecast settings user may want to play around with
pyteller.forecast_settings(
    pipeline = 'persistence',
    pred_length = 3,
    offset=5,
    goal = None,
    goal_window = None
)

pyteller.fit()
pyteller.save('../fit_models/persistence') # TODO dump training data
input_data=load_data('../pyteller/data/AL_Weather_input.csv')
forecast = pyteller.forecast()

# scores = pyteller.evaluate(forecast=forecast,metrics=['MAPE','MSE'])


