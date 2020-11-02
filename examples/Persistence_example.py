from pyteller.data import load_signal

#Pick a dataset from the S3 bucket or give a local path
dataset = 'AL_Weather'

# Here we load the dataset and the user specifies what the columns are
# Slpit the data into train and test data

train,test = load_signal(
    data=dataset,
    timestamp_col = 'valid',
    signals=['tmpf','dwpf'],
    static_variables=None,
    # entity_cols=None,
    entity_cols='station',
    train_size=.75
)
# TODO: variable column for longform
#Set up the pipeline
hyperparameters =  {
    "pyteller.primitives.estimators.Persistence#1": {
    }
}

#Make a Pyteller object
from pyteller.core import Pyteller
pyteller = Pyteller (
hyperparameters = hyperparameters,
    pipeline = 'persistence',
    pred_length = 3,
    offset=5,
    goal = None,
    goal_window = None

)

# TODO: split for supervised learning


forecast = pyteller.predict(test_data=test)

scores = pyteller.evaluate(train_data= train,test_data=test,forecast=forecast,metrics=['MAPE','MSE'])


