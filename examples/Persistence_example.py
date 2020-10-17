from pyteller.data import load_signal

#Pick a dataset from the S3 bucket or give a local path
dataset = 'FasTrak'

# Here we load the dataset and the user secifies what the columns are
# Slpit the data into train and test data
train,test = load_signal(
    data=dataset,
    timestamp_col = 'timestamp',
    targets='Total Flow',
    static_variables=None,
    # entity_cols=None,
    entity_cols='Location Identifier',
    train_size=.75
)

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
    lead = 3,
    goal = None,
    goal_window = None

)

# TODO: split for supervised learning, should this be a primitive?


forecast = pyteller.predict(test_data=test)
# pred_window=(test_entity['timestamp']>= forecast['timestamp'].iloc[0] )&(test_entity['timestamp']<= forecast['timestamp'].iloc[-1])
# actual=test_entity.loc[pred_window]

# TODO: group by entity i think i need a for loop idk how else to not mess with mlpipelines
# grouped=test.groupby('entity')
# for x in grouped.groups:
#     xtest=grouped.get_group(x)
#     forecast = pyteller._mlpipeline.predict(xtest.loc[:, ['timestamp', 'target']])

scores = pyteller.evaluate(train_data= train,test_data=test,forecast=forecast)
# mape = METRICS['MAPE'](ytest, forecast)

