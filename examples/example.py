from pyteller.data import load_signal

dataset = 'AL_Weather'


train,test = load_signal(
    data=dataset,
    timestamp_col = 'valid',
    target = 'tmpf'
)


from pyteller.core import Pyteller
hyperparameters =  {

    "pyteller.primitives.estimators.Persistence#1": {
        "lead": 11,
    }

}

pyteller = Pyteller (
hyperparameters = hyperparameters,
    pipeline = 'sandbox/dummy'
)

# TODO: split for supervised learning, should this be a primitive?

xtest=test
forecast = pyteller._mlpipeline.predict(xtest.loc[:, ['timestamp', 'target']])
pred_window=(test['timestamp']>= forecast['timestamp'].iloc[0] )&(test['timestamp']<= forecast['timestamp'].iloc[-1])
actual=test.loc[pred_window]

# TODO: group by entity
# grouped=test.groupby('entity')
# for x in grouped.groups:
#     xtest=grouped.get_group(x)
#     forecast = pyteller._mlpipeline.predict(xtest.loc[:, ['timestamp', 'target']])

scores = pyteller.evaluate(train_data= train,forecast=forecast,ground_truth=actual)
# mape = METRICS['MAPE'](ytest, forecast)








#
