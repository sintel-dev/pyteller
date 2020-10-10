from pyteller.data import load_signal


# signal = 'electricity_demand.csv'
signal = 'AL_Weather'
# load signal
# signal = 'nyc_taxi'
# df = load_signal(signal)
# df = load_signal(
#     data='signal'

# )

train,test = load_signal(
    data=signal,
    timestamp_col = 'valid',
    target = 'tmpf'
)


#Example
# truth = [1, 1, 1, 0, 0, 0]
# detected = [0, 1, 1, 1, 0, 0]

from pyteller.core import Pyteller
hyperparameters =  {

    "pyteller.primitives.estimators.Persistence#1": {
        "lead": 11,
    }

}

pyteller = Pyteller (
hyperparameters = hyperparameters,
    pipeline = 'sandbox/dummy3'
)

# from pyteller.primitives.estimators import split_sequence
# test = test.loc[:,['timestamp','target']]
# X= split_sequence(test)

# pyteller._mlpipeline.fit(train.loc[:,['timestamp','target']])
# forecast = pyteller._mlpipeline.predict(test.loc[:,['timestamp']],test.loc[:,['target']])
# TODO make this into a primitive
lead= hyperparameters["pyteller.primitives.estimators.Persistence#1"]['lead']
xtest=test
forecast = pyteller._mlpipeline.predict(xtest.loc[:, ['timestamp', 'target']])
# TODO index by time to find the actual value
actual=test[-lead:]
# grouped=test.groupby('entity')
# for x in grouped.groups:
#     xtest=grouped.get_group(x)
#     forecast = pyteller._mlpipeline.predict(xtest.loc[:, ['timestamp', 'target']])
#
#     ytest = test.shift(lead)
scores = pyteller.evaluate(forecast,actual)
# mape = METRICS['MAPE'](ytest, forecast)

# TODO make a primitive for splitting data, in this case, just shifting it







#
