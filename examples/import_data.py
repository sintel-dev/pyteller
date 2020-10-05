from pyteller.data import load_signal

# signal = 'nyc_taxi'
signal = 'electricity_demand.csv'

# load signal
df = load_signal(
    data='signal',
    timestamp_column = 'datetime_beginning_utc',
    entity_column='area',
    target_column = 'instantaneous_load'
)

#Example
truth = [1, 1, 1, 0, 0, 0]
detected = [0, 1, 1, 1, 0, 0]

from pyteller.evaluation import METRICS
mape=METRICS['MAPE'](truth,detected)

from pyteller.pyteller import Pyteller
hyperparameters =  {
    "pyteller.primitives.estimators.Persistence#1": {
        "lead": 11,
    },

}

pyteller = Pyteller (
hyperparameters = hyperparameters,
    pipeline = 'dummy.json'
)

pyteller._mlpipeline.fit(df)
anomalies = pyteller._mlpipeline.predict(df)
