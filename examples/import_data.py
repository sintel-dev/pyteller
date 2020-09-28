from pyteller.data import load_signal

# signal = 'nyc_taxi'
signal = 'electricity_demand.csv'

# load signal
df = load_signal(signal)

#Example
truth = [1, 1, 1, 0, 0, 0]
detected = [0, 1, 1, 1, 0, 0]

from pyteller.evaluation import METRICS
mape=METRICS['MAPE'](truth,detected)

from pyteller.pyteller import Pyteller

pyteller = Pyteller (
    pipeline = 'dummy.json'

)
