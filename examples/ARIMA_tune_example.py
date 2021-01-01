# Load the dataset from a dataset on the s3 bucket, or in this example the local file path
from pyteller.data import load_data

def score_func(hyperparameters=None):


    from mlblocks import MLPipeline


    pyteller = Pyteller(
        pipeline=pipeline,
        pred_length=5,
        offset=3
    )

    pyteller.fit(
        data=current_data,
        timestamp_col='timestamp',
        target_signal='value',
        # static_variables=None,
        # entity_col='station',
        # entities='8A0'
        )

    input_data=load_data('pyteller/data/taxi.csv')
    # input_data=load_data('pyteller/data/AL_Weather_input.csv').iloc[0:1000]
    y,y_hat = pyteller.forecast(data=input_data)

    return pyteller.evaluate(forecast=y_hat, test_data=y, metrics=['MAPE']).values.item()


current_data = load_data('pyteller/data/taxi_test.csv')
# current_data = load_data('pyteller/data/AL_Weather_current.csv').iloc[0:40000]
from pyteller.core import Pyteller
pipeline = 'pyteller/pipelines/sandbox/ARIMA/arima.json'
default = score_func()
from mlblocks import MLPipeline
template = MLPipeline.load(pipeline)
tunable_hyperparameters = template.get_tunable_hyperparameters(flat=True)
from btb.tuning import Tunable

tunable = Tunable.from_dict(tunable_hyperparameters)
from btb.tuning import GPTuner

tuner = GPTuner(tunable)
defaults = tunable.get_defaults()
tuner.record(defaults, default)

best_score = default
best_proposal = defaults

for iteration in range(10):
    print("scoring pipeline {}".format(iteration + 1))

    proposal = tuner.propose()
    score = score_func(proposal)

    tuner.record(proposal, score)

    if score > best_score:
        print("New best found: {}".format(score))
        best_score = score
        best_proposal = proposal
