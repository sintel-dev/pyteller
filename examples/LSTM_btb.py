import numpy as np
from mlblocks import MLPipeline, load_pipeline
from mlprimitives.datasets import Dataset
from sklearn.metrics import mean_absolute_error

from btb.tuning import GPTuner, Tunable
from pyteller.core import Pyteller
from pyteller.data import load_data

# Choose pipeline
pipeline_name = 'pyteller/pipelines/pyteller/LSTM/LSTM_offset.json'

# Load data and  create a dataset instance
current_data, input_data = load_data('taxi')
X, y = current_data, current_data
dataset = Dataset('data', X, y, mean_absolute_error, 'timeseries', 'forecast', shuffle=False)


def cross_validate(hyperparameters=None):
    scores = []
    for X_train, X_test, y_train, y_test in dataset.get_splits(3):
        global pipeline
        pipeline = MLPipeline(pipeline)

        if hyperparameters:
            pipeline.set_hyperparameters(hyperparameters)

        # only take the X data, the y data is made within the pipeline
        pyteller, all_output = run_pipeline(X_train, X_test, pipeline)
        scores.append(pyteller.evaluate(test_data=all_output['actual'], forecast=all_output['forecast'],
                                        metrics=['sMAPE']).values[0][0])
    return np.mean(scores)


def run_pipeline(X_train, X_test, pipeline):

    pyteller = Pyteller(
        pipeline=pipeline,
        pred_length=5,
        offset=2,
        timestamp_col='timestamp',
        target_signal='value'
    )

    pyteller.fit(X_train)
    output = pyteller.forecast(data=X_test, postprocessing=False,
                               predictions_only=False)

    return pyteller, output


pipeline = MLPipeline(load_pipeline(pipeline_name))
tunable_hyperparameters = pipeline.get_tunable_hyperparameters(flat=True)

tunable = Tunable.from_dict(tunable_hyperparameters)

default_score = cross_validate()

tuner = GPTuner(tunable)
defaults = tunable.get_defaults()
tuner.record(defaults, default_score)
best_score = default_score
best_proposal = defaults

for iteration in range(3):
    print("scoring pipeline {}".format(iteration + 1))
    proposal = tuner.propose()
    score = cross_validate(proposal)
    tuner.record(proposal, score)

    if score < best_score:
        print("New best found: {}".format(score))
        best_score = score
        best_proposal = proposal

pipeline = MLPipeline(pipeline)
pipeline.set_hyperparameters(best_proposal)

pyteller, all_output = run_pipeline(current_data, input_data, pipeline)

scores = pyteller.evaluate(test_data=all_output['actual'], forecast=all_output['forecast'],
                           metrics=['MAPE', 'sMAPE'])
