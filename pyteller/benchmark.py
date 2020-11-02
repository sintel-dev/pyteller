import os
import json
import ast
from pyteller.evaluation import METRICS_NORM as METRICS
from functools import partial
import pandas as pd
import logging
from pyteller.analysis import _load_pipeline
from pyteller.data import load_signal
from pyteller.core import Pyteller
LOGGER = logging.getLogger(__name__)

PIPELINE_DIR = os.path.join(os.path.dirname(__file__), 'pipelines', 'verified')

BUCKET = 'pyteller'
S3_URL = 'https://{}.s3.amazonaws.com/{}'

BENCHMARK_DATA = pd.read_csv(S3_URL.format(
    BUCKET, 'datasets.csv'), index_col=0, header=None).applymap(ast.literal_eval).to_dict()[1]
META_DATA = pd.read_csv(S3_URL.format(
    BUCKET, 'data_s3.csv'), index_col=0, header=0)
META_DATA=META_DATA.loc[META_DATA.index.dropna()]
# BENCHMARK_PARAMS = pd.read_csv(S3_URL.format(
#     BUCKET, 'parameters.csv'), index_col=0, header=None).applymap(ast.literal_eval).to_dict()[1]
BENCHMARK_PARAMS = []
BENCHMARK_PATH = os.path.join(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..'),
    'benchmark'
)

VERIFIED_PIPELINES = [
    'persistence'
]


def _sort_leaderboard(df, rank, metrics):
    if rank not in df.columns:
        rank_ = list(metrics.keys())[0]
        LOGGER.exception("Rank %s is not in %s, using %s instead.",
                         rank, df.columns, rank_)
        rank = rank_

    df.sort_values(rank, ascending=True, inplace=True)
    df.reset_index(drop=True, inplace=True)
    df.index.name = 'rank'
    df.reset_index(drop=False, inplace=True)
    df['rank'] += 1

    return df.set_index('pipeline').reset_index()


def _load_signal(dataset, columns, holdout):
    # TODO: make this a spread sheet in s3 bucket and make it so the user can input signal
    # meta_path = os.path.join(os.path.dirname(__file__), 'data', dataset + '.json')
    # with open(meta_path) as f:
    #     columns = json.load(f)
    #columns = META_DATA.loc[dataset].to_dict()

    train, test = load_signal(dataset, column_dict=columns.to_dict())

    return train, test


def _evaluate_signal(pipeline, name, dataset, columns, hyperparameter, metrics,
                     holdout=True, detrend=False):
    train, test = _load_signal(dataset, columns, holdout)
    pipeline = _load_pipeline(pipeline, hyperparameter)
    pyteller = Pyteller(
        pipeline=pipeline,
        pred_length=3
    )

    forecast = pyteller.predict(test)
    # pred_window = (test['timestamp'] >= forecast['timestamp'].iloc[0]) & (
    #         test['timestamp'] <= forecast['timestamp'].iloc[-1])
    # actual=test.loc[pred_window]
    # scores = {}
    # if 'MASE' in metrics:
    #     scores['MASE'] = metrics['MASE'](train['target'],forecast['target'],actual['target'])
    #
    # scores.update( {
    #     name: scorer(forecast['target'],actual['target'])
    #     for name, scorer in metrics.items() if name != 'MASE'
    # })

    scores = pyteller.evaluate(train_data=train, test_data=test, forecast=forecast, detailed=True)
    scores['pipeline'] = name
    # scores['holdout'] = holdout
    scores['dataset'] = dataset
    # scores['signal'] = signal
    # scores['prediction length'] = forecast.shape[0]
    # scores['length of training data'] = len(train)
    # scores['length of testing data'] = len(test)

    return scores


def _evaluate_pipeline(pipeline, pipeline_name, dataset, columns, hyperparameter, metrics,
                       distributed, holdout, detrend):
    if holdout is None:
        holdout = (True, False)
    elif not isinstance(holdout, tuple):
        holdout = (holdout, )

    # if hyperparameter is None:
    #     file_path = os.path.join(
    #         PIPELINE_DIR, pipeline_name, pipeline_name + '_' + dataset.lower() + '.json')
    #     if os.path.exists(file_path):
    #         hyperparameter = file_path

    if isinstance(hyperparameter, str) and os.path.exists(hyperparameter):
        LOGGER.info("Loading hyperparameter %s", hyperparameter)
        with open(hyperparameter) as f:
            hyperparameter = json.load(f)

    if distributed:
        function = dask.delayed(_evaluate_signal)
    else:
        function = _evaluate_signal

    scores = list()
#TODO Fix
    if isinstance(columns, str):
        for holdout_ in holdout:
            score = function(pipeline, pipeline_name, dataset, columns, hyperparameter,
                             metrics, holdout_, detrend)
            scores.append(score)
    else:

        for holdout_ in holdout:
            score = function(pipeline, pipeline_name, dataset, columns, hyperparameter,
                             metrics, holdout_, detrend)

        scores.append(score)

    return scores


def _evaluate_pipelines(pipelines, dataset, columns, hyperparameters, metrics, distributed,
                        holdout, detrend):

    scores = list()
    for name, pipeline in pipelines.items():
        hyperparameter = _get_parameter(hyperparameters, name)
        score = _evaluate_pipeline(pipeline, name, dataset, columns, hyperparameter,
                                   metrics, distributed, holdout, detrend)
        scores.extend(score)

    return scores


def _get_parameter(parameters, name):
    if isinstance(parameters, dict) and name in parameters.keys():
        return parameters[name]

    return None


def _evaluate_datasets(pipelines, datasets, hyperparameters, metrics, distributed, holdout,
                       detrend):
    delayed = []

    # for dataset, signals in datasets.items():
    for dataset, columns in datasets.iterrows():
        LOGGER.info("Starting dataset {} with {} signals..".format(
            dataset, len(columns)))




        # dataset configuration
        hyperparameters_ = _get_parameter(hyperparameters, dataset)
        parameters = _get_parameter(BENCHMARK_PARAMS, dataset)
        if parameters is not None:
            detrend, holdout = parameters.values()

        result = _evaluate_pipelines(
            pipelines, dataset, columns, hyperparameters_, metrics, distributed, holdout, detrend)

        delayed.extend(result)

    if distributed:
        persisted = dask.persist(*delayed)
        results = dask.compute(*persisted)

    else:
        results = delayed

    df = pd.concat(results)
    # return results[0]
    return df


def benchmark(pipelines=None, datasets=None, hyperparameters=None, metrics=METRICS, rank='MAPE',
              distributed=False, holdout=False, detrend=False, output_path=None):
    """Evaluate pipelines on the given datasets and evaluate the performance.
    The pipelines are used to analyze the given signals and later on the
    detected anomalies are scored against the known anomalies using the
    indicated metrics.
    Finally, the scores obtained with each metric are averaged accross all the signals,
    ranked by the indicated metric and returned on a ``pandas.DataFrame``.
    Args:
        pipelines (dict or list): dictionary with pipeline names as keys and their
            JSON paths as values. If a list is given, it should be of JSON paths,
            and the paths themselves will be used as names. If not give, all verified
            pipelines will be used for evaluation.
        datasets (dict or list): dictionary of dataset name as keys and list of signals as
            values. If a list is given then it will be under a generic name ``dataset``.
            If not given, all benchmark datasets will be used used.
        hyperparameters (dict or list): dictionary with pipeline names as keys
            and their hyperparameter JSON paths or dictionaries as values. If a list is
            given, it should be of corresponding order to pipelines.
        metrics (dict or list): dictionary with metric names as keys and
            scoring functions as values. If a list is given, it should be of scoring
            functions, and they ``__name__`` value will be used as the metric name.
            If not given, all the available metrics will be used.
        rank (str): Sort and rank the pipelines based on the given metric.
            If not given, rank using the first metric.
        distributed (bool): Whether to use dask for distributed computing. If not given,
            use ``False``.
        holdout (bool): Whether to use the prespecified train-test split. If not given,
            use ``False``.
        detrend (bool): Whether to use ``scipy.detrend``. If not given, use ``False``.
        output_path (str): Location to save the intermediatry results. If not given,
            intermediatry results will not be saved.
    Returns:
        pandas.DataFrame: Table containing the average of the scores obtained with
            each scoring function accross all the signals for each pipeline, ranked
            by the indicated metric.
    """
    pipelines = pipelines or VERIFIED_PIPELINES
    datasets = datasets or META_DATA#BENCHMARK_DATA

    if isinstance(pipelines, list):
        pipelines = {pipeline: pipeline for pipeline in pipelines}

    if isinstance(datasets, dict):
        drop_these = [x for x in list(META_DATA.index) if x not in list(datasets.keys())]
        datasets2=META_DATA.drop(drop_these)
        for index, value in datasets2['signals'].items():
            if ',' in value:
                keep_signals =','.join(datasets[index])
                datasets2.loc[index]['signals'] = keep_signals
        datasets=datasets2


    if isinstance(hyperparameters, list):
        hyperparameters = {pipeline: hyperparameter for pipeline, hyperparameter in
                           zip(pipelines.keys(), hyperparameters)}

    if isinstance(metrics, list):
        metrics_ = dict()
        for metric in metrics:
            if callable(metric):
                metrics_[metric.__name__] = metric
            elif metric in METRICS:
                metrics_[metric] = METRICS[metric]
            else:
                raise ValueError('Unknown metric: {}'.format(metric))

        metrics = metrics_

    results = _evaluate_datasets(
        pipelines, datasets, hyperparameters, metrics, distributed, holdout, detrend)

    if output_path:
        LOGGER.info('Saving benchmark report to %s', output_path)
        results.to_csv(output_path)

    return _sort_leaderboard(results, rank, metrics)


def main():
    # output path
    version = "results.csv"
    output_path = os.path.join(BENCHMARK_PATH, 'results', version)

    # metrics
    metrics = {k: partial(fun) for k, fun in METRICS.items()}

    # pipelines
    pipelines = VERIFIED_PIPELINES  # TODO : check if pipeline user inputs work
    # datasets = {
    #     'taxi':'value',
    #     'AL_Weather':'tmpf',
    #     'Wind': 'AT',
    #     'pjm_hourly_est' : 'AEP',
    #     'FasTrak' : 'Total Flow'
    # }

    results = benchmark(
        pipelines=pipelines, metrics=metrics, output_path=output_path)
    # TODO: summarize results for who wins compared to latest baseline
    output_path = os.path.join(BENCHMARK_PATH, 'leaderboard.csv')
    results.to_csv(output_path)

    # leaderboard = _summarize_results(results, metrics)
    # leaderboard.to_csv(output_path)


if __name__ == "__main__":
    main()
