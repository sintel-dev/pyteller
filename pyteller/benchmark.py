import ast
import json
import logging
import os
from functools import partial
from copy import deepcopy
from datetime import datetime
from pathlib import Path
import uuid

import numpy as np
import pandas as pd
import tqdm
from mlblocks import MLPipeline
from pyteller.progress import TqdmLogger, progress
from mlblocks.discovery import find_pipelines,load_pipeline

from pyteller.core import Pyteller
from pyteller.metrics import METRICS
from pyteller.data import load_data

LOGGER = logging.getLogger(__name__)

run_super=True

BUCKET = 'pyteller'
S3_URL = 'https://{}.s3.amazonaws.com/{}'

if run_super==False:
    BENCHMARK_DATA = pd.read_csv(S3_URL.format(
        BUCKET, 'datasets.csv'), index_col=0, header=None).applymap(ast.literal_eval).to_dict()[1]
else:
    BENCHMARK_DATA = pd.read_csv('pyteller_benchmark/datasets.csv', index_col=0, header=None).iloc[:,0:1].applymap(ast.literal_eval).to_dict()[1]

META_DATA = pd.read_csv(S3_URL.format(
    BUCKET, 'data_s3.csv'), index_col=0, header=0)
META_DATA = META_DATA.loc[META_DATA.index.dropna()]
# BENCHMARK_PARAMS = pd.read_csv(S3_URL.format(
#     BUCKET, 'parameters.csv'), index_col=0, header=None).applymap(ast.literal_eval).to_dict()[1]
BENCHMARK_PARAMS = []
BENCHMARK_PATH = os.path.join(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..'),
    'benchmark'
)

pipelines=find_pipelines('pyteller')


def _summarize_results(scores, rank):
    scores = scores.groupby(['dataset', 'signal'])
    scores_summary = pd.DataFrame()
    for signal, df in scores:
        arima = df.loc[df['pipeline'] == 'pyteller.ARIMA.arima'][rank].values
        mask = df[rank] < float(arima)
        mask = mask.astype(int)
        df['Beat ARIMA'] = mask

        scores_summary = scores_summary.append(df)

    return pd.DataFrame(scores_summary.groupby('pipeline').sum()['Beat ARIMA'])

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


def _load_pipeline(pipeline, hyperparams=None):
    if isinstance(pipeline, str) and os.path.isfile(pipeline):
        pipeline = MLPipeline.load(pipeline)
    else:
        pipeline = MLPipeline(pipeline)

    if hyperparams is not None:
        pipeline.set_hyperparameters(hyperparams)

    return pipeline


def _get_pipeline_hyperparameter(hyperparameters, dataset_name, pipeline_name):
    hyperparameters_ = deepcopy(hyperparameters)

    if hyperparameters:
        hyperparameters_ = hyperparameters_.get(dataset_name) or hyperparameters_
        hyperparameters_ = hyperparameters_.get(pipeline_name) or hyperparameters_

    if hyperparameters_ is None and dataset_name and pipeline_name:
        file_path = os.path.join(
            PIPELINE_DIR, pipeline_name, pipeline_name + '_' + dataset_name.lower() + '.json')
        if os.path.exists(file_path):
            hyperparameters_ = file_path

    if isinstance(hyperparameters_, str) and os.path.exists(hyperparameters_):
        with open(hyperparameters_) as f:
            hyperparameters_ = json.load(f)

    return hyperparameters_

def _load_dataset(dataset):

    if run_super== True:
        dataset='pyteller_benchmark/' + dataset + '.csv'


    train, test = load_data(dataset)

    return train, test


def _evaluate_signal(pipeline_name, dataset, signal, pred_length, hyperparameters, metrics):
    train, test = _load_dataset(dataset)
    try:
        LOGGER.info("Scoring pipeline %s on signal %s",
                    pipeline_name, signal)
        start = datetime.utcnow()
        pipeline = load_pipeline(pipeline_name)
        pyteller = Pyteller(
            pipeline=pipeline,
            pred_length=pred_length,
            offset=0,
            targets=signal,
            hyperparameters=hyperparameters
        )

        pyteller.fit(train, tune=False)
        output = pyteller.forecast(data=test, postprocessing=False, predictions_only=False)

        # scores = pyteller.evaluate(actuals=output['actuals'], forecasts=output['forecasts'],
        #                            metrics=['MAPE', 'sMAPE'])
        elapsed = datetime.utcnow() - start
        scores = {
            name: scorer(output['actuals'], output['forecasts'])
            for name, scorer in metrics.items()
        }
        status = 'OK'

    except Exception as ex:
        LOGGER.exception("Exception scoring pipeline %s on signal %s error %s.",
                         pipeline, signal, ex)
        elapsed = datetime.utcnow() - start
        scores = {
            name: 0 for name in metrics.keys()
        }

        status = 'ERROR'

    scores['status'] = status
    scores['elapsed'] = elapsed.total_seconds()


#
    return scores

def _run_job(args):
    # Reset random seed
    np.random.seed()

    (pipeline, pipeline_name, dataset, signal, pred_length, hyperparameter, metrics,
        iteration, cache_dir, pipeline_dir, run_id) = args

    pipeline_path = pipeline_dir
    if pipeline_dir:
        base_path = str(pipeline_dir / f'{pipeline_name}_{signal}_{dataset}_{iteration}')
        pipeline_path = base_path + '_pipeline.pkl'

    LOGGER.info('Evaluating pipeline %s on signal %s dataset %s (test split: %s); iteration %s',
                pipeline_name, signal, dataset, iteration)

    output = _evaluate_signal(
        pipeline,
        dataset,
        signal,
        pred_length,
        hyperparameter,
        metrics
    )
    scores = pd.DataFrame.from_records([output], columns=output.keys())

    scores.insert(0, 'dataset', dataset)
    scores.insert(1, 'pipeline', pipeline_name)
    scores.insert(2, 'signal', signal)
    scores.insert(3, 'prediction length', pred_length)
    scores.insert(4, 'iteration', iteration)
    scores['run_id'] = run_id

    if cache_dir:
        base_path = str(cache_dir / f'{pipeline_name}_{signal}_{dataset}_{pred_length}_{iteration}_{run_id}')
        scores.to_csv(base_path + '_scores.csv', index=False)

    return scores


def _run_on_dask(jobs, verbose):
    """Run the tasks in parallel using dask."""
    try:
        import dask
    except ImportError as ie:
        ie.msg += (
            '\n\nIt seems like `dask` is not installed.\n'
            'Please install `dask` and `distributed` using:\n'
            '\n    pip install dask distributed'
        )
        raise

    scorer = dask.delayed(_run_job)
    persisted = dask.persist(*[scorer(args) for args in jobs])
    if verbose:
        try:
            progress(persisted)
        except ValueError:
            pass

    return dask.compute(*persisted)

def benchmark(pipelines=None, datasets=None, pred_length=12, hyperparameters=None,
              metrics=METRICS, rank='MAPE', iterations=1, workers=1, show_progress=False,
              cache_dir=None, output_path=None, pipeline_dir=None):

    """Run pipelines on the multiple datasets and evaluate the performance.

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
        pred_length (int or list): The forecasting horizon that will be used in the benchmark.
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
    datasets = datasets or BENCHMARK_DATA
    #For testing
    import itertools
    datasets = dict(itertools.islice(datasets.items(), 2))

    run_id = os.getenv('RUN_ID') or str(uuid.uuid4())[:10]

    if isinstance(pipelines, list):
        pipelines = {pipeline: pipeline for pipeline in pipelines}

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

    if cache_dir:
        cache_dir = Path(cache_dir)
        os.makedirs(cache_dir, exist_ok=True)

    if pipeline_dir:
        pipeline_dir = Path(pipeline_dir)
        os.makedirs(pipeline_dir, exist_ok=True)

    jobs = list()
    for dataset, signals in datasets.items():
        if isinstance(signals,str):
            signals = [signals]
        for pipeline_name, pipeline in pipelines.items():
            hyperparameter = _get_pipeline_hyperparameter(hyperparameters, dataset, pipeline_name)

            for signal in signals:
                for iteration in range(iterations):
                    args = (
                        pipeline,
                        pipeline_name,
                        dataset,
                        signal,
                        pred_length,
                        hyperparameter,
                        metrics,
                        iteration,
                        cache_dir,
                        pipeline_dir,
                        run_id,
                    )
                    jobs.append(args)

    if workers == 'dask':
        scores = _run_on_dask(jobs, show_progress)
    else:
        if workers in (0, 1):
            scores = map(_run_job, jobs)
        else:
            pool = concurrent.futures.ProcessPoolExecutor(workers)
            scores = pool.map(_run_job, jobs)

        scores = tqdm.tqdm(scores, total=len(jobs), file=TqdmLogger())
        if show_progress:
            scores = tqdm.tqdm(scores, total=len(jobs))

    scores = pd.concat(scores)
    if output_path:
        LOGGER.info('Saving benchmark report to %s', output_path)
        scores.to_csv(output_path+'.csv', index=False)
        summary_scores = _summarize_results(scores, rank)
        summary_scores.reset_index(inplace=False)
        summary_scores.to_csv(output_path + '_summary.csv')

    return _sort_leaderboard(scores, rank, metrics), summary_score




def main(workers=1):
    # output path
    pipeline_dir = 'save_pipelines'
    cache_dir = 'cache'
    version = "results"
    output_path = os.path.join(BENCHMARK_PATH, 'results', version)

    # metrics
    metrics = {k: partial(fun) for k, fun in METRICS.items()}

    # pipelines

    pipelines= find_pipelines('pyteller')

    hyperparameters = {
        'a10':{
            'pyteller.ARIMA.arima':{
                'pyteller.primitives.preprocessing.format_data#1': {
                'make_index': True
            },
            'pyteller.primitives.postprocessing.flatten#1': {
                    'type': 'average'
                }
            },
            'pyteller.LSTM.LSTM': {
                'pyteller.primitives.preprocessing.format_data#1': {
                    'make_index': True
                },
            'pyteller.primitives.postprocessing.flatten#1': {
                    'type': 'average'
                }
            },
            'pyteller.persistence.persistence': {
                'pyteller.primitives.preprocessing.format_data#1': {
                    'make_index': True
                },
            'pyteller.primitives.postprocessing.flatten#1': {
                    'type': 'average'
                }
            }
        }
    }
    results = benchmark(pipelines=pipelines, hyperparameters=hyperparameters,metrics=metrics,
        output_path=output_path, workers='dask', show_progress=True,
         pipeline_dir=pipeline_dir, cache_dir=cache_dir)

    # results = benchmark(pipelines=pipelines, hyperparameters=hyperparameters,metrics=metrics,
    #     output_path=output_path, workers=1, show_progress=True,
    #      pipeline_dir=pipeline_dir, cache_dir=cache_dir)


if __name__ == "__main__":
    results= main()
