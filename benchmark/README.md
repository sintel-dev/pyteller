
# Benchmarking

This document explains the benchmarking procedure we develop in pyteller in order to evaluate how accurate
a pipeline forecasts.

## Releases

In every release, we run the pyteller benchmark and maintain an up to-date [leaderboard](../README.md###leaderboard).
Results obtained during the benchmarking process and in previous benchmarks can be found
within [benchmark/results](results) folder as CSV files and, you can find it in the [details Google Sheets document](https://docs.google.com/spreadsheets/d/1EQd2x4BPSYEs6KLLUKrxzY3e8TuysnYnaSYAsBiPwCA/edit?usp=sharing).
We summarize the results in the [leaderboard](benchmark/leaderboard.md) table.

## Evaluating the Pipelines

Using the [Evaluation sub-package](../pyteller/evaluation). In our terminology, one dataset may have many *signals* that are forecasted for.

A leaderboard entry for one *pipeline* across the many datasets is created using the following steps:

1. Split each dataset into train and test data
2. Use the pipeline to forecast on the default settings on the testing data
3. Evaluate the normalized metrics for each signal in the test dataset
4. For each metric, average the scores of all the entities in the signal and all the signals in a dataset
5. For each metric, average the scores of all the datasets

Finally, repeat this process for all pipelines, and rank the pipelines by sorting them by the specified metrics to rank by.

## Benchmark function
This function expects the following inputs:
* pipelines (dict or list): dictionary with pipeline names as keys and their JSON paths as values. If a list is given, it should be of JSON paths, and the paths themselves will be used as names. If not given, all verified pipelines will be used for evaluation.
* datasets (dict or list): dictionary of dataset name as keys and list of signals as values. If a list is given then all signals for each dataset will be forecasted. If not given, all benchmark datasets will be used.
* hyperparameters (dict or list): dictionary with pipeline names as keys and their hyperparameter JSON paths or dictionaries as values. If a list is given, it should be of corresponding order to pipelines.
* metrics (dict or list): dictionary with metric names as keys and scoring functions as values. If a list is given, it should be of scoring functions. If not given, all the available metrics will be used.
* rank (str): Sort and rank the pipelines based on the given metric. If not given, rank using the first metric.
* output_path (str): Location to save the results. If not given, results will not be saved.

This returns a `pandas.DataFrame` which contains the metrics obtained across all the signals for each dataset for a given pipeline.

This is an example call of the benchmarking function:
```
In [1]: from pyteller.benchmark import benchmark

In [2]: pipelines = [
                'Persistence'
   ...: ]

In [3]: metrics = ['MAPE', 'MASE', 'sMAPE']

In [4]: datasets = ['taxi', 'AL_wind', 'FasTrak']

In [5]: scores = benchmark(pipelines=pipelines, datasets=datasets, metrics=metrics, rank='MAPE')

