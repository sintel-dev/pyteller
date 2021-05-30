.. _benchmarking:


============
Benchmarking
============

We develop a benchmarking procedure in pyteller in order to compare multiple pipelines against each other across many datasets and signals within the datasets.


Releases
--------


In every release, we run the pyteller benchmark and maintain an up to-date [leaderboard](../README.md###leaderboard).
Results obtained during the benchmarking process and in previous benchmarks can be found
within `benchmark/results`_ folder as CSV files and, you can find it in the `details Google Sheets document`_.
We summarize the results in the `summary Google Sheets document`_.


Process
-------

In our terminology, one dataset may have many *signals* that are forecasted for.

A leaderboard entry for one *pipeline* across the many datasets is created using the following steps:

1. Split each dataset into train and test data
2. Use the pipeline to forecast on the default settings on the testing data
3. Evaluate the normalized metrics for each signal in the test dataset
4. For each metric, average the scores of all the entities in the signal and all the signals in a dataset
5. For each metric, average the scores of all the datasets

Finally, repeat this process for all pipelines, and rank the pipelines by sorting them by the specified metrics to rank by.

Benchmark function
~~~~~~~~~~~~~~~~~~




.. _benchmark/results: https://github.com/signals-dev/pyteller/tree/master/benchmark/results
.. _details Google Sheets document: https://docs.google.com/spreadsheets/d/1EQd2x4BPSYEs6KLLUKrxzY3e8TuysnYnaSYAsBiPwCA/edit?usp=sharing
.. _summary Google Sheets document: https://docs.google.com/spreadsheets/d/1OPwAslqfpWvzpUgiGoeEq-Wk_yK-GYPGpmS7TwEaSbw/edit?usp=sharing
