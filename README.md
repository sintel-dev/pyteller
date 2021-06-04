<p align="left">
<img width=15% src="https://dai.lids.mit.edu/wp-content/uploads/2018/06/Logo_DAI_highres.png" alt=“DAI-Lab” />
<i>An open source project from Data to AI Lab at MIT.</i>
</p>



<!-- Uncomment these lines after releasing the package to PyPI for version and downloads badges -->
<!--[![PyPI Shield](https://img.shields.io/pypi/v/pyteller.svg)](https://pypi.python.org/pypi/pyteller)-->
<!--[![Downloads](https://pepy.tech/badge/pyteller)](https://pepy.tech/project/pyteller)-->
[![Github Actions Shield](https://img.shields.io/github/workflow/status/signals-dev/pyteller/Run%20Tests)](https://github.com/signals-dev/pyteller/actions)
[![Coverage Status](https://codecov.io/gh/signals-dev/pyteller/branch/master/graph/badge.svg)](https://codecov.io/gh/signals-dev/pyteller)



# pyteller



- Documentation: https://signals-dev.github.io/pyteller
- Homepage: https://github.com/signals-dev/pyteller

# Overview

Pyteller is a time series forecasting library using MLPrimitives to build easy to use forecasting pipelines.



# Quickstart


## Install with pip

The easiest and recommended way to install **pyteller** is using [pip](https://pip.pypa.io/en/stable/):

```bash
pip install pyteller
```

This will pull and install the latest stable release from [PyPi](https://pypi.org/).

## 1. Input data
The expected input to pyteller pipelines is a .csv file with target data.

Depending on the format of the data, the user should specify the **string** denoting which columns contains the:

* `time_column`: Column denoting the timestamp column.
* `target_column`: Column denoting the target column.
* `targets`: List of the subset of targets to extract.
* `entity_column`: Column denoting the entities column.
* `entities`: Subset of entities to extract.

Here is an example of loading the [Alabama Weather](pyteller/data/ALWeather.csv) demo data which has multiple entities in long form:

```python3
from pyteller.data import load_data
current_data, input_data = load_data('AL_Weather')
```
`current_data` will be used to fit the pipeline and `input_data` to forecast. Both are dataframes:

| station | valid       | tmpf | dwpf | relh  | drct |
| ------- | ----------- | ---- | ---- | ----- | ---- |
| 8A0     | 1/1/16 0:15 | 41   | 39.2 | 93.24 | 350  |
| 4A6     | 1/1/16 0:15 | 41   | 32   | 70.08 | 360  |
| 8A0     | 1/1/16 0:35 | 39.2 | 37.4 | 93.19 | 360  |
| 4A6     | 1/1/16 0:35 | 41   | 32   | 70.08 | 360  |
| 8A0     | 1/1/16 0:55 | 37.4 | 37.4 | 100   | 360  |
| 4A6     | 1/1/16 0:55 | 39.2 | 32   | 75.16 | 350  |


## 2. Fit the pipeline
Once we have the data, create an instance of the `Pyteller` class, where the input arguments are the forecast settings and the column headers of the data.
In this example we use the `lstm` pipeline and set the training epochs to 20.

```python3
from pyteller.core import Pyteller

pipeline = 'pyteller/pipelines/pyteller/LSTM/LSTM.json'

hyperparameters = {
    'keras.Sequential.LSTMTimeSeriesRegressor#1': {
        'epochs': 20
    }
}

pyteller = Pyteller(
    pipeline=pipeline,
    time_column='valid',
    targets='tmpf',
    entity_column='station',
    entities='8A0'
    pred_length=12,
    offset=0,
    hyperparameters=hyperparameters
)

pyteller.fit(current_data)

```


 ## 3. Forecast
To make a forecast, the user calls the `pyteller.forecast` method

```python3
output = pyteller.forecast(data=input_data)
```
The output is a ``dictionary`` which includes the ``forecasts`` and ``actuals`` ``dataframes``. Here is ``output['forecasts']``:

```python3
 timestamp        8A0
 2/4/16 18:15    42.800
 2/4/16 18:35    42.800
 2/4/16 18:55    44.800
```

 ## 4. Evaluate
To see metrics of the forecast accuracy, the user calls the `pyteller.evaluate` method:
```python3
scores = pyteller.evaluate(test_data=output['actuals'],forecast=output['forecast'],
                           metrics=['sMAPE','MAPE'])

```
The output is a dataframe of the scores:

```python3
           8A0
 sMAPE     11.4
 MAPE      11.7
```




## Releases
In every release, we run a pyteller benchmark. We maintain an up-to-date leaderboard with the current scoring to the benchmarking procedure explained [here](benchmark).

Results obtained during the benchmarking process as well as previous benchmarks can be found
within [benchmark/results](benchmark/results) folder as CSV files. In addition, you can find it in the [details Google Sheets document](https://docs.google.com/spreadsheets/d/1EQd2x4BPSYEs6KLLUKrxzY3e8TuysnYnaSYAsBiPwCA/edit?usp=sharing).

### Leaderboard
We summarize the results in the [leaderboard](benchmark/leaderboard.md) table. We showcase the number of wins each pipeline has over the ARIMA pipeline.

The summarized results can also be browsed in the following [summary Google Sheets document](https://docs.google.com/spreadsheets/d/1OPwAslqfpWvzpUgiGoeEq-Wk_yK-GYPGpmS7TwEaSbw/edit?usp=sharing).


# What's next?

For more details about **pyteller** and all its possibilities
and features, please check the [documentation site](
https://signals-dev.github.io/pyteller/).

