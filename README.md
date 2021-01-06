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

Time series forecasting using MLPrimitives

- Documentation: https://signals-dev.github.io/pyteller
- Homepage: https://github.com/signals-dev/pyteller

# Overview

pyteller is a time series forecasting library built with the end user in mind.


## Table of Contents

* [I. Data Format](#data-format)
   * [I.1 Input](#input)
   * [I.2 Output](#output)
   * [I.3 Datasets in the library](#datasets-in-the-library)
* [II. pyteller Pipelines](#pyteller-pipelines)
   * [II.1 Current Available Pipelines](#current-available-pipelines)
* [III. Install](#install)
* [IV. Quick Start](#quick-start)


# Data Format

## Input

The expected input to pyteller pipelines is a .csv file with data in one of the following formats:

### Targets Table
#### Option 1: Single Entity (Academic Form)
The user must specify the following:
* `timestamp_col`: the **string** denoting which column contains the **pandas timestamp** objects or **python datetime** objects corresponding to the time at which the observation is made
* `target_signal`: an **integer** or **float** column with the observed target values at the indicated timestamps

This is an example of such table, where the `timestamp_col` is 'timestamp' and the `target_signal` is 'value'

|  timestamp |     value |
|------------|-----------|
| 7/1/14 1:00 |  6210 |
| 7/1/14 1:30 | 4656|
| 7/1/14 2:00 | 3820 |
|7/1/14 1:30|	4656|
|7/1/14 2:00|	3820|
|7/1/14 2:30|	2873|
|

#### Option 2: Multiple Entity (Flat Form)
The user must specify the following:
* `timestamp_col`: the **string** denoting which column contains the **pandas timestamp** objects or **python datetime** objects corresponding to the time at which the observation is made
* `entities`: the **list** denoting the columns the user wants to make forecasts for


This is an example of such table, where the `timestamp_col` is 'timestamp' and the `entities` can be ['taxi 1','taxi 3']

|  timestamp |     taxi 1 |     taxi 2 |    taxi 3 |
|------------|-----------|-----------| -----------|
| 7/1/14 1:00 |  6210 |  510 |  6230 |
| 7/1/14 1:30 | 4656| 5666|656|
| 7/1/14 2:00 | 3820 | 2420 | 3650 |
|7/1/14 1:30|	4656|	4664| 380 |
|7/1/14 2:00|	3820|	3520| 320 |
|7/1/14 2:30|	2873|	1373| 3640 |


#### Option 3: Multiple Entity (Long Form)
The user must specify the following:
* `timestamp_col`: the **string** denoting which column contains the **pandas timestamp** objects or **python datetime** objects corresponding to the time at which the observation is made
* `entity_col`: the **string** denoting which column contains the entities you will seperately make forecasts for
* `target_signal`: the **string** denoting which columns contain the observed target value that you want to forecast for


This is an example of such table, where the `timestamp_col` is 'timestamp', the `entity_col` is 'region', and the `target_signal` is 'demand'.



 |  timestamp | region  |   demand |   Temp |   Rain |
|------------|------------|-----------| -----------|-----------|
 9/27/20 21:20 |  DAYTON|1841.6 | 65.78|	0|
|  9/27/20 21:20 | DEOK|2892.5 |75.92|	0|
| 9/27/20 21:20|  DOM|11276 | 55.29|	0|
|9/27/20 21:20| DPL|2113.7| 75.02|	0.06|
| 9/27/20 21:25 | DAYTON|1834.1 | 65.72|	0|
| 9/27/20 21:25 |DEOK| 2880.2 | 75.92|	0|
| 9/27/20 21:25| DOM| 11211.7 | 55.54|	0|
|9/27/20 21:25|DPL| 2086.6| 75.02|	0.06|


## Output

The output of the pyteller Pipelines is another table that contains the timestamp and the forecasting value(s), matching the format of the input targets table.

## Datasets in the library

For development and evaluation of pipelines, we include the following datasets:
#### NYC taxi data
* Found on the [nyc website](https://www1.nyc.gov/site/tlc/about/tlc-trip-record-data.page), or the processed version maintained by Numenta [here](https://github.com/numenta/NAB/tree/master/data).
* No modifications were made from the Numenta version

#### Wind data
* Found here on [kaggle](https://www.kaggle.com/sohier/30-years-of-european-wind-generation/metadata)
* After downloading the FasTrak 5-Minute .txt files the .txt files for each day from 1/1/13-1/8/18 were compiled into one .csv file


#### Weather data
* Maintained by Iowa State University's [IEM](https://mesonet.agron.iastate.edu/request/download.phtml?network=ILASOS)
* The downloaded data was from the selected network of 8A0 Albertville and the selected date range was 1/1/16 0:15 - 2/16/16 0:55

#### Traffic data
* Found on [Caltrans PeMS](http://pems.dot.ca.gov/?dnode=Clearinghouse)
* No modifications were made from the Numenta version

#### Energy data
* Found on [kaggle](https://www.kaggle.com/robikscube/hourly-energy-consumption/metadata)
* No modifications were made after downloading pjm_hourly_est.csv
We also use PJM electricity demand data found [here](https://dataminer2.pjm.com/feed/inst_load).



## Current Available Pipelines

The pipelines are included as **JSON** files, which can be found
in the subdirectories inside the [pyteller/pipelines](orion/pipelines) folder.

This is the list of pipelines available so far, which will grow over time:

| name | location | description |
|------|----------|-------------|
| Persistence | [pyteller/pipelines/sandbox/persistence](../pipelines/sandbox/persistence) | uses the latest input to the model as the next output


# Install

## Requirements

**pyteller** has been developed and tested on [Python 3.5, 3.6, 3.7 and 3.8](https://www.python.org/downloads/)

Also, although it is not strictly required, the usage of a [virtualenv](https://virtualenv.pypa.io/en/latest/)
is highly recommended in order to avoid interfering with other software installed in the system
in which **pyteller** is run.

These are the minimum commands needed to create a virtualenv using python3.6 for **pyteller**:

```bash
pip install virtualenv
virtualenv -p $(which python3.6) pyteller-venv
```

Afterwards, you have to execute this command to activate the virtualenv:

```bash
source pyteller-venv/bin/activate
```

Remember to execute it every time you start a new console to work on **pyteller**!

<!-- Uncomment this section after releasing the package to PyPI for installation instructions
## Install from PyPI

After creating the virtualenv and activating it, we recommend using
[pip](https://pip.pypa.io/en/stable/) in order to install **pyteller**:

```bash
pip install pyteller
```

This will pull and install the latest stable release from [PyPI](https://pypi.org/).
-->

## Install from source

With your virtualenv activated, you can clone the repository and install it from
source by running `make install` on the `stable` branch:

```bash
git clone git@github.com:signals-dev/pyteller.git
cd pyteller
git checkout stable
make install
```

## Install for Development

If you want to contribute to the project, a few more steps are required to make the project ready
for development.

Please head to the [Contributing Guide](https://signals-dev.github.io/pyteller/contributing.html#get-started)
for more details about this process.

# Quick Start

In this short tutorial we will guide you through a series of steps that will help you
getting started with **pyteller**.

## 1. Load the data

In the first step we will load the **Alabama Weather** data into a dataframe from the demo datasets in the `data` folder. This represents all of the data up-to-date that will be used to train the model.

```python3
from pyteller.data import load_data
current_data=load_data('../pyteller/data/AL_Weather_current.csv')
```
The output is a dataframe:

```
    station     valid       tmpf        dwpf        relh        drct        sknt        p01i        alti      vsby        feel
0     8A0    1/1/16 0:15   41.000     39.200       93.240      350.000      6.000      0.000       30.250    10.000      36.670
1     4A6    1/1/16 0:15   41.000     39.000       70.080      360.000      5.000      0.000       30.300    10.000      37.080
2     8A0    1/1/16 0:35   39.200     37.400       93.190      360.000      6.000      0.000       30.250    10.000      34.200
3     4A6    1/1/16 0:35   41.000     32.000       70.080      360.000      5.000      0.000       30.290    10.000      37.080
4     8A0    1/1/16 0:55   37.400     37.400       100.000     360.000      8.000      0.000       30.250    10.000      30.760
```


Once we have the data, create an instance of the `Pyteller` class, where the input arguments are the forecast settings.

```python3
from pyteller.core import Pyteller
pyteller = Pyteller (
    pipeline='persistence',
    pred_length=3,
    offset=5,
)
```

## 2. Fit the data
The user now calls the `pyteller.fit` method to fit the data to the pipeline. The inputs are the loaded data and the column names. The user also specifies which signal or entities they want to predict for here.
```python3
pyteller.fit(
    data=current_data,
    timestamp_col='valid',
    target_signal='tmpf',
    entity_col='station')
```


## 3. Save the trained model
At this point, the user has a trained model that can be pickled by calling the `pyteller.save` method, inputting the desired output path:

```python3
pyteller.save('../fit_models/persistence')
```

## 4. Load the new data
Once the user gets new data that they want to use to make a prediction, they can load it in the same way they loaded the training data.
```python3
input_data=load_data('../pyteller/data/AL_Weather_input.csv')
 ```

 ## 5. Forecast
To make a forecast, the user calls the `pyteller.forecast` method, which will output the forecasts for all signals and all entities.

```python3
forecast = pyteller.forecast(input_data)
```
The output is a dataframe of all the predictions:

```python3
 timestamp        8A0            4A6
 2/4/16 18:15    42.800        44.800
 2/4/16 18:35    42.800        42.600
 2/4/16 18:55    44.800        43.000






# What's next?

For more details about **pyteller** and all its possibilities
and features, please check the [documentation site](
https://signals-dev.github.io/pyteller/).
