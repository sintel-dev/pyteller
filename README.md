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

pyteller is a time series forecasting library built the end user.

## Table of Contents

* [I. Data Format](#data-format)
   * [I.1 Input](#input)
   * [I.2 Output](#output)
   * [I.3 Dataset we use in this library](#dataset-we-use-in-this-library)


# Data Format

## Input

The expected input to pyteller pipelines are .csv files of a **target table** which contains the history of what you want to predict for, and an optional **exogenous inputs** table of features that may help the prediction. They should be provided in the following formats:

### Targets Table
#### Option 1: Single Entity: Academic Form
The user must specify the following:
* `timestamp_column`: the **pandas timestamp** object or **python datetime** object represents the time at which the observation is made
* `target_column`: an **integer** or **float** column with the observed target values at the indicated timestamps

This is an example of such table, where the values are the number of NYC taxi passengers at the corresponding timestamp.

|  timestamp |     value |
|------------|-----------|
| 7/1/14 1:00 |  6210 |
| 7/1/14 1:30 | 4656|
| 7/1/14 2:00 | 3820 |
|7/1/14 1:30|	4656|
|7/1/14 2:00|	3820|
|7/1/14 2:30|	2873|
|7/1/14 3:00|	2369|
|7/1/14 3:30|	2064|
|7/1/14 4:00|	2221|
|7/1/14 4:30|	2158|
|7/1/14 5:00|	2515|

#### Option 2: Multiple Entity-Instances: Flatform
The user must specify the following:
* `timestamp_col`: the **pandas timestamp** object or **python datetime** object corresponding to the time at which the observation is made
* `entity_col`: the **string** denoting which column is the entity you will seperately make forecasts for
* `target`: the **string** denoting which column is the observed target values that you want to forecast for
* `dynamic_variable`: the **string** denoting which column are other input time series that will help the forecast
* `static_variable`: the **string** denoting which columns are a static varible

This is an example of such table, where the `timestamp_col` is 'timestamp', the `entity_col` is 'region',  the `target` is 'demand,' the  and `dynamic_variable` are 'Temp' and 'Rain':



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


#### Option 3: Multiple Entity-Instances: Longform
The user must specify the following:
* `timestamp_col`: the **pandas timestamp** object or **python datetime** object corresponding to the time at which the observation is made
* `entity_col`: the **string** denoting which column is the entity you will seperately make forecasts for
* `variable_col`: the **string** denoting which column is the observed variables
* `target`: the **string** denoting which variable name is the observed target values in the `variable_col` that you want to forecast for
* `dynamic_variable`: the **string** denoting which variable names are other input time series in the `variable_col` that will help the forecast
* `static_variable`: the **string** denoting which variable names are a static varible in the `variable_col`
* `value_col`: the **string** denoting which column contains the values of the observations of the `variable_col`

This is an example of such table, where the `timestamp_col` is 'timestamp', the `entity_col` is 'region', the `variable_col` is 'var_name', the `target` is 'demand,' the  `dynamic_variable` are 'Temp' and 'Rain', and the `value_col` is 'value':



 |  timestamp | region  |   var_name |   value |
|------------|------------|-----------| -----------|
 |9/27/20 21:20 |  DAYTON|demand | 1841.6|
 |9/27/20 21:20 | DAYTON|Temp | 65.78|
 |9/27/20 21:20 | DAYTON|Temp | 0|
 |9/27/20 21:20 |DEOK| demand | 2892.5|
 |9/27/20 21:20  | DEOK|Temp |75.92|
 |9/27/20 21:20  |DEOK| Rain |0|
 |9/27/20 21:20 | DOM|demand| 11276|
 |9/27/20 21:20 | DOM| Temp | 55.29|
 |9/27/20 21:20 |DOM| Rain| 0|
|9/27/20 21:20| DPL|demand| 2113.7|
 |9/27/20 21:20 | DPL| Temp | 75.02|
 |9/27/20 21:20 |DPL| Rain| 0.06|
 |9/27/20 21:25 |  DAYTON|demand | 1834.1|
 |9/27/20 21:25 | DAYTON|Temp | 65.72|
 |9/27/20 21:25 | DAYTON|Temp | 0|
 |9/27/20 21:25 |DEOK| demand | 2880.2|
 |9/27/20 21:25  | DEOK|Temp |75.92|
 |9/27/20 21:25  |DEOK| Rain |0|
 |9/27/20 21:25 | DOM|demand| 11211.7|
 |9/27/20 21:25 | DOM| Temp | 55.54|
 |9/27/20 21:25 |DOM| Rain| 0|
|9/27/20 21:25| DPL|demand| 2086.6|
 |9/27/20 21:25 | DPL| Temp | 75.02|
 |9/27/20 21:25 |DPL| Rain| 0.06|





## Output

The output of the pyteller Pipelines is another table that contains the timestamp and the forecasting value(s), matching the format of the input targets table.

## Datasets in the library

For development and evaluation of pipelines, we include the NYC taxi data that can be found [here](https://www1.nyc.gov/site/tlc/about/tlc-trip-record-data.page), or the processed version maintained by Numenta [here](https://github.com/numenta/NAB/tree/master/data). We also use PJM electricity demand data found [here](https://dataminer2.pjm.com/feed/inst_load).

### Data Loading

This formatted dataset can be browsed and downloaded directly from the
[d3-ai-orion AWS S3 Bucket](https://d3-ai-orion.s3.amazonaws.com/index.html).



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

# Quickstart

In this short tutorial we will guide you through a series of steps that will help you
getting started with **pyteller**.

## 1. Load the data

In the first step we will load the **electricity_demand** data from the **Demo Dataset**.



Import the `pyteller.data.load_signal` function and call it

```python3
from pyteller.data import load_signal

df = load_signal(
    data='electricity_demand.csv',
    timestamp_col = 'timestamp',
    entity_col='region',
    target = 'demand'
    dynamic_variables = ['Rain', 'Temp']
)
```




# What's next?

For more details about **pyteller** and all its possibilities
and features, please check the [documentation site](
https://signals-dev.github.io/pyteller/).
