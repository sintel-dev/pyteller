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

The expected input to pyteller pipelines are .csv files of time series that are provided in the following formats:

### Targets Table
#### Option 1: Single Entity
* `timestamp`: the **pandas timestamp** object or **python datetime** object represents the time at which the observation is made
* `value`: an **integer** or **float** column with the observed target values at the indicated timestamps

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

#### Option 2: Single Entity, Multiple Entity-Instances
* `entity_id`: the **string** denoting which entity instance the observation is for
* `timestamp`: the **pandas timestamp** object or **python datetime** object corresponding to the time at which the observation is made
* `value`: an **integer** or **float** column with the observed target values at the indicated timestamps

This is an example of such table, where  the values are for energy demand and the entity_id's are for 4 seperate locations we want to forecast for:



 |  timestamp | entity_id  |     value |
|------------|------------|-----------|
 9/27/20 21:20 |  DAYTON|1841.6 |
|  9/27/20 21:20 | DEOK|2892.5 |
| 9/27/20 21:20|  DOM|11276 |
|9/27/20 21:20| DPL|2113.7|
| 9/27/20 21:25 | DAYTON|1834.1 |
| 9/27/20 21:25 |DEOK| 2880.2 |
| 9/27/20 21:25| DOM| 11211.7 |
|9/27/20 21:25|DPL| 2086.6|

### Exogenous Inputs Table
Optionally, a second .csv file of exogenous inputs can be included. Exogenous inputs are time series that are not influenced by variables in the system, but they affect the output. In the first example, weather data is an example of exogenous input that has a strong correlation to taxi demand.

|  timestamp |     Temp |   Rain |
|------------|-----------|-----------|
| 7/1/14 0:51|	75.92|	0|
|7/1/14 1:51|	75.92|	0|
|7/1/14 2:51|	75.02|	0|
|7/1/14 3:51|	75.92|	0|
|7/1/14 4:51|	75.02|	0.02|
|7/1/14 5:51|	75.02|	0.06|

The timestamp must begin at or before the target value table's first timestamp and end at or after the target value table's last timestamp, but the level of granularity between the target table and the exogenous input table does not have to match.



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

TODO: Create a step by step guide here.

# What's next?

For more details about **pyteller** and all its possibilities
and features, please check the [documentation site](
https://signals-dev.github.io/pyteller/).
