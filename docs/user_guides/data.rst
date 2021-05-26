.. highlight:: shell

====
Data Format
====

Pyteller takes time series signals in multiple formats, and the user simply needs to specify the column names corresponding to any of the following:

* `time_column`: Column denoting the timestamp column.
* `target_column`: Column denoting the target column.
* `targets`: List of the subset of targets to extract.
* `entity_column`: Column denoting the entities column.
* `entities`: Subset of entities to extract.


This is an example of a table, where there are multiple entities and only one target signal is to be forecasted for.


+---------+-------------+------+------+-------+------+
| station | valid       | tmpf | dwpf | relh  | drct |
+---------+-------------+------+------+-------+------+
| 8A0     | 1/1/16 0:15 | 41   | 39.2 | 93.24 | 350  |
+---------+-------------+------+------+-------+------+
| 4A6     | 1/1/16 0:15 | 41   | 32   | 70.08 | 360  |
+---------+-------------+------+------+-------+------+
| 8A0     | 1/1/16 0:35 | 39.2 | 37.4 | 93.19 | 360  |
+---------+-------------+------+------+-------+------+
| 4A6     | 1/1/16 0:35 | 41   | 32   | 70.08 | 360  |
+---------+-------------+------+------+-------+------+
| 8A0     | 1/1/16 0:55 | 37.4 | 37.4 | 100   | 360  |
+---------+-------------+------+------+-------+------+
| 4A6     | 1/1/16 0:55 | 39.2 | 32   | 75.16 | 350  |
+---------+-------------+------+------+-------+------+

The `time_column`, `entity_column`, `entities`, and `targets` would be specified.
The `target_column` does not need to be specified becuase there is not a column that denotes the variable name of the target.


Demo Dataset we use in this library
------------------------------


For development and evaluation of pipelines, we include the following datasets:

**NYC taxi data**
    Found on the `nyc website`_, or the processed version maintained by `Numenta`_. No modifications were made from the Numenta version

**Weather data**
    Maintained by Iowa State University's `IEM`_. The downloaded data was from the selected network of 8A0 Albertville and the selected date range was 1/1/16 0:15 - 2/16/16 0:55


**Energy data**
    Found on `kaggle`_. No modifications were made after downloading pjm_hourly_est.csv

.. _nyc website: https://www1.nyc.gov/site/tlc/about/tlc-trip-record-data.page
.. _Numenta: https://github.com/numenta/NAB/tree/master/data
.. _IEM: https://mesonet.agron.iastate.edu/request/download.phtml?network=ILASOS
.. _kaggle: https://www.kaggle.com/robikscube/hourly-energy-consumption/metadata
