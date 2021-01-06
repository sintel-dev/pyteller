.. highlight:: shell

====
Data
====

Pyteller takes time series signals in longform or flatform as well as indetifiers for which column names represent the `timestamp column`,
`target signal`, `entitiy column`, and the `entities` you want to forecast.


Data Format
-----------

Input
~~~~~

Orion Pipelines work on time Series that are provided as a single table of telemetry
observations with two columns:

* `timestamp_col`: the STRING denoting which column contains the pandas timestamp objects or python datetime objects corresponding to the time at which the observation is made
* `target_signal`: the STRING denoting the INTEGERS or FLOATS column with the observed target values at the indicated timestamps
* `entity_col`: the STRING denoting which column contains the entities the observations are for
* `entities`: the STRING denoting either which entities from `entity_col` to make forecsts for, or denoting which coumns are the target entities

This is an example of such table, where the `timestamp_col` is 'timestamp', the `entity_col` is 'region',  the `target` is 'demand,'




+------------------+-----------+-----------+
|  timestamp       |  region   |   demand  |
+------------------+-----------+-----------+
| 9/27/20 21:20    |   DAYTON  |    1841.6 |
+------------------+-----------+-----------+
| 9/27/20 21:20    |   DEOK    |    2892.5 |
+------------------+-----------+-----------+
| 9/27/20 21:25    |   DAYTON  |    1834.1 |
+------------------+-----------+-----------+
| 9/27/20 21:25    |   DEOK    |    75.92  |
+------------------+-----------+-----------+


Output
~~~~~~

The output of pyteller is a table that contains the scores of the forecasts for each entity:


An example of such a table is:

+------------+------------+----------+
|     Metric |     DAYTON |     DEOK |
+------------+------------+----------+
| MAPE       | 8.8        | 102.3    |
+------------+------------+----------+
| MAE        | 9.8        | 96.5     |
+------------+------------+----------+

Dataset we use in this library
------------------------------


For development and evaluation of pipelines, we include the following datasets:

**NYC taxi data**
    Found on the `nyc website`_, or the processed version maintained by `Numenta`_. No modifications were made from the Numenta version

**Wind data**
    Found here on `here`_. After downloading the FasTrak 5-Minute .txt files the .txt files for each day from 1/1/13-1/8/18 were compiled into one .csv file


**Weather data**
    Maintained by Iowa State University's `IEM`_. The downloaded data was from the selected network of 8A0 Albertville and the selected date range was 1/1/16 0:15 - 2/16/16 0:55

**Traffic data**
    Found on `Caltrans PeMS`_. No modifications were made from the Numenta version

**Energy data**
    Found on `kaggle`_. No modifications were made after downloading pjm_hourly_est.csv

.. _nyc website: https://www1.nyc.gov/site/tlc/about/tlc-trip-record-data.page
.. _Numenta: https://github.com/numenta/NAB/tree/master/data
.. _here: https://www.kaggle.com/sohier/30-years-of-european-wind-generation/metadata
.. _IEM: https://mesonet.agron.iastate.edu/request/download.phtml?network=ILASOS
.. _Caltrans PeMS: http://pems.dot.ca.gov/?dnode=Clearinghouse
.. _kaggle: https://www.kaggle.com/robikscube/hourly-energy-consumption/metadata
