.. _quickstart:

Quickstart
==========

In the following steps we will show a short guide about how to run one of the **pyteller pipelines**
on one of the signals from the demo datasets.

1. Load the data
----------------

Here is an example of loading the **Alabama Weather** demo data which has multiple entities in long form:
`current_data` will be used to fit the pipeline and `input_data` to forecast. Both are dataframes:

.. ipython:: python
    :okwarning:

    from pyteller.data import load_data

    current_data, input_data = load_data('AL_Weather')

    current_data.head()


2. Fit the pipeline
-------------------------------

Once we have the data, create an instance of the `Pyteller` class, where the input arguments are the forecast settings and the column specifications of the data.
In this example we use the `lstm` pipeline and set the training epochs as 5.

.. ipython:: python
    :okwarning:

	from pyteller import Pyteller

    pipeline = 'pyteller.LSTM.LSTM'

    hyperparameters = {
        'keras.Sequential.LSTMTimeSeriesRegressor#1': {
            'epochs': 4
        }
    }


.. ipython:: python
    :okwarning:

    pyteller = Pyteller(
        pipeline=pipeline,
        pred_length=12,
        time_column='valid',
        targets='tmpf',
        entity_column='station',
        entities='8A0',
        hyperparameters=hyperparameters
    )

    pyteller.fit(current_data)

3. Forecast
-------------------------------
To make a forecast, the user calls the `pyteller.forecast` method. The output is a `dictionary` which includes the `forecasts` and `actuals` `dataframes`:

.. ipython:: python
	:okwarning:

    output = pyteller.forecast(data=input_data)
    output['forecasts'].head()

4. Evaluate
-------------------------------
To see metrics of the forecast accuracy, the user calls the `pyteller.evaluate` method. The output is a dataframe of the scores:

.. ipython:: python
    :okwarning:

    scores = pyteller.evaluate(actuals=output['actuals'],forecasts=output['forecasts'],
                               metrics=['sMAPE','MAPE'])
    scores.head()


