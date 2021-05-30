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

	from pyteller.core import Pyteller

    pipeline = 'pyteller/pipelines/pyteller/LSTM/LSTM.json'

    hyperparameters = {
        'keras.Sequential.LSTMTimeSeriesRegressor#1': {
            'epochs': 20
        }
    }


.. ipython:: python
    :okwarning:

    pyteller = Pyteller(
        pipeline=pipeline,
        time_column='valid',
        targets='tmpf',
        entity_column='station',
        entities='8A0'
        pred_length= 12,
        offset= 0,
        hyperparameters=hyperparameters
    )

    pyteller.fit(current_data)

