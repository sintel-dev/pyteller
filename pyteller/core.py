# -*- coding: utf-8 -*-

"""Main module."""
import json
import os
from typing import List
from pyteller.data import organize_data, post_process
import pandas as pd
from mlblocks import MLPipeline
from pyteller.evaluation import METRICS_NORM as METRICS
import pickle


class Pyteller:
    def _load_pipeline(self,pipeline, hyperparams=None):
        if isinstance(pipeline, str) and os.path.isfile(pipeline):
            pipeline = MLPipeline.load(pipeline)
        else:
            pipeline = MLPipeline(pipeline)

        if hyperparams is not None:
            pipeline.set_hyperparameters(hyperparams)

        return pipeline

    def _get_mlpipeline(self):
        pipeline = self._pipeline
        if isinstance(pipeline, str) and os.path.isfile(pipeline):
            with open(pipeline) as json_file:
                pipeline = json.load(json_file)
        mlpipeline = MLPipeline(pipeline)
        if self._hyperparameters:
            mlpipeline.set_hyperparameters(self._hyperparameters)
        return mlpipeline

    def __init__(self,
              pipeline=None,
              hyperparameters: dict = None,
              pred_length=None,
              offset=None,
              goal=None,
              goal_window=None):
        self._pipeline = pipeline
        self._hyperparameters = hyperparameters
        self.pred_length = pred_length
        self.offset=offset
        self.goal = goal
        self.goal_window=goal_window
        self._mlpipeline = self._get_mlpipeline()
        self._fitted = False

        hyperparameters = {
            "mlprimitives.custom.timeseries_preprocessing.rolling_window_sequences#1": {
                "target_size": self.pred_length,
                "step_size": self.pred_length
            },
            "keras.Sequential.LSTMTimeSeriesRegressor#1": {
                "dense_units": self.pred_length,
            }

        }


        self._hyperparameters=hyperparameters


# TODO: fit user facing abstraction
# TODO: save/load
# TODO: commnet in blocks
# TODO: switch entity with signal
    def fit(self,
            data=None,
            timestamp_col = None,
            target_signals=None,
            static_variables=None,
            entity_col=None,
            entities=None,
            train_size=None
        ):
        """Fit the pipeline to the given data.

        Args:
            data (DataFrame):
                Input data, passed as a ``pandas.DataFrame`` containing
                exactly two columns: timestamp and value.
        """
        self.target_signal=target_signals
        self.timestamp_col=timestamp_col
        self.static_variables=static_variables
        self.entity_cols=entity_col
        self.entities=entities
        self.train_size=train_size

        # self._mlpipeline = self._get_mlpipeline()
        self._mlpipeline = self._load_pipeline(self._pipeline, self._hyperparameters)
        self.train_length = round(len(data) * train_size)
        # train_df = data.iloc[:self.train_length]
        train = organize_data(self,
            data=data,
            timestamp_col = self.timestamp_col,
            signal=self.target_signal,
            static_variables=self.static_variables,
            entity_cols=self.entity_cols,
            entities=self.entities,
            train_size=self.train_size
        )


        self._mlpipeline.fit(X=train,
                         pred_length=self.pred_length,
                        )
        self._fitted = True

        print('The pipeline is fitted')

    def forecast(self, data=None):

        test = organize_data(self,
            data=data,
            timestamp_col = self.timestamp_col,
            signal=self.target_signal,
            static_variables=self.static_variables,
            entity_cols=self.entity_cols,
            entities=self.entities,
            train_size=self.train_size
        )

        prediction = self._mlpipeline.predict(X=test,
                                                pred_length=self.pred_length,
                                                offset=self.offset,
                                                goal=self.goal,
                                                goal_window=None
                                              )

        prediction = post_process(self,prediction)
        #
        # self.time = prediction['timestamp'].iloc[0] + ' to ' + \
        #                prediction['timestamp'].iloc[-1]
        # print('pred')
        # to_print = [
        #     'Forecast Summary:',
        #     "\tSignals predicted: {}".format(self.target_signal),
        #     # "\tEntities predicted: {}".format(preds.entity.unique()),
        #     "\tEntities predicted: {} from {}".format(self.entities, self.time),
        #     "\tPipeline: : {}".format(self._pipeline),
        #     "\tOffset: : {}".format(self.offset),
        #     "\tPrediction length: : {}".format(self.pred_length),
        #     "\tPrediction goal: : {}".format(self.goal),
        # ]
        # print('\n'.join(to_print))
        return prediction

    def evaluate(self, forecast: pd.DataFrame,
                 fit: bool = False,
                 train_data: pd.DataFrame = None,
                 test_data: pd.DataFrame = None,
                 detailed=False,
                 metrics: List[str] = METRICS) -> pd.Series:


        metrics_ = {}
        if isinstance(metrics, str):
            metrics = [metrics]
        for metric in metrics:
            metrics_[metric] = METRICS[metric]
        scores = list()

        test_data = organize_data(self,
            data=test_data,
            timestamp_col = self.timestamp_col,
            signal=self.target_signal,
            static_variables=self.static_variables,
            entity_cols=self.entity_cols,
            entities=self.entities,
            train_size=self.train_size
        )

        # pred_window = (test_data['timestamp']
        #                >= forecast['timestamp'].iloc[0]) & (
        #     test_data['timestamp']
        #     <= forecast['timestamp'].iloc[-1])
        test_data['timestamp'] = pd.to_datetime(test_data['timestamp'] * 1e9)
        test_data = test_data.set_index('timestamp')
        actual = test_data[test_data.index.isin(forecast.index)]
        # signals = [col for col in forecast_entity if col.startswith('signal')]
        if isinstance(self.entities, str):
            self.entities = [self.entities]
        for entity in self.entities:
            score = {}
            # if 'MASE' in metrics_:
            #     score['MASE'] = metrics_['MASE'](train_data[entity],
            #                                      forecast[entity], actual[entity])

            score.update({
                metric: METRICS[metric](actual[entity], forecast[entity])
                for metric in metrics_ if metric != 'MASE'
            })


            if detailed == True:
                score['granularity'] = self.freq
                score['prediction length'] = forecast.shape[0]
                score['length of training data'] = len(train_data.get_group(entity))
                score['length of testing data'] = len(test_data.get_group(entity))
            scores.append(score)
        # scores = pd.DataFrame.from_records(scores)

        return pd.DataFrame(scores)


    def save(self, path: str):
        """Save this object using pickle.

        Args:
            path (str):
                Path to the file where the serialization of
                this object will be stored.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as pickle_file:
            pickle.dump(self, pickle_file)

    @classmethod
    def load(cls, path: str):
        """Load a pyteller instance from a pickle file.

        Args:
            path (str):
                Path to the file where the instance has been
                previously serialized.

        Returns:
            Orion

        Raises:
            ValueError:
                If the serialized object is not a pyteller instance.
        """
        with open(path, 'rb') as pickle_file:
            pyteller = pickle.load(pickle_file)
            if not isinstance(orion, cls):
                raise ValueError('Serialized object is not a pyteller instance')

            return pyteller
