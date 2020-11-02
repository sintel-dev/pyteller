# -*- coding: utf-8 -*-

"""Main module."""
import json
import os
from typing import List

import pandas as pd
from mlblocks import MLPipeline
from pyteller.evaluation import METRICS_NORM as METRICS


class Pyteller:

    def _get_mlpipeline(self):
        pipeline = self._pipeline
        if isinstance(pipeline, str) and os.path.isfile(pipeline):
            with open(pipeline) as json_file:
                pipeline = json.load(json_file)
        mlpipeline = MLPipeline(pipeline)
        if self._hyperparameters:
            mlpipeline.set_hyperparameters(self._hyperparameters)
        return mlpipeline

    def __init__(self, pipeline=None,
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
        self._mlpipeline = self._get_mlpipeline()
        self._fitted = False

# TODO: fit user facing abstraction
# TODO: save/load
# TODO: commnet in blocks
    def predict(self, test_data):
        # Allow for multiple entities
        preds = pd.DataFrame()
        for entity, test_entity in test_data:
            preds_entity = self._mlpipeline.predict(X=test_entity,
                                                    pred_length=self.pred_length,
                                                    offset=self.offset,
                                                    goal=self.goal,
                                                    goal_window=None)
            preds_entity['entity'] = entity
            preds = preds.append(preds_entity)
        return preds

    def evaluate(self, forecast: pd.DataFrame,
                 fit: bool = False,
                 train_data: pd.DataFrame = None,
                 test_data: pd.DataFrame = None,
                 detailed=False,
                 metrics: List[str] = METRICS) -> pd.Series:
        forecast = forecast.groupby('entity')
        metrics_ = {}
        for metric in metrics:
            metrics_[metric] = METRICS[metric]
        scores = list()
        for entity, forecast_entity in forecast:

            pred_window = (test_data.get_group(entity)['timestamp']
                           >= forecast_entity['timestamp'].iloc[0]) & (
                test_data.get_group(entity)['timestamp']
                <= forecast_entity['timestamp'].iloc[-1])
            actual_entity = test_data.get_group(entity).loc[pred_window]
            signals = [col for col in forecast_entity if col.startswith('signal')]
            for signal in signals:
                score = {}
                if 'MASE' in metrics_:
                    score['MASE'] = metrics_['MASE'](train_data.get_group(entity)[signal],
                                                     forecast_entity[signal], actual_entity[signal])

                score.update({
                    metric: METRICS[metric](actual_entity[signal], forecast_entity[signal])
                    for metric in metrics_ if metric != 'MASE'
                })
                granularity = pd.to_datetime(
                    train_data.get_group(entity)['timestamp'].iloc[1]) - pd.to_datetime(
                    train_data.get_group(entity)['timestamp'].iloc[0])
                score['entity'] = entity
                score['signal']=signal

                if detailed == True:
                    score['granularity'] = granularity
                    score['prediction length'] = forecast_entity.shape[0]
                    score['length of training data'] = len(train_data.get_group(entity))
                    score['length of testing data'] = len(test_data.get_group(entity))
                scores.append(score)
        # scores = pd.DataFrame.from_records(scores)

        return pd.DataFrame(scores)
