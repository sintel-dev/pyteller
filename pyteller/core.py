# -*- coding: utf-8 -*-

"""Main module."""
import json
import logging
import os
import pickle
from typing import List, Union

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

    def __init__(self, pipeline= None,
                 hyperparameters: dict = None,
                 pred_length = None,
                 goal = None,
                 goal_window = None):
        self._pipeline = pipeline
        self._hyperparameters = hyperparameters
        self.pred_length = pred_length
        self.goal = goal
        self._mlpipeline = self._get_mlpipeline()
        self._fitted = False

# TODO: fit user facing abstraction
# TODO: save/load
# TODO: commnet in blocks
    def predict(self,test_data):
        #Allow for multiple entities
        preds=pd.DataFrame()
        for entity, test_entity in test_data:
            preds_entity= self._mlpipeline.predict(X=test_entity,
                                     pred_length=self.pred_length,
                                     goal=self.goal,
                                     goal_window=None)
            preds_entity['entity']=entity
            preds=preds.append(preds_entity)
        return preds

    def evaluate(self, forecast: pd.DataFrame,
                 fit: bool = False,
                 train_data: pd.DataFrame = None,
                 test_data: pd.DataFrame = None,
                 detailed = False,
                 metrics: List[str] = METRICS) -> pd.Series:
        forecast = forecast.groupby('entity')
        metrics_ = {}
        for metric in metrics:
            metrics_[metric] = METRICS[metric]
        scores = list()
        for entity, forecast_entity in forecast:
            score = {}
            pred_window = (test_data.get_group(entity)['timestamp'] >=
                           forecast_entity['timestamp'].iloc[0]) & (
                              test_data.get_group(entity)['timestamp'] <=
                              forecast_entity['timestamp'].iloc[-1])
            actual_entity = test_data.get_group(entity).loc[pred_window]

            if 'MASE' in metrics_:
                score['MASE'] = metrics_['MASE'](train_data.get_group(entity)['target'],
                                                forecast_entity['target'], actual_entity['target'])

            score.update({
                metric: METRICS[metric](actual_entity['target'], forecast_entity['target'])
                for metric in metrics_ if metric != 'MASE'
            })
            granularity = pd.to_datetime(
                train_data.get_group(entity)['timestamp'].iloc[1]) - pd.to_datetime(
                train_data.get_group(entity)['timestamp'].iloc[0])
            score['entity'] = entity
            scores.append(score)
            if detailed == True:
                score['granularity']=granularity
                score['prediction length'] = forecast_entity.shape[0]
                score['length of training data'] = len(train_data.get_group(entity))
                score['length of testing data'] = len(test_data.get_group(entity))

        scores = pd.DataFrame.from_records(scores)

        return scores


