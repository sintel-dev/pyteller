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
                 lead = None,
                 goal = None,
                 goal_window = None):
        self._pipeline = pipeline
        self._hyperparameters = hyperparameters
        self.lead = lead
        self.goal = goal
        self._mlpipeline = self._get_mlpipeline()
        self._fitted = False

    def predict(self,test_data):
        #Allow for multiple entities
        preds=pd.DataFrame()
        for entity, test_entity in test_data:
            preds_entity= self._mlpipeline.predict(X=test_entity,
                                     lead=self.lead,
                                     goal=self.goal,
                                     goal_window=None)
            preds_entity['entity']=entity
            preds=preds.append(preds_entity)
        return preds

    def evaluate(self, forecast: pd.DataFrame,
                 fit: bool = False,
                 train_data: pd.DataFrame = None,
                 test_data: pd.DataFrame = None,
                 metrics: List[str] = METRICS) -> pd.Series:
        forecast = forecast.groupby('entity')

        scores = list()
        for entity, forecast_entity in forecast:
            score = {}
            pred_window = (test_data.get_group(entity)['timestamp'] >=
                           forecast_entity['timestamp'].iloc[0]) & (
                              test_data.get_group(entity)['timestamp'] <=
                              forecast_entity['timestamp'].iloc[-1])
            actual_entity = test_data.get_group(entity).loc[pred_window]

            if 'MASE' in metrics:
                score['MASE'] = metrics['MASE'](train_data.get_group(entity)['target'],
                                                forecast_entity['target'], actual_entity['target'])

            score.update({
                metric: METRICS[metric](actual_entity['target'], forecast_entity['target'])
                for metric in metrics if metric != 'MASE'
            })
            score['entity'] = entity
            scores.append(score)
        scores = pd.DataFrame.from_records(scores)

        return scores


