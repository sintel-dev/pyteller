# -*- coding: utf-8 -*-

"""Main module."""
import json
import logging
import os
import pickle
from copy import deepcopy

import pandas as pd
from mlblocks import MLPipeline

from pyteller.metrics import METRICS

LOGGER = logging.getLogger(__name__)


class Pyteller:
    """Pyteller Class.

    The Pyteller Class provides the time series forecasting functionalities
    of pyteller and is responsible for the interaction with the underlying
    MLBlocks pipelines.

    Args:
        pipeline (str, dict or MLPipeline):

            Pipeline to use. It can be passed as:

                * A ``str`` with a path to a JSON file.
                * A ``str`` with the name of a registered pipeline.
                * An ``MLPipeline`` instance.
                * A ``dict`` with an ``MLPipeline`` specification.

        hyperparameters (dict):
            Additional hyperparameters to set to the Pipeline.
    """

    def _update_init_params(self, pipeline, primitives, value):
        for primitive in primitives:
            primitive_params = {}
            if primitive in pipeline['init_params'].keys():
                primitive_params = pipeline['init_params'][primitive]

            primitive_params[primitives[primitive]] = value
            pipeline['init_params'][primitive] = primitive_params
            # pipeline['init_params'][primitive].update(primitive_params)

        return pipeline

    def _build_pipeline(self):
        if isinstance(self.pipeline, str) and os.path.isfile(self.pipeline):
            with open(self.pipeline) as json_file:
                pipeline = json.load(json_file)

        elif isinstance(self.pipeline, MLPipeline):
            pipeline = pipeline.to_dict()

        # Pipeline args are specified in all pyteller pipelines jsons and allow for
        # shared hyperparamters
        if 'pipeline_arguments' in pipeline.keys():
            pipeline_args = deepcopy(pipeline['pipeline_arguments'])

            pred_length_primitives = pipeline_args['pred_length']
            pipeline = self._update_init_params(pipeline, pred_length_primitives, self.pred_length)

            offset_primitives = pipeline_args['offset']
            pipeline = self._update_init_params(pipeline, offset_primitives, self.offset)

        pipeline = MLPipeline(pipeline)
        if self._hyperparameters:
            pipeline.set_hyperparameters(deepcopy(self._hyperparameters))

        return pipeline

    def __init__(self, pipeline, timestamp_col, target_signal=None,
                 static_variables=None, entity_col=None, entities=None,
                 hyperparameters=None, pred_length=None, offset=None):

        self.target_signal = target_signal
        self.timestamp_col = timestamp_col
        self.static_variables = static_variables
        self.entity_cols = entity_col
        self.entities = entities
        self.pred_length = pred_length
        self.offset = offset

        self.pipeline = pipeline
        self._fitted = False
        self._hyperparameters = hyperparameters or {}

        self.pipeline = self._build_pipeline()

    def _get_outputs_spec(self, spec):
        try:
            output_spec = self.pipeline.get_output_names(spec)
        except ValueError:
            output_spec = []

        return output_spec

    def fit(self, data=None, start_=None, output_=None, **kwargs):
        """Fit the pipeline to the given data.

        Args:
            data (DataFrame):
                Input data, passed as a ``pandas.DataFrame`` containing
                exactly two columns: timestamp and value.
        Returns:
            Dictionary:
                Optional. A dictionary containing the specified ``training`` outputs from the
                ``MLPipeline``

        """

        if data is None:
            data = kwargs['train'].pop('X')


        training_names = self._get_outputs_spec('training')

        outputs = self.pipeline.fit(X=data, pred_length=self.pred_length, offset=self.offset,
                                    entities=self.entities, target_signal=self.target_signal,
                                    timestamp_col=self.timestamp_col,
                                    static_variables=self.static_variables,
                                    entity_col=self.entity_cols, target_column=None, start_=start_, #TODO Fix target column
                                    output_=output_, **kwargs)

        self._fitted = True
        LOGGER.info('The pipeline is fitted')
        if training_names:
            outputs_ = [outputs]
            names = training_names
            training_dict = dict(zip(names, outputs_))
            return training_dict

    def forecast(self, data, visualization=False):
        """Forecast input data on a trained model.

        Args:
            data (DataFrame):
                Input data, passed as a ``pandas.DataFrame`` containing
                exactly two columns: timestamp and value.
            visualization (bool):
                If ``True``, also capture the ``visualization`` named
                output from the ``MLPipeline`` and return it as a second
                output.
        Returns:
            Dictionary:
                A dictionary containing the specified ``default`` and ``visualization`` output
                 from the ``MLPipeline``
        """
        outputs_spec = 'visualization' if visualization else 'default'

        default_names = self._get_outputs_spec('default')
        outputs = self.pipeline.predict(X=data, pred_length=self.pred_length, offset=self.offset,
                                        entities=self.entities, target_signal=self.target_signal,
                                        timestamp_col=self.timestamp_col,
                                        static_variables=self.static_variables,
                                        entity_col=self.entity_cols,
                                        target_column=None, output_=outputs_spec)

        if visualization:
            visualization_names = self._get_outputs_spec('visualization')
            visualization_dict = {}
            if visualization_names:
                visualization_outputs = outputs[-len(visualization_names):]
                default_outputs = outputs[:len(visualization_names) + 1]
                names = visualization_names + default_names
                outputs = visualization_outputs + default_outputs
                visualization_dict = dict(zip(names, outputs))

            return visualization_dict

        default_dict = dict(zip(default_names, outputs))
        return default_dict

    def evaluate(self, forecast, test_data, detailed=False, metrics=METRICS, train_data=None):
        """Evaluate the performance against test set

        Args:
            forecast (DataFrame):
               Forecasts, passed as a ``pandas.DataFrame`` containing
                exactly two columns: timestamp and value.
            test_data (DataFrame):
               Testing data or the target data, passed as a ``pandas.DataFrame`` containing
                exactly two columns: timestamp and value.
            detailed (bool):
                Whether to output the detailed score report
            metrics (list):
                List of metrics to used passed as a list of strings.
                If not given, it defaults to all the pyteller metrics.
            train_data (DataFrame):
               Optional. Training data used for some metrics, passed as a ``pandas.DataFrame`` containing
                exactly two columns: timestamp and value.

        Returns:
            DataFrame:
                ``pandas.DataFrame`` with columns as entities and the metric name as index.
        """
        if isinstance(metrics, str):
            metrics = [metrics]

        if isinstance(metrics, list):
            metrics = {
                name: metric
                for name, metric in METRICS.items()
                if name in metrics
            }

        scores = list()
        entities = forecast.columns

        if isinstance(entities, str):
            entities = [entities]

        for entity in entities:
            score = {}
            score.update({
                name: metric(test_data[entity].values, forecast[entity].values)
                for name, metric in metrics.items() if name != 'MASE'
            })

            if detailed:
                score['granularity'] = forecast.index[1] - forecast.index[0]
                score['prediction length'] = forecast.shape[0]
                score['length of training data'] = len(train_data.get_group(entity))
                score['length of testing data'] = len(test_data.get_group(entity))

            scores.append(score)

        return pd.DataFrame(scores, index=entities).transpose()

    def save(self, path):
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
    def load(cls, path):
        """Load a pyteller instance from a pickle file.

        Args:
            path (str):
                Path to the file where the instance has been
                previously serialized.

        Returns:
            Orion
        """
        with open(path, 'rb') as pickle_file:
            pyteller = pickle.load(pickle_file)
            return pyteller
