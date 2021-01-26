# -*- coding: utf-8 -*-

"""Main module."""
import json
import logging
import os
import pickle


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
                * An ``str`` with a path to a JSON file.
                * An ``str`` with the name of a registered pipeline.
                * An ``MLPipeline`` instance.
                * A ``dict`` with an ``MLPipeline`` specification.
        hyperparameters (dict):
            Additional hyperparameters to set to the Pipeline.
    """
    def _get_mlpipeline(self):
        pipeline = self._pipeline
        if isinstance(pipeline, str) and os.path.isfile(pipeline):
            with open(pipeline) as json_file:
                pipeline = json.load(json_file)

        # Pipeline args are specified in all pyteller pipelines jsons and allow for
        # shared hyperparamters
        if 'pipeline_arguments' in pipeline.keys():
            pipeline_args = pipeline['pipeline_arguments']

            pred_length_primitives = pipeline_args['pred_length']
            for i in pred_length_primitives:
                if i in pipeline['init_params'].keys():
                    dict = pipeline['init_params'][i]
                else:
                    dict = {}

                dict[pred_length_primitives[i]] = self.pred_length
                pipeline['init_params'][i] = dict

            offset_primitives = pipeline_args['offset']
            for i in offset_primitives:
                if i in pipeline['init_params'].keys():
                    dict = pipeline['init_params'][i]
                else:
                    dict = {}

                dict[offset_primitives[i]] = self.offset
                pipeline['init_params'][i] = dict

        mlpipeline = MLPipeline(pipeline)
        if self._hyperparameters:
            mlpipeline.set_hyperparameters(self._hyperparameters)

        return mlpipeline

    def _load_pipeline(self, pipeline, hyperparams=None):
        if isinstance(pipeline, str) and os.path.isfile(pipeline):
            pipeline = MLPipeline.load(pipeline)
        else:
            pipeline = MLPipeline(pipeline)

        if hyperparams is not None:
            pipeline.set_hyperparameters(hyperparams)

        return pipeline

    def __init__(self, timestamp_col, target_signal=None, static_variables=None,
                 entity_col=None, entities=None, pipeline=None, hyperparameters=None,
                 pred_length=None, offset=None):

        self._pipeline = pipeline
        self._hyperparameters = hyperparameters
        self.pred_length = pred_length
        self.offset = offset
        self._fitted = False
        self.target_signal = target_signal
        self.timestamp_col = timestamp_col
        self.static_variables = static_variables
        self.entity_cols = entity_col
        self.entities = entities

    def _get_outputs_spec(self):
        outputs_spec = ["default"]
        default_outputs = self._mlpipeline.get_output_names('default')
        try:
            training_outputs = self._mlpipeline.get_output_names('training')
            outputs_spec.append('training')
        except ValueError:
            training_outputs = []

        try:
            visualization_outputs = self._mlpipeline.get_output_names('visualization')
            outputs_spec.append('visualization')
        except ValueError:
            visualization_outputs = []

        return outputs_spec, default_outputs, training_outputs, visualization_outputs

    def _fit(self, method, data):
        self._fitted = True
        LOGGER.info('The pipeline is fitted')

        outputs_spec, default_names, training_names, visualization_names = self._get_outputs_spec()
        outputs_spec = 'training'

        outputs = method(X=data, pred_length=self.pred_length, offset=self.offset,
                         entities=self.entities, target_signal = self.target_signal,
                         timestamp_col = self.timestamp_col,
                         static_variables = self.static_variables, entity_col = self.entity_cols,
                         target_column=None, output_=outputs_spec)

        if training_names:
            outputs_ = [outputs]
            names = training_names
            training_dict = dict(zip(names, outputs_))
        else:
            training_dict = {}

        return training_dict

    def fit(self, data):
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
        self._mlpipeline = self._get_mlpipeline()
        return self._fit(self._mlpipeline.fit, data)

    def _forecast(self, method, data, visualization=False):
        outputs_spec, default_names, training_names, visualization_names = self._get_outputs_spec()

        if ~visualization:
            outputs_spec = 'default'
        else:
            outputs_spec = 'visualization'

        outputs = method(X=data, pred_length=self.pred_length, offset=self.offset,
                         entities=self.entities,target_signal = self.target_signal,
                         timestamp_col = self.timestamp_col,
                         static_variables = self.static_variables, entity_col = self.entity_cols,
                         target_column=None, output_=outputs_spec)

        if visualization:
            if visualization_names:
                visualization_outputs = outputs[-len(visualization_names):]
                default_outputs = outputs[0:len(visualization_names)+1]
                names = visualization_names + default_names
                outputs_ = visualization_outputs + default_outputs
                visualization_dict = dict(zip(names, outputs_))
            else:
                visualization_dict = {}

            return visualization_dict

        else:
            default_dict = dict(zip(default_names, outputs))

            return default_dict

    def forecast(self, data, visualization: bool = False):
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
        return self._forecast(self._mlpipeline.predict, data, visualization)

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


    def evaluate(self, forecast, train_data=None, test_data=None, detailed=False, metrics=METRICS):
        """Evaluate the performance against test set

        Args:
            forecast (DataFrame):
               Forecasts, passed as a ``pandas.DataFrame`` containing
                exactly two columns: timestamp and value.
            train_data (DataFrame):
               Training data used for some metrics, passed as a ``pandas.DataFrame`` containing
                exactly two columns: timestamp and value.
            test_data (DataFrame):
               Testing data or the target data, passed as a ``pandas.DataFrame`` containing
                exactly two columns: timestamp and value.
            detailed (bool):
                Whether to output the detailed score report
            metrics (list):
                List of metrics to used passed as a list of strings.
                If not given, it defaults to all the pyteller metrics.

        Returns:
            DataFrame:
                ``pandas.DataFrame`` with columns as entities and the metric name as index.
        """

        metrics_ = {}
        if isinstance(metrics, str):
            metrics = [metrics]
        for metric in metrics:
            metrics_[metric] = METRICS[metric]

        scores = list()
        entities = forecast.columns

        if isinstance(self.entities, str):
            entities = [entities]

        for entity in entities:
            score = {}
            score.update({
                metric: METRICS[metric](test_data[entity].values, forecast[entity].values)
                for metric in metrics_ if metric != 'MASE'
            })

            if detailed:
                score['granularity'] = forecast.index[1] - forecast.index[0]
                score['prediction length'] = forecast.shape[0]
                score['length of training data'] = len(train_data.get_group(entity))
                score['length of testing data'] = len(test_data.get_group(entity))

            scores.append(score)

        return pd.DataFrame(scores, index=entities).transpose()
