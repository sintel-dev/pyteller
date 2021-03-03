# -*- coding: utf-8 -*-

"""Main module."""
import logging
import os
import pickle
from copy import deepcopy

import pandas as pd
from mlblocks import MLPipeline, load_pipeline
from sklearn.exceptions import NotFittedError

from pyteller.metrics import METRICS
from pyteller.utils import plot_forecast

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
    @staticmethod
    def _update_init_params(pipeline, primitives, value):
        for primitive in primitives:
            primitive_params = {}
            if primitive in pipeline['init_params'].keys():
                primitive_params = pipeline['init_params'][primitive]

            primitive_params[primitives[primitive]] = value
            pipeline['init_params'][primitive] = primitive_params

        return pipeline

    def _build_pipeline(self, pipeline):
        self._fitted = False

        if isinstance(pipeline, MLPipeline):
            pipeline = pipeline.to_dict()

        elif not isinstance(pipeline, dict):
            pipeline = load_pipeline(pipeline)

        # Pipeline arguments are specified in all pyteller pipeline jsons and allow
        # for shared hyperparamters
        if 'pipeline_arguments' in pipeline.keys():
            pipeline_args = deepcopy(pipeline['pipeline_arguments'])

            pred_length_primitives = pipeline_args.get('pred_length', {})
            pipeline = self._update_init_params(pipeline, pred_length_primitives, self.pred_length)

            offset_primitives = pipeline_args.get('offset', {})
            pipeline = self._update_init_params(pipeline, offset_primitives, self.offset)

            target_column_primitives = pipeline_args.get('target_column', {})
            pipeline = self._update_init_params(
                pipeline, target_column_primitives, self.target_signal_column)

        # Create the MLPipeline
        pipeline = MLPipeline(pipeline)

        if self._hyperparameters:
            pipeline.set_hyperparameters(deepcopy(self._hyperparameters))

        return pipeline

    def __init__(self, pipeline, timestamp_col='timestamp', target_signal=None,
                 target_signal_col=None, static_variables=None, entities=None,
                 entity_col=None, pred_length=None, offset=None, hyperparameters=None):

        self.timestamp_col = timestamp_col
        self.target_signal = target_signal
        self.target_signal_column = target_signal_col
        self.static_variables = static_variables
        self.entity_cols = entity_col
        self.entities = entities
        self.pred_length = pred_length
        self.offset = offset

        self._hyperparameters = deepcopy(hyperparameters) or {}

        self.pipeline = self._build_pipeline(pipeline)

    def _get_outputs_spec(self, spec):
        try:
            output_spec = self.pipeline.get_output_names(spec)
        except ValueError:
            output_spec = []

        return output_spec

    def _to_dict(self):
        return {
            'pred_length': self.pred_length,
            'offset': self.offset,
            'entities': self.entities,
            'target_signal': self.target_signal,
            'target_column': self.target_signal_column,
            'timestamp_col': self.timestamp_col,
            'static_variables': self.static_variables,
            'entity_col': self.entity_cols,
        }

    def fit(self, data=None, start_=None, output_=None, **kwargs):
        """Fit the pipeline to the given data.

        Args:
            data (DataFrame):
                Input data, passed as a ``pandas.DataFrame``
            start_ (str or int or None):
                Block index or block name to start processing from. The
                value can either be an integer, which will be interpreted as a block index,
                or the name of a block, including the conter number at the end.
                If given, the execution of the pipeline will start on the specified block,
                and all the blocks before that one will be skipped.
            output_ (str or int or list or None):
                Output specification, as required by ``get_outputs``. If ``None`` is given,
                nothing will be returned.

        Returns:
            Dictionary:
                Optional. A dictionary containing the specified outputs from the
                ``MLPipeline``

        """
        if data is None:
            data = kwargs.pop('X')

        kwargs.update(self._to_dict())

        out = self.pipeline.fit(
            X=data,
            start_=start_,
            output_=output_,
            **kwargs
        )

        if output_ is None:
            LOGGER.info('The pipeline is fitted')
            self._fitted = True

        return out

    def forecast(self, data, postprocessing=False, predictions_only=False, **kwargs):
        """Forecast input data on a trained model.

        Args:
            data (DataFrame):
                Input data, passed as a ``pandas.DataFrame`` containing
                exactly two columns: timestamp and value.

            postprocessing (bool):
                If ``True``, also return the ``input`` data as output from the
                ``MLPipeline``.
        Returns:
            Dictionary:
                A dictionary containing the specified ``default`` and ``visualization`` output
                 from the ``MLPipeline``
        """
        if not self._fitted:
            raise NotFittedError()

        output_spec = ['default', 'postprocessing'] if postprocessing else 'default'

        default_names = self._get_outputs_spec('default')
        kwargs.update(self._to_dict())

        outputs = self.pipeline.predict(X=data, output_=output_spec, **kwargs)

        if postprocessing:
            postprocessing = self._get_outputs_spec('postprocessing')
            postprocessing_outputs = outputs[-len(postprocessing):]
            default_outputs = outputs[:len(postprocessing) + 1]
            names = postprocessing + default_names
            outputs = postprocessing_outputs + default_outputs
            output_dict = dict(zip(names, outputs))

        else:
            output_dict = dict(zip(default_names, outputs))

        if predictions_only:
            return output_dict['forecast']

        return output_dict

    @staticmethod
    def plot(forecast_output, frequency='day'):
        actual = forecast_output['actual'].iloc[:, 0:1]
        forecast = forecast_output['forecast'].iloc[:, 0:1]
        return plot_forecast([actual, forecast], frequency=frequency)

    def evaluate(self, forecast, test_data, detailed=False, metrics=METRICS):
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
            Pyteller:
                Loaded Pyteller instance.
        """
        with open(path, 'rb') as pickle_file:
            return pickle.load(pickle_file)
