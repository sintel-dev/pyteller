# -*- coding: utf-8 -*-

"""Main module."""
import logging
import os
import pickle
from copy import deepcopy

import pandas as pd
from mlblocks import MLPipeline, load_pipeline
from sklearn.exceptions import NotFittedError
from btb.session import BTBSession
from sklearn.model_selection import cross_val_score
from mlprimitives.datasets import Dataset
from sklearn.metrics import mean_absolute_error
import numpy as np
from btb.tuning import GPTuner, Tunable

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

        time_column (string):
            Optional. A ``str`` specifying the name of the timestamp column of the input data

        target_column (string):
            Optional. A ``str`` specifying the name of the column containing the target signal

        static_variables (string):
            Optional. A ``str`` specifying the name of the column of the input data containing
            static variables

        entities (string or list):
            Optional. A ``str`` or ``list`` specifying the name(s) of the entities from the
            entity_column the user wants to make forecasts for

        entity_column (string):
            A ``str`` specifying the name of the column of the input data containing the entity
            names

        pred_length (int):
            Optional. An ``int`` specifying the number of timesteps to forecast ahead for

        offset (int):
            An ``int`` specifying the number of timesteps between the input and the target
            sequence

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

        # Create the MLPipeline
        pipeline = MLPipeline(pipeline)

        if self._hyperparameters:
            pipeline.set_hyperparameters(deepcopy(self._hyperparameters))

        return pipeline

    def __init__(self, pipeline, time_column=None, target_column=None, targets=None,
                 static_variables=None, entities=None, entity_column=None, pred_length=None,
                 offset=None, hyperparameters=None):

        self.time_column = time_column or 'timestamp'
        self.target_column = target_column
        self.targets=targets
        self.static_variables = static_variables
        self.entity_column = entity_column
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
            'targets' : self.targets,
            'target_column': self.target_column,
            'time_column': self.time_column,
            'static_variables': self.static_variables,
            'entity_column': self.entity_column,
        }


    def k_fold_validation(self, hyperparameters, X, y, scoring=None):
        """Score the pipeline through k-fold validation with the given scoring function.
        Args:
            hyperparameters (dict or None):
                A dictionary of hyper-parameters for each primitive in the target pipeline.
            X (pandas.DataFrame or ndarray):
                Inputs to the pipeline.
            y (pandas.Series or ndarray):
                Target values.
            scoring (str):
                The name of the scoring function.
        Returns:
            np.float64:
                The average score in the k-fold validation.
        """
        model_instance = MLPipeline(self._pipeline)
        X = pd.DataFrame(X)
        y = pd.Series(y)

        if hyperparameters:
            model_instance.set_hyperparameters(hyperparameters)

        if self._problem_type == 'regression':
            scorer = self.regression_metrics[scoring or 'R2 Score']
        else:
            scorer = self.classification_metrics[scoring or 'F1 Macro']

        scores = []
        kf = KFold(n_splits=10, random_state=None, shuffle=True)
        for train_index, test_index in kf.split(X):
            model_instance.fit(X.iloc[train_index], y.iloc[train_index])
            y_pred = model_instance.predict(X.iloc[test_index])
            scores.append(scorer(y.iloc[test_index], y_pred))

        return np.mean(scores)

    def scoring_function(self, X, hyperparameters=None):
        # choose the model
        # model_instance = MLPipeline(self.pipeline)
        dataset = Dataset('data', X, X, mean_absolute_error, 'timeseries', 'forecast',
                          shuffle=False)

        # instantiate the model
        if hyperparameters:
            # model_instance.set_hyperparameters(hyperparameters)
            self.pipeline.set_hyperparameters(hyperparameters)
        scores = []
        for X_train, X_test, y_train, y_test in dataset.get_splits(3):
            self.fit(X_train)
            output=self.forecast(X_test)
            scores.append(self.evaluate(actuals=output['actuals'],
                                            forecasts=output['forecasts'],
                                            metrics=['MAE']).values[0][0])
        return np.mean(scores)
    def tune(self, X, y, max_evals=10, scoring=None, verbose=False):
        """ Tune the pipeline hyper-parameters and select the optimized model.
        Args:
            X (pandas.DataFrame or ndarray):
                Inputs to the pipeline.
            y (pandas.Series or ndarray):
                Target values.
            max_evals (int):
                Maximum number of hyper-parameter optimization iterations.
            scoring (str):
                The name of the scoring function.
            verbose (bool):
                Whether to log information during processing.
        """

        from sklearn.metrics import make_scorer, f1_score
        from sklearn.model_selection import cross_val_score



        tunables = self.pipeline.get_tunable_hyperparameters(flat=True)
        tunable = Tunable.from_dict(tunables)
        default_score = self.scoring_function(X)
        tuner = GPTuner(tunable)
        defaults = tunable.get_defaults()
        tuner.record(defaults, default_score)
        best_score = default_score
        best_proposal = defaults

        for iteration in range(max_evals):
            print("scoring pipeline {}".format(iteration + 1))
            proposal = tuner.propose()
            score = self.scoring_function(X,proposal)
            tuner.record(proposal, score)

            if score < best_score:
                print("New best found: {}".format(score))
                best_score = score
                best_proposal = proposal
        self.pipeline.set_hyperparameters(best_proposal)



    def fit(self, data=None, tune=False, max_evals=10, scoring=None,  start_=None, verbose=False, output_=None, **kwargs):
        """Fit the pipeline to the given data.

        Args:
            tune (bool):
                Whether to optimize hyper-parameters of the pipelines.
            max_evals (int):
                Maximum number of hyper-parameter optimization iterations.
            scoring (str):
                The name of the scoring function used in the hyper-parameter optimization.
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

        else:
            # if self.time_column == None:
            #     self.time_column = data.columns[0]
            kwargs.update(self._to_dict())

        if tune:
            # tune and select pipeline
            self.tune(data, data, max_evals=max_evals, scoring=scoring, verbose=verbose)

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

            predictions_only (bool):
                If ``True``, only return forecasts as an output from the ``MLPipeline``.

        Returns:
            Dictionary:
                A dictionary containing the specified ``default`` and ``postprocessing`` output
                 from the ``MLPipeline``
        """
        if not self._fitted:
            raise NotFittedError()

        output_spec = ['default', 'postprocessing'] if postprocessing else 'default'

        default_names = self._get_outputs_spec('default')
        kwargs.update(self._to_dict())

        outputs = self.pipeline.predict(X=data, output_=output_spec, debug=True, **kwargs)

        if postprocessing:
            postprocessing = self._get_outputs_spec('postprocessing')
            postprocessing_outputs = outputs[0][-len(postprocessing):]
            default_outputs = outputs[0][:len(postprocessing) + 1]
            names = postprocessing + default_names
            outputs = postprocessing_outputs + default_outputs
            output_dict = dict(zip(names, outputs))

        else:
            output_dict = dict(zip(default_names, outputs[0]))

        if predictions_only:
            return output_dict['forecasts']

        return output_dict

    @staticmethod
    def plot(forecast_output, frequency='day'):
        actuals = forecast_output['actuals'].iloc[:, 0:1]
        forecasts = forecast_output['forecasts'].iloc[:, 0:1]
        return plot_forecast([actuals, forecasts], frequency=frequency)

    def evaluate(self, actuals, forecasts, detailed=False, metrics=METRICS):
        """Evaluate the performance against test set

        Args:
            forecasts (DataFrame):
               Forecasts, passed as a ``pandas.DataFrame`` containing
                exactly two columns: timestamp and value.
            actuals (DataFrame):
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
        forecasts = forecasts[forecasts.index.isin(actuals.index)]

        if isinstance(metrics, str):
            metrics = [metrics]

        if isinstance(metrics, list):
            metrics = {
                name: metric
                for name, metric in METRICS.items()
                if name in metrics
            }

        scores = list()
        entities = forecasts.columns

        if isinstance(entities, str):
            entities = [entities]

        for entity in entities:
            score = {}
            score.update({
                name: metric(actuals[entity].values, forecasts[entity].values)
                for name, metric in metrics.items() if name != 'MASE'
            })

            if detailed:
                score['granularity'] = forecasts.index[1] - forecasts.index[0]
                score['prediction length'] = forecasts.shape[0]
                score['length of testing data'] = len(actuals.get_group(entity))

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
