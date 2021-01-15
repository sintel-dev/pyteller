# -*- coding: utf-8 -*-

"""Main module."""
import json
import os
import pickle

import pandas as pd
import numpy as np
from mlblocks import MLPipeline

# from pyteller.data import egest_data, ingest_data
from pyteller.evaluation import METRICS_NORM as METRICS


class Pyteller:
    def _load_pipeline(self, pipeline, hyperparams=None):
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
            target_col_primitives = pipeline_args['target_column']
            for i in target_col_primitives:
                if i in pipeline['init_params'].keys():
                    dict = pipeline['init_params'][i]
                else:
                    dict = {}
                dict[target_col_primitives[i]] = self.target_column
                pipeline['init_params'][i] = dict
        mlpipeline = MLPipeline(pipeline)
        if self._hyperparameters:
            mlpipeline.set_hyperparameters(self._hyperparameters)
        return mlpipeline

    def __init__(self,
                 pipeline=None,
                 hyperparameters=None,
                 pred_length=None,
                 offset=None):
        self._pipeline = pipeline
        self._hyperparameters = hyperparameters
        self.pred_length = pred_length
        self.offset = offset
        self._fitted = False

    def fit(self,
            data=None,
            timestamp_col=None,
            target_signal=None,
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
        self.target_signal = target_signal
        self.timestamp_col = timestamp_col
        self.static_variables = static_variables
        self.entity_cols = entity_col
        self.entities = entities
        self.train_size = train_size

        train = ingest_data(self,
                            data=data,
                            timestamp_col=self.timestamp_col,
                            signal=self.target_signal,
                            static_variables=self.static_variables,
                            entity_col=self.entity_cols,
                            entities=self.entities
                            )
        self._mlpipeline = self._get_mlpipeline()
        self._mlpipeline.fit(X=train,
                             pred_length=self.pred_length,
                             offset=self.offset,
                             freq=self.freq,
                             entities=self.entities
                             )

        self._fitted = True
        print('The pipeline is fitted')

    def forecast(self, data=None):
        """Forecast input data on a trained model.

        Args:
            data (DataFrame):
                Input data, passed as a ``pandas.DataFrame`` containing
                exactly two columns: timestamp and value.

        Returns:
            DataFrame or tuple:
                A tuple containing the input data followed by the predictions which both contain
                values only at the same timestamps
        """
        test = ingest_data(self,
                           data=data,
                           timestamp_col=self.timestamp_col,
                           signal=self.target_signal,
                           static_variables=self.static_variables,
                           entity_col=self.entity_cols,
                           entities=self.entities,
                           )

        prediction = self._mlpipeline.predict(
            X=test,
            pred_length=self.pred_length,
            offset=self.offset,
            freq=self.freq,
            entities=self.entities
        )
        actual, prediction = egest_data(test, prediction)

        return actual, prediction

    def evaluate(self, forecast,
                 train_data=None,
                 test_data=None,
                 detailed=False,
                 metrics=METRICS):
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
                If not given, it defaults to all the Orion metrics.

        Returns:
            Series:
                ``pandas.Series`` containing one element for each
                metric applied, with the metric name as index.
        """

        metrics_ = {}
        if isinstance(metrics, str):
            metrics = [metrics]
        for metric in metrics:
            metrics_[metric] = METRICS[metric]
        scores = list()

        if isinstance(self.entities, str):
            self.entities = [self.entities]
        for entity in self.entities:
            score = {}
            score.update({
                metric: METRICS[metric](test_data[entity].values, forecast[entity].values)
                # metric: METRICS[metric](test_data[entity].values, forecast[entity].values[:, 0])
                for metric in metrics_ if metric != 'MASE'
            })

            if detailed:
                score['granularity'] = self.freq
                score['prediction length'] = forecast.shape[0]
                score['length of training data'] = len(train_data.get_group(entity))
                score['length of testing data'] = len(test_data.get_group(entity))
            scores.append(score)

        return pd.DataFrame(scores, index=self.entities).transpose()

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

        Raises:
            ValueError:
                If the serialized object is not a pyteller instance.
        """
        with open(path, 'rb') as pickle_file:
            pyteller = pickle.load(pickle_file)
            # if not isinstance(orion, cls):
            #     raise ValueError('Serialized object is not a pyteller instance')

            return pyteller

def ingest_data(self,
                data,
                timestamp_col=None,
                entity_col=None,
                entities=None,
                signal=None,
                dynamic_variables=None,
                static_variables=None,
                ):
    # Fix if the user specified multiple targets. They should be specified as multiple entities
    entities = signal if isinstance(signal, list) else entities
    signal = None if isinstance(signal, list) else signal

    columns = {
        'signal': signal,
        'timestamp': timestamp_col,
        'entity': entity_col,
        'dynamic_variable': dynamic_variables,
        'static_variable': static_variables
    }
    columns = {k: v for k, v in columns.items() if v is not None}

    df = pd.DataFrame()
    for key in columns:
        df[key] = data[columns[key]]

    # Scenario 1: (longform) user specifies entity column and target variable column
    if 'entity' in columns:
        df['entity'] = df['entity'].astype(str)  # Make all entities strings
        all_entities = df.entity.unique()  # Find the unique values in the entity column
        all_entities = [x for x in list(all_entities) if x != 'nan']
        self.entities = all_entities  # entities are the unique values in specified entity column

        # Make the long form into flatform by having entities as columns
        df = df.pivot(index='timestamp', columns='entity')['signal'].reset_index()

        # Scenario 1b User specifies certain entities from entity column
        if entities is not None:
            entities = [entities] if isinstance(entities, str) else entities
            to_remove = list(all_entities)
            to_remove = list(set(to_remove) - set(entities))  # Don't remove it
            self.entities = entities

            df = df.drop(to_remove, axis=1)

    # Scenario 2: (flatform) user specifies one signal
    elif signal is not None:
        self.entities = [signal]
        df = df.rename(columns={'signal': signal})
        if True in df.duplicated('timestamp'):
            raise ValueError('Multiple observations per timestamp')

    # Scenario 3: (flatform) user specifies multiple entities but there is no entity column
    else:
        self.entities = entities
        for entity in entities:
            df[entity] = data[[entity]]

    # Convert to epoch time
    if df['timestamp'].dtypes != 'float' and df['timestamp'].dtypes != 'int':
        df['timestamp'] = pd.to_datetime(df['timestamp']).values.astype(np.int64) // 1e9
    df = df.sort_values('timestamp')
    self.freq = df['timestamp'][1] - df['timestamp'][0]
    if isinstance(self.entities, str):
        self.target_column = [0]
    else:
        self.target_column = list(range(len(self.entities)))
    return df


def egest_data(test, prediction):
    if prediction.index.dtype == 'float' or prediction.index.dtype == 'int':
        prediction.index = pd.to_datetime(prediction.index.values * 1e9)
    else:
        prediction.index = pd.to_datetime(prediction.index)

    if test['timestamp'].dtypes == 'float' or test['timestamp'].dtypes == 'int':
        test['timestamp'] = pd.to_datetime(test['timestamp'] * 1e9)
    else:
        test['timestamp'] = pd.to_datetime(test['timestamp'])

    actual = test.set_index('timestamp')
    actual = actual[actual.index.isin(prediction.index)]
    return actual, prediction
