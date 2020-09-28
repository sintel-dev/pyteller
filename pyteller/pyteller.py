# -*- coding: utf-8 -*-

"""Main module."""
import json
import logging
import os
import pickle
from typing import List, Union

import pandas as pd
from mlblocks import MLPipeline

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
                 hyperparameters: dict = None):
        self._pipeline = pipeline
        self._hyperparameters = hyperparameters
        self._mlpipeline = self._get_mlpipeline()
        self._fitted = False
