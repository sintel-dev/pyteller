# -*- coding: utf-8 -*-

"""Top-level package for pyteller."""

__author__ = 'MIT Data To AI Lab'
__email__ = 'dailabmit@gmail.com'
__version__ = '0.1.1.dev0'

import os

from mlblocks import find_pipelines

from pyteller.core import Pyteller

_BASE_PATH = os.path.abspath(os.path.dirname(__file__))
MLBLOCKS_PIPELINES = os.path.join(_BASE_PATH, 'pipelines')
MLBLOCKS_PRIMITIVES = [os.path.join(_BASE_PATH, 'primitives', 'jsons')]


def get_pipelines(filter=None):
    pipelines = find_pipelines('pyteller')
    if filter:
        pipelines = [pipeline for pipeline in pipelines if filter.lower() in pipeline.lower()]

    return pipelines


__all__ = ('Pyteller', 'get_pipelines')
