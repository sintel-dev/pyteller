# -*- coding: utf-8 -*-

"""Top-level package for pyteller."""

__author__ = 'MIT Data To AI Lab'
__email__ = 'dailabmit@gmail.com'
__version__ = '0.1.0'

import os
_BASE_PATH = os.path.abspath(os.path.dirname(__file__))
MLBLOCKS_PRIMITIVES = os.path.join(_BASE_PATH, 'primitives', 'jsons')
# MLBLOCKS_PIPELINES = os.path.join(_BASE_PATH, 'pipelines')
MLBLOCKS_PIPELINES = tuple(
    dirname
    for dirname, _, _ in os.walk(os.path.join(_BASE_PATH, 'pipelines'))
)
