import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from dateutil.parser import parse
class NaiveForecaster:

    def __init__(self, offset=1):
        super(NaiveForecaster, self).__init__()
        self.offset = offset
