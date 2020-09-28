import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from pyteller.data import load_signal
signal = 'nyc_taxi'
# load signal
df = load_signal(signal)
