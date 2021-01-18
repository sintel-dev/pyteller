import logging
import os
from datetime import datetime

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()


def convert_date(timelist):
    converted = list()
    for x in timelist:
        converted.append(datetime.fromtimestamp(x))
    return converted


def plot(dfs, output_path=None, labels=['actual', 'predicted'], frequency=None):
    """ Line plot for time series.

    This function plots time series
    Args:
        dfs (list or `pd.DataFrame`):
            List of time series in `pd.DataFrame`.
            Or a single dataframe. All dataframes must have the same shape.
        output_path (string):
            Optional. String of the path to save the figure
        labels (list of strings):
            Optional. List of strings for the legend entries. Default is ['actual','predicted']
        frequency (string):
            Optional. String of what frequency of tick marks for the x axis.

    """
    if isinstance(dfs, pd.DataFrame):
        dfs = [dfs]
    months = mdates.MonthLocator()  # every month
    days = mdates.DayLocator()  # every day
    hours = mdates.HourLocator()  # every day
    freq_dict = {
        'hour': '%H',
        "day": "%D",
        "month": "%M",
    }
    freq = freq_dict[frequency]
    if frequency is None:
        freq = "%H"
    tik_fmt = mdates.DateFormatter(freq)

    fig = plt.figure(figsize=(30, 6))
    ax = fig.add_subplot(111)

    for df in dfs:
        plt.plot(df.index, df.iloc[:, 0] / 100, linewidth=3)

    plt.title('Predicted and Actual', size=34)
    plt.ylabel('', size=30)
    plt.xlabel('Hour', size=30)
    plt.xticks(size=26)
    plt.yticks(size=26)

    ax.xaxis.set_major_formatter(tik_fmt)

    if labels:
        plt.legend(labels=labels, loc=1, prop={'size': 26})

    if output_path:
        os.path.join('figs', output_path)
        plt.savefig('figs/lstm.png')
    plt.show()


def logging_setup(verbosity=1, logfile=None, logger_name=None):
    logger = logging.getLogger(logger_name)
    log_level = (3 - verbosity) * 10
    fmt = '%(asctime)s - %(process)d - %(levelname)s - %(module)s - %(message)s'
    formatter = logging.Formatter(fmt)
    logger.setLevel(log_level)
    logger.propagate = False

    if logfile:
        file_handler = logging.FileHandler(logfile)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    else:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
