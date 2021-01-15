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


def plot(dfs, output_path, labels=None):
    """ Line plot for time series.

    This function plots time series
    Args:
        dfs (list or `pd.DataFrame`): List of time series in `pd.DataFrame`.
            Or a single dataframe. All dataframes must have the same shape.

    """
    if isinstance(dfs, pd.DataFrame):
        dfs = [dfs]
    # months = mdates.MonthLocator()  # every month
    # days = mdates.DayLocator()  # every day
    hours = mdates.HourLocator()  # every day

    month_fmt = mdates.DateFormatter('%H')

    fig = plt.figure(figsize=(30, 6))
    ax = fig.add_subplot(111)

    for df in dfs:
        plt.plot(df.index, df.iloc[:, 0] / 100, linewidth=3)

    plt.title('Normalized Demand', size=34)
    plt.ylabel('', size=30)
    plt.xlabel('Hour', size=30)
    plt.xticks(size=26)
    plt.yticks(size=26)
    # plt.xlim([time[0], time[-1]])

    # format xticks
    ax.xaxis.set_major_locator(hours)
    ax.xaxis.set_major_formatter(month_fmt)
    # ax.xaxis.set_minor_locator(days)

    # format yticks
    # ylabels = ['{:,.0f}'.format(x) + 'K' for x in ax.get_yticks() / 1000]
    # ax.set_yticklabels(ylabels)

    if labels:
        plt.legend(labels=labels, loc=1, prop={'size': 26})
    os.path.join('figs', output_path)
    plt.savefig('figs/lstm.png')
    plt.show()

import logging


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
