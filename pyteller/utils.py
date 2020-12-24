from datetime import datetime

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
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
    months = mdates.MonthLocator()  # every month
    days = mdates.DayLocator()  # every day
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
