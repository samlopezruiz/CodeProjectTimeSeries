import datetime
import os

import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import probplot, moment, shapiro
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import adfuller

from timeseries.utils.files import new_dir

# plt.style.use('ggplot')

def plot_hurst(H, c, data, file_name='plot', save=False):
    f, ax = plt.subplots()
    ax.plot(data[0], c * data[0] ** H, color="deepskyblue")
    ax.scatter(data[0], data[1], color="purple")
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Time interval')
    ax.set_ylabel('R/S ratio')
    ax.grid(True)
    if save:
        new_dir('images')
        file_name = file_name + '_' + datetime.datetime.now().strftime("%Y_%m_%d_%H-%M") + ".png"
        plt.savefig(os.path.join('images', file_name))
    plt.show()


def plot_correlogram(x, lags=None, title=None, figsize=(14, 10), file_name='plot', save=False):
    x = x[~np.isnan(x)]
    lags = min(10, len(x) // 5) if lags is None else lags
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=figsize)
    axes[0][0].plot(x)
    axes[0][0].title.set_text('Time Series')
    axes[0][0].set_xlabel('Time')
    q_p = np.max(acorr_ljungbox(x, lags=lags)[1])
    y_label = 0.75
    stats = f'Box-Ljung: {np.max(q_p):>8.2f}\nShapiro: {shapiro(x)[1]:>11.2f}\nADF: {adfuller(x)[1]:>11.2f}'
    axes[0][0].text(x=.02, y=y_label, s=stats, transform=axes[0][0].transAxes)
    probplot(x, plot=axes[0][1])
    mean, var, skew, kurtosis = moment(x, moment=[1, 2, 3, 4])
    s = f'Mean: {mean:>12.2f}\nSD: {np.sqrt(var):>16.2f}\nSkew: {skew:12.2f}' #\nKurtosis:{kurtosis:9.2f}'
    axes[0][1].text(x=.02, y=y_label, s=s, transform=axes[0][1].transAxes)
    plot_acf(x=x, lags=lags, zero=False, ax=axes[1][0])
    plot_pacf(x, lags=lags, zero=False, ax=axes[1][1])
    axes[1][0].set_xlabel('Lag')
    axes[1][1].set_xlabel('Lag')
    fig.suptitle(title, fontsize=20)
    fig.tight_layout()
    fig.subplots_adjust(top=.9)

    if save:
        new_dir('images')
        file_name = file_name + '_' + datetime.datetime.now().strftime("%Y_%m_%d_%H-%M") + ".png"
        plt.savefig(os.path.join('images', file_name))
    plt.show()