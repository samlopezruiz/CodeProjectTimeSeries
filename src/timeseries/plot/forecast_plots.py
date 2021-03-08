import numpy as np
import os
import matplotlib as mpl
import matplotlib.pyplot as plt


def plot_series(n_steps, series, y=None, y_pred=None, x_label="$t$", y_label="$x(t)$"):
    plt.plot(series, ".-")
    if y is not None:
        plt.plot(n_steps, y, "bo-", markersize=5, label="Actual")
    if y_pred is not None:
        plt.plot(n_steps, y_pred, "rx-", label="Forecast")
    plt.grid(True)
    plt.legend(fontsize=14)
    if x_label:
        plt.xlabel(x_label, fontsize=16)
    if y_label:
        plt.ylabel(y_label, fontsize=16, rotation=0)
    # plt.hlines(0, 0, n_steps, linewidth=1)
    # plt.axis([0, n_steps + 1, min(series), 1])
    plt.show()