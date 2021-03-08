import numpy as np
import pandas as pd

from timeseries.data.lorenz.lorenz import Lorenz
from timeseries.model.univariate.explore.functions import grid_search, train_test_split
from timeseries.model.univariate.naive.naive import simple_configs, simple_forecast
from timeseries.plot.forecast_plots import plot_series
import matplotlib.pyplot as plt

if __name__ == '__main__':
    start_time = 0
    end_time = 100
    t = np.linspace(start_time, end_time, end_time * 100)

    lorenz_sys = Lorenz(sigma=10., rho=28., beta=8. / 3.)
    lorenz_sys.solve(t)
    xyz = lorenz_sys.get_time_series()
    data = xyz[0]
    plt.plot(data)
    plt.show()
    # data split
    n_test = 1000
    # model configs
    # max_length = len(data) - n_test
    max_length = 1000
    cfg_list = simple_configs(max_length)
    # grid search
    scores = grid_search(data, cfg_list, n_test, simple_forecast)
    print('done')
    # list top 3 configs
    for cfg, error in scores[:3]:
        print(cfg, error)

    # %% PLOT FORECAST
    cfg = (1, 1, 'persist')
    n_steps = 100
    train, test = train_test_split(data, n_test)
    yhat = simple_forecast(train, cfg)
    plot_series(n_steps, train[-n_steps:], y=test[0], y_pred=yhat)