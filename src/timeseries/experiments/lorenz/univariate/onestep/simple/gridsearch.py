from timeseries.data.lorenz.lorenz import univariate_lorenz
from timeseries.experiments.lorenz.functions.functions import grid_search
from timeseries.experiments.lorenz.univariate.onestep.simple.func import simple_configs, simple_forecast, \
    simple_fit
from timeseries.plotly.plot import plotly_time_series

if __name__ == '__main__':
    lorenz_df, train, test, t_train, t_test = univariate_lorenz()
    plotly_time_series(lorenz_df, features=['x'], title="Lorenz Attractor X", markers='lines')

    # max_length = len(data) - n_test
    max_avg_len = 1000
    # models configs
    cfg_list = simple_configs(max_avg_len)
    # grid search
    scores = grid_search(train, test, cfg_list, simple_forecast, simple_fit)
    print('done')
    # list top 3 configs
    for cfg, error in scores[:3]:
        print(cfg, error)