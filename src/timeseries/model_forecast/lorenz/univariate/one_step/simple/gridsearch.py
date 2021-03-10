from timeseries.data.lorenz.lorenz import lorenz_system
from timeseries.model_forecast.lorenz.univariate.one_step.functions.functions import grid_search
from timeseries.model_forecast.lorenz.univariate.one_step.simple.func import simple_configs, simple_forecast, \
    simple_creation
from timeseries.plotly.plot import plotly_time_series

if __name__ == '__main__':
    lorenz_df, xyz, t, _ = lorenz_system()
    test_size = 1000
    data = xyz[0]
    plotly_time_series(lorenz_df, features=['x'], title="Lorenz Attractor X", markers='lines')

    # max_length = len(data) - n_test
    max_avg_len = 1000
    # model_forecast configs
    cfg_list = simple_configs(max_avg_len)
    # grid search
    scores = grid_search(data, cfg_list, test_size, simple_forecast, simple_creation)
    print('done')
    # list top 3 configs
    for cfg, error in scores[:3]:
        print(cfg, error)