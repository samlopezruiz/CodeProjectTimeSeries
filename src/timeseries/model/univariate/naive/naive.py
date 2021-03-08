# example of an average forecast
from numpy import mean
from numpy import median


def simple_forecast(history, config):
    n, offset, avg_type = config
    # persist value, ignore other config
    if avg_type == 'persist':
        return history[-n]
    # collect values to average
    values = list()
    if offset == 1:
        values = history[-n:]
    else:
        # skip bad configs
        if n * offset > len(history):
            raise Exception('Config beyond end of data: %d %d' % (n, offset))
        # try and collect n values using offset
        for i in range(1, n + 1):
            ix = i * offset
            values.append(history[-ix])
    # check if we can average
    if len(values) < 2:
        raise Exception('Cannot calculate average')
    # mean of last n values
    if avg_type == 'mean':
        return mean(values)
    # median of last n values
    return median(values)


# create a set of simple configs to try
def simple_configs(max_length, offsets=[1]):
    configs = list()
    for i in range(1, max_length + 1):
        for o in offsets:
            for t in ['persist', 'mean', 'median']:
                cfg = [i, o, t]
                configs.append(cfg)
    return configs


if __name__ == '__main__':
    import pandas as pd
    from timeseries.model.univariate.explore.functions import grid_search, train_test_split
    from timeseries.plot.forecast_plots import plot_series
    import matplotlib.pyplot as plt

    series = pd.read_csv('../../../data/tests/daily-total-female-births.csv', header=0, index_col=0)
    data = series.values
    plt.plot(data)
    plt.show()
    # data split
    n_test = 165
    # model configs
    max_length = len(data) - n_test
    cfg_list = simple_configs(max_length)
    # grid search
    scores = grid_search(data, cfg_list, n_test, simple_forecast)
    print('done')
    # list top 3 configs
    for cfg, error in scores[:3]:
        print(cfg, error)

    # %% PLOT FORECAST
    cfg = (22, 1, 'mean')
    n_steps = 100
    train, test = train_test_split(data, n_test)
    yhat = simple_forecast(train, cfg)
    plot_series(n_steps, train[-n_steps:], yhat, test[0])

