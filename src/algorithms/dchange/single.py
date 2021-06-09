from algorithms.dchange.func import direct_change
from timeseries.data.lorenz.lorenz import multivariate_lorenz
from timeseries.plotly.plot import plotly_time_series

if __name__ == '__main__':
    dc_cfg = {'delta_t': 0.01, 'delta_y': 5}

    lorenz_df, train, test, t_train, t_test = multivariate_lorenz(test_perc=25, t_ini=15, granularity=1, end_time=1000,
                              y_col=0, positive_offset=True, noise=True, sigma=1.5, trend=True)

    lorenz_df = lorenz_df[lorenz_df.index > 15]
    lorenz_dc = direct_change(lorenz_df['x'], dc_cfg)
    #
    plotly_time_series(lorenz_dc, rows=list(range(lorenz_dc.shape[1])), title='name', markers='lines+markers')