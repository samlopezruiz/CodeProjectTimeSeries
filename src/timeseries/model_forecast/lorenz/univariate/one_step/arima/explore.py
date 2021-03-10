import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import pacf, acf, kpss
from timeseries.data.lorenz.lorenz import lorenz_system
from timeseries.model_forecast.lorenz.univariate.one_step.functions.functions import train_test_split
from timeseries.plotly.plot import plotly_time_series, plotly_acf_pacf


def kpss_test(series, **kw):
    statistic, p_value, n_lags, critical_values = kpss(series, **kw)
    # Format Output
    print(f'KPSS Statistic: {statistic}')
    print(f'p-value: {p_value}')
    print(f'num lags: {n_lags}')
    print('Critial Values:')
    for key, value in critical_values.items():
        print(f'   {key} : {value}')
    print(f'Result: The series is {"not " if p_value < 0.05 else ""}stationary')


if __name__ == '__main__':
    lorenz_df, xyz, t = lorenz_system()
    test_size = 1000
    x = lorenz_df['x']
    x = x[x.index > 15]
    data = np.array(x)
    train, test = train_test_split(data, test_size)
    t_train, t_test = train_test_split(t, test_size)

#%%
    lags = 100
    df_pacf = pacf(x, nlags=lags)
    df_acf = acf(x, nlags=lags)
    x_diff = x.diff().diff().dropna()
    pacf_diff = pacf(x_diff, nlags=lags)
    acf_diff = acf(x_diff, nlags=lags)
    kpss_test(x_diff, nlags=lags)

#%% PLOT ACF and PACF
    frame = pd.DataFrame(data=[x_diff]).transpose()
    frame.columns = ['x_diff']
    plotly_time_series(frame, markers='lines')
    plotly_acf_pacf(df_acf, df_pacf)
    plotly_acf_pacf(acf_diff, pacf_diff)
