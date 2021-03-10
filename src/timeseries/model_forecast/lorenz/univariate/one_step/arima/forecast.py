import numpy as np
import pandas as pd
from pmdarima import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import pacf, acf, kpss
from timeseries.data.lorenz.lorenz import lorenz_system
from timeseries.model_forecast.lorenz.univariate.one_step.arima.func import arima_forecast, arima_creation
from timeseries.model_forecast.utils.forecast import multi_step_forecast_df
from timeseries.plot.util import plot_correlogram
from timeseries.model_forecast.lorenz.univariate.one_step.functions.functions import train_test_split, \
    walk_forward_validation, measure_rmse, walk_forward_step_forecast
from timeseries.plotly.plot import plotly_time_series, plotly_acf_pacf

if __name__ == '__main__':
    lorenz_df, xyz, t, _ = lorenz_system()
    test_size = 2000
    x = lorenz_df['x']
    x = x[x.index > 15]
    data = np.array(x)
    train, test = train_test_split(data, test_size)
    t_train, t_test = train_test_split(t, test_size)

    # %% FIT MODEL
    cfg = (9, 1, 5)
    model = SARIMAX(endog=train, order=cfg, seasonal_order=(0, 0, 0, 0), trend='n', enforce_stationarity=False)
    model_fit = model.fit(disp=0)
    print(model_fit.summary())
    # plot_correlogram(pd.Series(model_fit.resid[15:]), lags=10)

    # %% PLOT FORECAST
    pred_size = 50
    pred = model_fit.forecast(steps=pred_size)
    df = multi_step_forecast_df(train, test[:pred_size], pred, t_train, t_test[:pred_size], steps=500)
    plotly_time_series(df, title="ARIMA Forecast")

    # %% PLOT WALK FORWARD FORECAST
    error, predictions = walk_forward_step_forecast(train, test, cfg, arima_forecast, arima_creation, steps=pred_size, verbose=True)
    df = multi_step_forecast_df(train, test, predictions, steps=500)
    plotly_time_series(df, title="ARIMA Walk-forward Forecast", markers='lines')
    print("MSE:", error)