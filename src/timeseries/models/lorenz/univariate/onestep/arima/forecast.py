import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from timeseries.data.lorenz.lorenz import univariate_lorenz
from timeseries.models.lorenz.functions.functions import walk_forward_step_forecast
from timeseries.models.lorenz.univariate.onestep.arima.func import arima_forecast, arima_creation
from timeseries.models.utils.forecast import multi_step_forecast_df
from timeseries.models.utils.metrics import forecast_accuracy
from timeseries.plot.util import plot_correlogram
from timeseries.plotly.plot import plotly_time_series

if __name__ == '__main__':
    lorenz_df, train, test, t_train, t_test = univariate_lorenz(granularity=5)

    # %% FIT MODEL
    cfg = (9, 1, 5)
    model = SARIMAX(endog=train, order=cfg, seasonal_order=(0, 0, 0, 0), trend='n', enforce_stationarity=False)
    model_fit = model.fit(disp=0)
    print(model_fit.summary())
    plot_correlogram(pd.Series(model_fit.resid[15:]), lags=10)

    # %% PLOT FORECAST
    pred_size = 1
    pred = model_fit.forecast(steps=pred_size)
    df = multi_step_forecast_df(train, test[:pred_size], pred, t_train, t_test[:pred_size], train_prev_steps=500)
    plotly_time_series(df, title="ARIMA Forecast")

    # %% PLOT WALK FORWARD FORECAST
    error, forecast = walk_forward_step_forecast(train, test, cfg, arima_forecast, arima_creation, steps=pred_size, verbose=2)
    df = multi_step_forecast_df(train, test, forecast, train_prev_steps=500)
    plotly_time_series(df, title="ARIMA Walk-forward Forecast", markers='lines')
    print("MSE:", error)
    metrics = forecast_accuracy(forecast, test)
    print(metrics)