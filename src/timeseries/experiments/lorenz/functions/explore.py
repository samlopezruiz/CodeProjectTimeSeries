from timeseries.data.lorenz.lorenz import univariate_lorenz
from timeseries.experiments.lorenz.functions.functions import walk_forward_step_forecast
from timeseries.experiments.utils.forecast import multi_step_forecast_df
from timeseries.experiments.utils.metrics import forecast_accuracy
from timeseries.plotly.plot import plotly_time_series
import pandas as pd


def explore_model(forecast_fn, creation_fn, cfg, granularity=1, end_time=100, name="errors"):
    lorenz_df, train, test, t_train, t_test = univariate_lorenz(granularity=granularity, end_time=end_time)
    error, forecast = walk_forward_step_forecast(train, test, cfg, forecast_fn, creation_fn)
    df = multi_step_forecast_df(train, test, forecast, t_train, t_test, train_prev_steps=500)
    plotly_time_series(df, title=name+" Walk-forward Forecast", markers='lines')
    metrics = forecast_accuracy(forecast, test)
    return pd.DataFrame.from_dict(metrics, orient='index', columns=[name])