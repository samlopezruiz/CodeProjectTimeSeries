from timeseries.data.lorenz.lorenz import univariate_lorenz
from timeseries.models.lorenz.functions.functions import walk_forward_step_forecast
from timeseries.models.lorenz.univariate.onestep.simple.func import simple_forecast, simple_fit
from timeseries.models.utils.forecast import one_step_forecast_df, multi_step_forecast_df
from timeseries.models.utils.metrics import forecast_accuracy
from timeseries.plotly.plot import plotly_time_series


if __name__ == '__main__':
    lorenz_df, train, test, t_train, t_test = univariate_lorenz()

    cfg = (1, 1, 'persist')
    model = simple_fit(train, cfg)
    one_step_frcst = simple_forecast(model)

    frcst_df = one_step_forecast_df(train, test, one_step_frcst, t_train, t_test, train_prev_steps=100)
    plotly_time_series(frcst_df, title="Forecast")

#%%
    error, forecast = walk_forward_step_forecast(train, test, cfg, simple_forecast, simple_fit)
    df = multi_step_forecast_df(train, test, forecast, train_prev_steps=500)
    plotly_time_series(df, title="PERSISTENT Walk-forward Forecast", markers='lines')

    print("RMSE:", error)
    metrics = forecast_accuracy(forecast, test)
    print(metrics)
