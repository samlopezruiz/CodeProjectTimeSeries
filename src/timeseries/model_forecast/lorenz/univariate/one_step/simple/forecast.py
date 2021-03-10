from timeseries.data.lorenz.lorenz import lorenz_system
from timeseries.model_forecast.lorenz.univariate.one_step.functions.functions import train_test_split, \
    walk_forward_step_forecast
from timeseries.model_forecast.lorenz.univariate.one_step.simple.func import simple_forecast, simple_creation
from timeseries.model_forecast.utils.forecast import one_step_forecast_df, multi_step_forecast_df
from timeseries.plotly.plot import plotly_time_series


if __name__ == '__main__':
    lorenz_df, xyz, t, _ = lorenz_system()
    test_size = 1000
    data = xyz[0]
    cfg = (1, 1, 'persist')
    train, test = train_test_split(data, test_size)
    t_train, t_test = train_test_split(t, test_size)
    model = simple_creation(train, cfg)
    pred = simple_forecast(model)

    frcst_df = one_step_forecast_df(train, test, pred, t_train, t_test, steps=100)
    plotly_time_series(frcst_df, title="Forecast")

#%%
    error, predictions = walk_forward_step_forecast(train, test, cfg, simple_forecast, simple_creation)
    df = multi_step_forecast_df(train, test, predictions, steps=500)
    plotly_time_series(df, title="PERSISTENT Walk-forward Forecast", markers='lines')

    print("MSE:", error)