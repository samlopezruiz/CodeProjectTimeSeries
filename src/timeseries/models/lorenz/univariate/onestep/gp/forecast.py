from timeseries.data.lorenz.lorenz import univariate_lorenz
from timeseries.models.lorenz.functions.functions import walk_forward_step_forecast, walk_forward_step_forecast_werrs
from timeseries.models.lorenz.univariate.onestep.gp.func import gp_one_step_uv_fit, \
    gp_one_step_uv_predict
from timeseries.models.utils.forecast import one_step_forecast_df, multi_step_forecast_df
from timeseries.models.utils.metrics import forecast_accuracy
from timeseries.plotly.plot import plotly_time_series

if __name__ == "__main__":
    # %% INPUTS
    lorenz_df, train, test, t_train, t_test = univariate_lorenz(granularity=5)
    name = "GP"
    cfg = {"in_steps": 10, "out_steps": 1, "ngen": 40}

    # %% FORECAST
    # model = gp_one_step_uv_fit(train, cfg)
    # pred = gp_one_step_uv_predict(model, train, [0 for _ in range(cfg['in_steps'])], cfg)
    # df = one_step_forecast_df(train, test[:1], pred, t_train, t_test[:1], train_prev_steps=500)
    # plotly_time_series(df, title=name + " Forecast")

    # %% PLOT WALK FORWARD FORECAST
    error, forecast = walk_forward_step_forecast_werrs(train, test, cfg, gp_one_step_uv_predict, gp_one_step_uv_fit,
                                                 steps=1, verbose=2)
    df = multi_step_forecast_df(train, test, forecast, train_prev_steps=500)
    plotly_time_series(df, title=name + " Walk-forward Forecast", markers='lines')
    print("MSE:", error)
    metrics = forecast_accuracy(forecast, test)
    print(metrics)