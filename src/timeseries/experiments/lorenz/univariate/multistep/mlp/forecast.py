from timeseries.data.lorenz.lorenz import univariate_lorenz
from timeseries.experiments.lorenz.functions.functions import walk_forward_step_forecast
from timeseries.experiments.lorenz.univariate.multistep.mlp.func import mlp_multi_step_uv_fit, mlp_multi_step_uv_predict
from timeseries.experiments.utils.forecast import multi_step_forecast_df
from timeseries.experiments.utils.metrics import forecast_accuracy
from timeseries.plotly.plot import plotly_time_series

if __name__ == '__main__':
    # %% INPUTS
    lorenz_df, train, test, t_train, t_test = univariate_lorenz(granularity=5)
    name = "MLP"
    cfg = {"n_steps_in": 24, "n_steps_out": 10, "n_nodes": 500,
           "n_epochs": 100, "n_batch": 100}

    # %% STEP FORECAST
    model = mlp_multi_step_uv_fit(train, cfg)
    pred = mlp_multi_step_uv_predict(model, train, cfg)
    df = multi_step_forecast_df(train, test[:cfg["n_steps_out"]], pred, t_train, t_test[:cfg["n_steps_out"]], train_prev_steps=500)
    plotly_time_series(df, title=name+" Forecast")

    # %% PLOT WALK FORWARD FORECAST
    error, forecast = walk_forward_step_forecast(train, test, cfg, mlp_multi_step_uv_predict, mlp_multi_step_uv_fit,
                                                 steps=cfg["n_steps_out"], verbose=2)
    df = multi_step_forecast_df(train, test, forecast, train_prev_steps=500)
    plotly_time_series(df, title=name+" Walk-forward Forecast", markers='lines')
    print("MSE:", error)
    metrics = forecast_accuracy(forecast, test)
    print(metrics)