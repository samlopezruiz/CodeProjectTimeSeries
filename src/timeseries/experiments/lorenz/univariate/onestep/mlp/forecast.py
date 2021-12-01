from timeseries.data.lorenz.lorenz import univariate_lorenz
from timeseries.experiments.lorenz.functions.functions import walk_forward_step_forecast
from timeseries.experiments.lorenz.univariate.onestep.mlp.func import mlp_one_step_uv_fit, mlp_one_step_uv_predict
from timeseries.experiments.utils.forecast import multi_step_forecast_df, one_step_forecast_df
from timeseries.experiments.utils.metrics import forecast_accuracy
from timeseries.plotly.plot import plotly_time_series

if __name__ == '__main__':
    # %% INPUTS
    lorenz_df, train, test, t_train, t_test = univariate_lorenz(granularity=5)
    name = "MLP"
    cfg = {"n_steps_in": 24, "n_nodes": 500, "n_epochs": 100, "n_batch": 100}

    #%% STEP FORECAST
    model = mlp_one_step_uv_fit(train, cfg)
    pred = mlp_one_step_uv_predict(model, train, cfg)
    df = one_step_forecast_df(train, test[:1], pred, t_train, t_test[:1], train_prev_steps=500)
    plotly_time_series(df, title=name+" Forecast")

    # %% PLOT WALK FORWARD FORECAST
    error, forecast = walk_forward_step_forecast(train, test, cfg, mlp_one_step_uv_predict, mlp_one_step_uv_fit, steps=1, verbose=2)
    df = multi_step_forecast_df(train, test, forecast, train_prev_steps=500)
    plotly_time_series(df, title=name+" Walk-forward Forecast", markers='lines')
    print("MSE:", error)
    metrics = forecast_accuracy(forecast, test)
    print(metrics)