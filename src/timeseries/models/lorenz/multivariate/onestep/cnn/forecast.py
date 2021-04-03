from timeseries.data.lorenz.lorenz import univariate_lorenz, multivariate_lorenz, lorenz_wrapper
from timeseries.models.lorenz.functions.functions import walk_forward_step_forecast
from timeseries.models.lorenz.multivariate.onestep.cnn.func import cnn_one_step_mv_fit, cnn_one_step_mv_predict
from timeseries.models.lorenz.multivariate.onestep.mlp.func import mlp_one_step_mv_predict
from timeseries.models.lorenz.univariate.onestep.cnn.func import cnn_one_step_uv_fit, cnn_one_step_uv_predict
from timeseries.models.utils.forecast import multi_step_forecast_df, one_step_forecast_df
from timeseries.models.utils.metrics import forecast_accuracy
from timeseries.plotly.plot import plotly_time_series

if __name__ == '__main__':
    #%% INPUTS
    input_cfg = {"variate": "multi", "granularity": 5, "noise": True}
    lorenz_df, train, test, t_train, t_test = lorenz_wrapper(input_cfg)

    name = "CNN"
    cfg = {"n_steps_in": 36, "n_filters": 256, "n_kernel": 3,
           "n_epochs": 100, "n_batch": 100}

    #%% FORECAST
    # model = cnn_one_step_mv_fit(train, cfg)
    # # history doesn't contain last y column
    # history = train[:, :-1]
    # pred = cnn_one_step_mv_predict(model, history, cfg)
    # df = one_step_forecast_df(train[:, -1], test[:1, -1], pred, t_train, t_test[:1], train_prev_steps=500)
    # plotly_time_series(df, title=name + " Forecast")

    #%% PLOT WALK FORWARD FORECAST
    error, forecast = walk_forward_step_forecast(train, test, cfg, cnn_one_step_mv_predict, cnn_one_step_mv_fit,
                                                 steps=1, verbose=2)
    df = multi_step_forecast_df(train[:, -1], test[:, -1], forecast, train_prev_steps=500)
    plotly_time_series(df, title=name+" | "+str(input_cfg)+" | "+str(cfg), markers='lines')
    print("MSE:", error)
    metrics = forecast_accuracy(forecast, test[:, -1])
    print(metrics)