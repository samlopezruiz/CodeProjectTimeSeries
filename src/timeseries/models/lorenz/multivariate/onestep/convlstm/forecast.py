from timeseries.data.lorenz.lorenz import multivariate_lorenz, lorenz_wrapper
from timeseries.models.lorenz.functions.functions import walk_forward_step_forecast
from timeseries.models.lorenz.multivariate.onestep.convlstm.func import convlstm_one_step_mv_predict, \
    convlstm_one_step_mv_fit
from timeseries.models.utils.forecast import multi_step_forecast_df, one_step_forecast_df
from timeseries.models.utils.metrics import forecast_accuracy
from timeseries.plotly.plot import plotly_time_series

if __name__ == '__main__':
    # %% INPUT
    input_cfg = {"variate": "multi", "granularity": 5, "noise": True}
    lorenz_df, train, test, t_train, t_test = lorenz_wrapper(input_cfg)
    name = "CONV-LSTM"
    cfg = {"n_seq": 3, "n_steps_in": 12, "n_filters": 256,
           "n_kernel": 3, "n_nodes": 200, "n_epochs": 100, "n_batch": 100}

    # %% FORECAST
    # model = convlstm_one_step_mv_fit(train, cfg)
    # # history doesn't contain last y column
    # history = train[:, :-1]
    # pred = convlstm_one_step_mv_predict(model, history, cfg)
    # df = one_step_forecast_df(train[:, -1], test[:1, -1], pred, t_train, t_test[:1], train_prev_steps=500)
    # plotly_time_series(df, title=name + " Forecast")

    # %% PLOT WALK FORWARD FORECAST
    error, forecast = walk_forward_step_forecast(train, test, cfg, convlstm_one_step_mv_predict, convlstm_one_step_mv_fit,
                                                 steps=1, verbose=2)
    df = multi_step_forecast_df(train[:, -1], test[:, -1], forecast, train_prev_steps=500)
    plotly_time_series(df, title=name+" | "+str(input_cfg)+" | "+str(cfg), markers='lines')
    metrics = forecast_accuracy(forecast, test[:, -1])
    print(metrics)