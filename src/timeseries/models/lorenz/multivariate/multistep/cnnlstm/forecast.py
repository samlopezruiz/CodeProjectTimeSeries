from timeseries.data.lorenz.lorenz import univariate_lorenz, multivariate_lorenz
from timeseries.models.lorenz.functions.functions import walk_forward_step_forecast
from timeseries.models.lorenz.multivariate.multistep.cnnlstm.func import cnnlstm_multi_step_mv_fit, \
    cnnlstm_multi_step_mv_predict
from timeseries.models.lorenz.multivariate.onestep.cnnlstm.func import cnnlstm_one_step_mv_fit, \
    cnnlstm_one_step_mv_predict
from timeseries.models.lorenz.univariate.onestep.cnnlstm.func import cnnlstm_one_step_uv_fit, \
    cnnlstm_one_step_uv_predict
from timeseries.models.utils.forecast import multi_step_forecast_df, one_step_forecast_df
from timeseries.models.utils.metrics import forecast_accuracy
from timeseries.plotly.plot import plotly_time_series

if __name__ == '__main__':
    # %% INPUT
    lorenz_df, train, test, t_train, t_test = multivariate_lorenz(positive_offset=False, granularity=5, noise=False)
    name = "CNN-LSTM"
    cfg = {"n_seq": 3, "n_steps_in": 12, "n_steps_out": 20, "n_filters": 64,
           "n_kernel": 3, "n_nodes": 200, "n_epochs": 100, "n_batch": 100}

    # %% FORECAST
    # model = cnnlstm_multi_step_mv_fit(train, cfg)
    # # history doesn't contain last y column
    # history = train[:, :-1]
    # pred = cnnlstm_multi_step_mv_predict(model, history, cfg)
    # df = multi_step_forecast_df(train[:, -1], test[:cfg['n_steps_out'], -1], pred, t_train,
    #                           t_test[:cfg['n_steps_out']], train_prev_steps=500)
    # plotly_time_series(df, title=name + " Forecast")

    # %% PLOT WALK FORWARD FORECAST
    error, forecast = walk_forward_step_forecast(train, test, cfg, cnnlstm_multi_step_mv_predict, cnnlstm_multi_step_mv_fit,
                                                 steps=cfg['n_steps_out'], verbose=2)
    df = multi_step_forecast_df(train[:, -1], test[:, -1], forecast, train_prev_steps=500)
    plotly_time_series(df, title=name + " Walk-forward Forecast", markers='lines')
    metrics = forecast_accuracy(forecast, test[:, -1])
    print(metrics)