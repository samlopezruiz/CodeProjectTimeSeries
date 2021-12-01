from timeseries.data.lorenz.lorenz import univariate_lorenz
from timeseries.experiments.lorenz.functions.functions import walk_forward_step_forecast
from timeseries.experiments.lorenz.univariate.onestep.convlstm.func import convlstm_one_step_uv_fit, \
    convlstm_one_step_uv_predict
from timeseries.experiments.utils.forecast import multi_step_forecast_df, one_step_forecast_df
from timeseries.experiments.utils.metrics import forecast_accuracy
from timeseries.plotly.plot import plotly_time_series
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from timeseries.experiments.lorenz.multivariate.multistep.convlstm.func import convlstm_get_multi_step_mv_funcs
from timeseries.data.lorenz.lorenz import lorenz_wrapper
from timeseries.experiments.lorenz.functions.harness import eval_multi_step_forecast, run_multi_step_forecast
from timeseries.experiments.lorenz.functions.preprocessing import preprocess
#%%

if __name__ == '__main__':
    # %% GENERAL INPUTS
    in_cfg = {'steps': 3, 'save_results': False, 'verbose': 1, 'plot_title': True, 'plot_hist': False,
              'image_folder': 'images', 'results_folder': 'results',
              'detrend_ops': ['ln_return', ('ema_diff', 5), 'ln_return']}

    # MODEL AND TIME SERIES INPUTS
    name = "CONV-LSTM"
    input_cfg = {"variate": "multi", "granularity": 5, "noise": True, 'preprocess': True,
                 'trend': True, 'detrend': 'ln_return'}
    model_cfg = {"n_steps_out": 3, "n_steps_in": 10, "n_seq": 3, "n_kernel": 3,
                 "n_filters": 32, "n_nodes": 64, "n_batch": 64, "n_epochs": 30}
    functions = convlstm_get_multi_step_mv_funcs()

    # %% DATA
    lorenz_df, train, test, t_train, t_test = lorenz_wrapper(input_cfg)
    train_pp, test_pp, ss = preprocess(input_cfg, train, test)
    data_in = (train_pp, test_pp, train, test, t_train, t_test)

    # %% FORECAST
    # model, forecast_reconst, df = run_multi_step_forecast(name, input_cfg, model_cfg, functions, in_cfg, data_in, ss)

    # %% WALK FORWARD FORECAST
    metrics, forecast = eval_multi_step_forecast(name, input_cfg, model_cfg, functions, in_cfg, data_in, ss)

if __name__ == '__main__':
    # %% INPUT
    lorenz_df, train, test, t_train, t_test = univariate_lorenz(granularity=5)
    name = "CONV-LSTM"
    cfg = {"n_seq": 3, "n_steps_in": 12, "n_filters": 256,
           "n_kernel": 3, "n_nodes": 200, "n_epochs": 100, "n_batch": 100}

    #%% FORECAST
    model = convlstm_one_step_uv_fit(train, cfg)
    pred = convlstm_one_step_uv_predict(model, train, cfg)
    df = one_step_forecast_df(train, test[:1], pred, t_train, t_test[:1], train_prev_steps=500)
    plotly_time_series(df, title=name+" Forecast")

    # %% PLOT WALK FORWARD FORECAST
    error, forecast = walk_forward_step_forecast(train, test, cfg, convlstm_one_step_uv_predict, convlstm_one_step_uv_fit, steps=1, verbose=2)
    df = multi_step_forecast_df(train, test, forecast, train_prev_steps=500)
    plotly_time_series(df, title=name+" Walk-forward Forecast", markers='lines')
    print("MSE:", error)
    metrics = forecast_accuracy(forecast, test)
    print(metrics)