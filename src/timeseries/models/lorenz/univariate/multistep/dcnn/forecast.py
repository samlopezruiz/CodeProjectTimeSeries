from timeseries.data.lorenz.lorenz import lorenz_wrapper
from timeseries.models.lorenz.functions.dataprep import feature_multi_step_xy_from_uv, split_uv_seq_one_step
from timeseries.models.lorenz.functions.functions import walk_forward_step_forecast
from timeseries.models.lorenz.functions.preprocessing import reconstruct_x, preprocess, reconstruct
from timeseries.models.lorenz.univariate.multistep.dcnn.func import dcnn_multi_step_uv_fit, dcnn_multi_step_uv_predict, \
    dcnn_multi_step_uv_predict_walk
from timeseries.models.lorenz.univariate.multistep.stroganoff.func import stroganoff_multi_step_uv_fit, \
    stroganoff_multi_step_uv_predict, stroganoff_multi_step_uv_predict_walk
from timeseries.models.lorenz.univariate.onestep.dcnn.func import dcnn_one_step_uv_fit

from timeseries.models.lorenz.univariate.onestep.stroganoff.func import stroganoff_one_step_uv_fit, \
    stroganoff_one_step_uv_predict
from timeseries.models.utils.forecast import multi_step_forecast_df, one_step_forecast_df
from timeseries.models.utils.metrics import forecast_accuracy
from timeseries.plotly.plot import plotly_time_series

if __name__ == '__main__':
    # %% GENERAL INPUTS
    detrend_ops = ['ln_return', ('ema_diff', 5), 'ln_return']
    save_folder = 'images'
    plot_title = True
    save_plots = False
    plot_hist = False
    verbose = 1
    suffix = 'trend_pp_noise_1'

    # MODEL AND TIME SERIES INPUTS
    name = "D-CNN"
    input_cfg = {"variate": "uni", "granularity": 5, "noise": True, 'preprocess': True,
                 'trend': True, 'detrend': 'ln_return'}
    model_cfg = {"n_steps_in": 50, "n_steps_out": 6, 'n_layers': 5, "n_filters": 50,
                 "n_kernel": 3, "n_epochs": 20, "n_batch": 32, 'reg': 'l2'}

    lorenz_df, train, test, t_train, t_test = lorenz_wrapper(input_cfg)
    train_pp, test_pp, ss = preprocess(input_cfg, train, test)


    # %% FORECAST
    # model = dcnn_multi_step_uv_fit(train_pp, model_cfg, verbose=verbose)
    # forecast = dcnn_multi_step_uv_predict(model, train_pp, model_cfg)
    # forecast_reconst = reconstruct(forecast, train, test, input_cfg, model_cfg, ss=ss)
    # df = multi_step_forecast_df(train, test[:model_cfg['n_steps_out']], forecast_reconst, t_train,
    #                           t_test[:model_cfg['n_steps_out']], train_prev_steps=200)
    # plotly_time_series(df, title="SERIES: " + str(input_cfg) + '<br>' + name + ': ' + str(model_cfg),
    #                    file_path=[save_folder, name], plot_title=plot_title, save=save_plots)

    # %% PLOT WALK FORWARD FORECAST
    forecast = walk_forward_step_forecast(train_pp, test_pp, model_cfg, dcnn_multi_step_uv_predict,
                                          dcnn_multi_step_uv_fit, steps=model_cfg["n_steps_out"],
                                          plot_hist=plot_hist, verbose=verbose)
    # forecast = walk_forward_step_forecast(train_pp, test_pp, model_cfg, dcnn_multi_step_uv_predict_walk,
    #                                       dcnn_one_step_uv_fit, steps=model_cfg["n_steps_out"],
    #                                       plot_hist=plot_hist, verbose=verbose)

    # %%
    forecast_reconst = reconstruct(forecast, train, test, input_cfg, model_cfg, ss=ss)
    metrics = forecast_accuracy(forecast_reconst, test)

    # %%
    df = multi_step_forecast_df(train, test, forecast_reconst, train_prev_steps=200)
    plotly_time_series(df,
                       title="SERIES: " + str(input_cfg) + '<br>' + name + ': ' + str(model_cfg) + '<br>RES: ' + str(
                           metrics),
                       markers='lines',
                       file_path=[save_folder, name + "_" + suffix], plot_title=plot_title, save=save_plots)
    print(metrics)
