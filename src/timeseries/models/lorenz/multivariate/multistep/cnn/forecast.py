from timeseries.data.lorenz.lorenz import univariate_lorenz, multivariate_lorenz, lorenz_wrapper
from timeseries.models.lorenz.functions.functions import walk_forward_step_forecast
from timeseries.models.lorenz.functions.preprocessing import reconstruct, preprocess
from timeseries.models.lorenz.multivariate.multistep.cnn.func import cnn_multi_step_mv_fit, cnn_multi_step_mv_predict
from timeseries.models.lorenz.multivariate.onestep.cnn.func import cnn_one_step_mv_fit, cnn_one_step_mv_predict
from timeseries.models.lorenz.multivariate.onestep.mlp.func import mlp_one_step_mv_predict
from timeseries.models.lorenz.univariate.onestep.cnn.func import cnn_one_step_uv_fit, cnn_one_step_uv_predict
from timeseries.models.utils.forecast import multi_step_forecast_df, one_step_forecast_df
from timeseries.models.utils.metrics import forecast_accuracy
from timeseries.plotly.plot import plotly_time_series

if __name__ == '__main__':
    # %% INPUTS
    detrend_ops = ['ln_return', ('ema_diff', 5), 'ln_return']
    save_folder = 'images'
    plot_title = True
    save_plots = False
    plot_hist = False
    verbose = 1
    suffix = 'trend_pp_noise_1'

    # MODEL AND TIME SERIES INPUTS
    name = "CNN"
    input_cfg = {"variate": "multi", "granularity": 5, "noise": True, 'preprocess': True,
                 'trend': True, 'detrend': 'ln_return'}
    model_cfg = {"n_steps_in": 36, "n_steps_out": 6, "n_filters": 256, "n_kernel": 3,
                 "n_epochs": 20, "n_batch": 100}

    lorenz_df, train, test, t_train, t_test = lorenz_wrapper(input_cfg)
    train_pp, test_pp, ss = preprocess(input_cfg, train, test)
    # %% FORECAST
    # model = cnn_multi_step_mv_fit(train_pp, cfg)
    # # history doesn't contain last y column
    # history = train_pp[:, :-1]
    # pred = cnn_multi_step_mv_predict(model, history, cfg)
    # df = multi_step_forecast_df(train[:, -1], test[:cfg["n_steps_out"], -1], pred,
    #                           t_train, t_test[:cfg["n_steps_out"]], train_prev_steps=500)
    # plotly_time_series(df, title=name+" Forecast")

    # %% PLOT WALK FORWARD FORECAST
    forecast = walk_forward_step_forecast(train_pp, test_pp, model_cfg, cnn_multi_step_mv_predict,
                                          cnn_multi_step_mv_fit, steps=model_cfg['n_steps_out'],
                                          plot_hist=plot_hist, verbose=verbose)

    # %%
    forecast_reconst = reconstruct(forecast, train, test, input_cfg, model_cfg, ss=ss)
    metrics = forecast_accuracy(forecast_reconst, test[:, -1])

    # %%
    df = multi_step_forecast_df(train[:, -1], test[:, -1], forecast_reconst, train_prev_steps=200)
    plotly_time_series(df,
                       title="SERIES: " + str(input_cfg) + '<br>' + name + ': ' + str(model_cfg) + '<br>RES: ' + str(
                           metrics),
                       markers='lines',
                       file_path=[save_folder, name + "_" + suffix], plot_title=plot_title, save=save_plots)
    print(metrics)
