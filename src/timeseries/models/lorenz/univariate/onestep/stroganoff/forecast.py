from timeseries.data.lorenz.lorenz import lorenz_wrapper
from timeseries.models.lorenz.functions.functions import walk_forward_step_forecast
from timeseries.models.lorenz.functions.preprocessing import reconstruct_x, preprocess, reconstruct
from timeseries.models.lorenz.multivariate.multistep.convlstm.func import convlstm_multi_step_mv_fit, \
    convlstm_multi_step_mv_predict
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
    plot_hist = True
    verbose = 1
    suffix = 'trend_pp_noise_1'

    # MODEL AND TIME SERIES INPUTS
    name = "STROGANOFF"
    input_cfg = {"variate": "uni", "granularity": 5, "noise": True, 'preprocess': True,
                 'trend': True, 'detrend': 'ln_return'}
    model_cfg = {"n_steps_in": 20, "n_gen": 20, "n_pop": 300, "cxpb": 0.9, "mxpb": 0.1,
                 "depth": 10, 'elitism_size': 2, 'selection': 'tournament', 'tour_size': 5}


    lorenz_df, train, test, t_train, t_test = lorenz_wrapper(input_cfg)
    train_pp, test_pp, ss = preprocess(input_cfg, train, test)

    # %% FORECAST
    # model = stroganoff_one_step_uv_fit(train_pp, model_cfg, verbose=1)
    # # history doesn't contain last y column
    # history = train
    # forecast = stroganoff_one_step_uv_predict(model, train_pp, model_cfg)
    # forecast_reconst = reconstruct(forecast, train, test, input_cfg, model_cfg, ss=ss)
    # df = one_step_forecast_df(train, test[:1], forecast_reconst[0], t_train, t_test[:1], train_prev_steps=200)
    # plotly_time_series(df, title="SERIES: " + str(input_cfg) + '<br>' + name + ': ' + str(model_cfg),
    #                    file_path=[save_folder, name], plot_title=plot_title, save=save_plots)

    # %% PLOT WALK FORWARD FORECAST
    forecast = walk_forward_step_forecast(train_pp, test_pp, model_cfg, stroganoff_one_step_uv_predict,
                                          stroganoff_one_step_uv_fit, steps=1,
                                          plot_hist=plot_hist, verbose=verbose)

    # %%
    forecast_reconst = reconstruct(forecast, train, test, input_cfg, model_cfg, ss=ss)
    metrics = forecast_accuracy(forecast_reconst, test)

    # %%
    df = multi_step_forecast_df(train, test, forecast_reconst, train_prev_steps=200)
    plotly_time_series(df,
                       title="SERIES: " + str(input_cfg) + '<br>' + name + ': ' + str(model_cfg) + '<br>RES: ' + str(
                           metrics),
                       markers='lines',
                       file_path=[save_folder, name+"_"+suffix], plot_title=plot_title, save=save_plots)
    print(metrics)
