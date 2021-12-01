from algorithms.gpregress.prim import primitives1
from timeseries.data.lorenz.lorenz import lorenz_wrapper
from timeseries.experiments.lorenz.functions.functions import walk_forward_step_forecast
from timeseries.experiments.lorenz.functions.preprocessing import preprocess, reconstruct
from timeseries.experiments.lorenz.univariate.onestep.gpregress.func import gpregress_one_step_uv_fit, \
    gpregress_one_step_uv_predict
from timeseries.experiments.utils.forecast import multi_step_forecast_df, one_step_forecast_df
from timeseries.experiments.utils.metrics import forecast_accuracy
from timeseries.plotly.plot import plotly_time_series


if __name__ == '__main__':
    # %% GENERAL INPUTS
    detrend_ops = ['ln_return', ('ema_diff', 5), 'ln_return']
    save_folder = 'images'
    plot_title = True
    save_plots = False
    plot_hist = True
    verbose = 2
    suffix = 'trend_pp_noise_1'

    # MODEL AND TIME SERIES INPUTS
    name = "GP_REGRESS"
    input_cfg = {"variate": "uni", "granularity": 5, "noise": True, 'preprocess': True,
                 'trend': True, 'detrend': 'ln_return'}
    model_cfg = {"n_steps_in": 10, "n_gen": 10, "n_pop": 500, "cxpb": 0.6, "mxpb": 0.1,
                 "depth": 8, 'elitism_size': 2, 'selection': 'tournament', 'tour_size': 5,
                 'primitives': primitives1()}

    lorenz_df, train, test, t_train, t_test = lorenz_wrapper(input_cfg)
    train_pp, test_pp, ss = preprocess(input_cfg, train, test)

    # %% FORECAST
    # model = gpregress_one_step_uv_fit(train_pp, model_cfg, verbose=verbose)
    # # history doesn't contain last y column
    # history = train
    # forecast = gpregress_one_step_uv_predict(model, train_pp, model_cfg)
    # forecast_reconst = reconstruct(forecast, train, test, input_cfg, model_cfg, ss=ss)
    # df = one_step_forecast_df(train, test[:1], forecast_reconst[0], t_train, t_test[:1], train_prev_steps=200)
    # plotly_time_series(df, title="SERIES: " + str(input_cfg) + '<br>' + name + ': ' + str(model_cfg),
    #                    file_path=[save_folder, name], plot_title=plot_title, save=save_plots)

    # %% PLOT WALK FORWARD FORECAST
    forecast = walk_forward_step_forecast(train_pp, test_pp, model_cfg, gpregress_one_step_uv_predict,
                                          gpregress_one_step_uv_fit, steps=1,
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
