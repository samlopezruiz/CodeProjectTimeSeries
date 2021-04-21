from algorithms.gpregress.prim import primitives1
from timeseries.data.lorenz.lorenz import lorenz_wrapper
from timeseries.models.lorenz.functions.dataprep import split_mv_seq_multi_step, multi_step_xy_from_mv
from timeseries.models.lorenz.functions.functions import walk_forward_step_forecast
from timeseries.models.lorenz.functions.preprocessing import reconstruct_x, preprocess, reconstruct
from timeseries.models.lorenz.multivariate.multistep.gpregress.func import gpregress_multi_step_mv_fit, \
    gpregress_multi_step_mv_predict
from timeseries.models.lorenz.multivariate.multistep.stroganoff.func import stroganoff_multi_step_mv_fit, \
    stroganoff_multi_step_mv_predict
from timeseries.models.lorenz.univariate.multistep.stroganoff.func import stroganoff_multi_step_uv_fit, \
    stroganoff_multi_step_uv_predict, stroganoff_multi_step_uv_predict_walk

from timeseries.models.lorenz.univariate.onestep.stroganoff.func import stroganoff_one_step_uv_fit, \
    stroganoff_one_step_uv_predict
from timeseries.models.utils.forecast import multi_step_forecast_df, one_step_forecast_df
from timeseries.models.utils.metrics import forecast_accuracy
from timeseries.plotly.plot import plotly_time_series
import pandas as pd

if __name__ == '__main__':
    # %% GENERAL INPUTS
    detrend_ops = ['ln_return', ('ema_diff', 5), 'ln_return']
    save_folder = 'images'
    plot_title = True
    save_plots = False
    plot_hist = False
    verbose = 2
    suffix = 'trend_pp_noise_1'

    # MODEL AND TIME SERIES INPUTS
    name = "GP_REGRESS"
    input_cfg = {"variate": "multi", "granularity": 5, "noise": True, 'preprocess': True,
                 'trend': True, 'detrend': 'ln_return'}
    model_cfg = {"n_steps_in": 10, "n_steps_out": 6, "n_gen": 10, "n_pop": 800, "cxpb": 0.7, "mxpb": 0.1,
                 "depth": 5, 'elitism_size': 5, 'selection': 'tournament', 'tour_size': 3,
                 'primitives': primitives1()}

    lorenz_df, train, test, t_train, t_test = lorenz_wrapper(input_cfg)
    train_pp, test_pp, ss = preprocess(input_cfg, train, test)

    # %% FORECAST
    # stroganoff_one_step_uv_fit with stroganoff_multi_step_uv_predict_walk
    # model = gpregress_multi_step_mv_fit(train_pp, model_cfg, verbose=0)
    # for m in model:
    #     print('---')
    #     m.print_tree()
    # # history doesn't contain last y column
    # train_pp_x = train_pp[:, :-1]
    # forecast = gpregress_multi_step_mv_predict(model, train_pp_x, model_cfg)
    # forecast_reconst = reconstruct(forecast, train, test, input_cfg, model_cfg, ss=ss)
    # df = multi_step_forecast_df(train[:, -1], test[:model_cfg['n_steps_out'], -1], forecast_reconst, t_train,
    #                           t_test[:model_cfg['n_steps_out']], train_prev_steps=200)
    # plotly_time_series(df, title="SERIES: " + str(input_cfg) + '<br>' + name + ': ' + str(model_cfg),
    #                    file_path=[save_folder, name], plot_title=plot_title, save=save_plots)

    # %% PLOT WALK FORWARD FORECAST
    forecast = walk_forward_step_forecast(train_pp, test_pp, model_cfg, gpregress_multi_step_mv_predict,
                                          gpregress_multi_step_mv_fit, steps=model_cfg["n_steps_out"],
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
                       file_path=[save_folder, name+"_"+suffix], plot_title=plot_title, save=save_plots)
    print(metrics)
