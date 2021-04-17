from timeseries.data.lorenz.lorenz import univariate_lorenz, lorenz_wrapper
from timeseries.models.lorenz.functions.functions import walk_forward_step_forecast, walk_forward_step_forecast_werrs
from timeseries.models.lorenz.functions.preprocessing import preprocess, reconstruct
from timeseries.models.lorenz.univariate.onestep.gp.func import gp_one_step_uv_fit, \
    gp_one_step_uv_predict
from timeseries.models.utils.forecast import one_step_forecast_df, multi_step_forecast_df
from timeseries.models.utils.metrics import forecast_accuracy
from timeseries.plotly.plot import plotly_time_series


if __name__ == "__main__":
    # %% GENERAL INPUTS
    detrend_ops = ['ln_return', ('ema_diff', 5), 'ln_return']
    save_folder = 'images'
    plot_title = True
    save_plots = False
    plot_hist = True
    verbose = 0
    suffix = 'no_trend'

    # MODEL AND TIME SERIES INPUTS
    name = "GP"
    input_cfg = {"variate": "uni", "granularity": 5, "noise": False, 'preprocess': False,
                 'trend': False, 'detrend': 'ln_return'}
    model_cfg = {"n_steps_in": 10, "n_steps_out": 1, "ngen": 40, 'cxpb': 0.8, 'mutpb': 0.2}

    lorenz_df, train, test, t_train, t_test = lorenz_wrapper(input_cfg)
    train_pp, test_pp, ss = preprocess(input_cfg, train, test)

    # %% FORECAST
    # model = gp_one_step_uv_fit(train, cfg)
    # pred = gp_one_step_uv_predict(model, train, [0 for _ in range(cfg['in_steps'])], cfg)
    # df = one_step_forecast_df(train, test[:1], pred, t_train, t_test[:1], train_prev_steps=500)
    # plotly_time_series(df, title=name + " Forecast")

    # %% PLOT WALK FORWARD FORECAST
    forecast = walk_forward_step_forecast_werrs(train_pp, test_pp, model_cfg, gp_one_step_uv_predict,
                                                gp_one_step_uv_fit, verbose=2, plot_hist=plot_hist)

    # %%
    forecast_reconst = reconstruct(forecast, train, test, input_cfg, model_cfg, ss=ss)
    metrics = forecast_accuracy(forecast_reconst, test)

    # %%
    df = multi_step_forecast_df(train, test, forecast_reconst, train_prev_steps=200)
    plotly_time_series(df,
                       title="SERIES: " + str(input_cfg) + '<br>' + name + ': ' + str(model_cfg) + '<br>RES: ' + str(
                           metrics),
                       markers='lines',
                       file_path=[save_folder, name+'_'+suffix], plot_title=plot_title, save=save_plots)
    print(metrics)