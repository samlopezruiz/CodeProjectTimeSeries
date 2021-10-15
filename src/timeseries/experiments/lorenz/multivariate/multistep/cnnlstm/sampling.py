import os

import numpy as np

from timeseries.experiments.lorenz.functions.summarize import summarize_scores
from timeseries.experiments.utils.config import unpack_in_cfg

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import pandas as pd
from timeseries.experiments.utils.models import get_suffix
from timeseries.plotly.plot import plotly_time_series
from timeseries.utils.func import interpolate, append_interpolation
from timeseries.experiments.utils.metrics import forecast_accuracy
from timeseries.experiments.lorenz.functions.harness import eval_multi_step_forecast, run_multi_step_forecast, \
    repeat_evaluate
from timeseries.experiments.lorenz.functions.preprocessing import preprocess, reconstruct
from timeseries.data.lorenz.lorenz import lorenz_wrapper
from timeseries.experiments.lorenz.multivariate.multistep.cnnlstm.func import cnnlstm_get_multi_step_mv_funcs
# %%

if __name__ == '__main__':
    # %% GENERAL INPUTS
    model_name = "CNN-LSTM"
    in_cfg = {'steps': 3, 'save_results': True, 'verbose': 1, 'plot_title': False, 'plot_hist': False,
              'image_folder': 'samp_img', 'results_folder': 'results'}
    name = "CNN-LSTM"
    input_cfg = {"variate": "multi", "granularity": 1, "noise": True, 'preprocess': True,
                  'trend': True, 'detrend': 'ln_return'}
    func_cfg = cnnlstm_get_multi_step_mv_funcs()
    lorenz_df, train, test, t_train, t_test = lorenz_wrapper(input_cfg)

    #%%
    model_cfg = {"n_steps_out": 15, "n_steps_in": 8, "n_seq": 2, "n_kernel": 3,
                 "n_filters": 64, "n_nodes": 128, "n_batch": 32, "n_epochs": 25}
    in_cfg['steps'] = model_cfg['n_steps_out']
    T = 2
    n_repeats = 7
    score_type = 'minmax'
    verbose = 0
    train1, test1, t_train1, t_test1 = train[::T], test[::T], t_train[::T], t_test[::T]
    train_pp, test_pp, ss = preprocess(input_cfg, train1, test1)
    data_in = (train_pp, test_pp, train1, test1, t_train1, t_test1)
    metrics, forecast1 = eval_multi_step_forecast(name, input_cfg, model_cfg, func_cfg,
                                                  in_cfg, data_in, ss, train_prev_steps=0)


    result = repeat_evaluate(train_pp, test_pp, train1, test1, input_cfg, model_cfg, func_cfg[0],
                             func_cfg[1], ss=ss, n_repeats=n_repeats, verbose=verbose)
    metrics, predictions, times, n_params, loss = result
    summarize_scores(model_name, metrics, score_type=score_type)
    times = np.array(times)
    train_t, pred_t = times[:, 0], times[:, 1]
    print('Times: train={}s +-({}), pred={}s +-({})'.format(round(np.mean(train_t), 2), round(np.std(train_t), 4),
                                                            round(np.mean(pred_t), 2), round(np.std(pred_t)), 4))

    # %% SAMPLING PERIOD AND OUTPUT STEPS
    model_cfg = {"n_steps_out": 6, "n_steps_in": 8, "n_seq": 2, "n_kernel": 3,
                 "n_filters": 64, "n_nodes": 128, "n_batch": 32, "n_epochs": 25}
    in_cfg['steps'] = model_cfg['n_steps_out']
    T = 5
    train2, test2, t_train2, t_test2 = train[::T], test[::T], t_train[::T], t_test[::T]
    train_pp, test_pp, ss = preprocess(input_cfg, train2, test2)
    data_in = (train_pp, test_pp, train2, test2, t_train2, t_test2)
    metrics, forecast2 = eval_multi_step_forecast(name, input_cfg, model_cfg, func_cfg,
                                                  in_cfg, data_in, ss, train_prev_steps=0)

    result = repeat_evaluate(train_pp, test_pp, train2, test2, input_cfg, model_cfg, func_cfg[0],
                             func_cfg[1], ss=ss, n_repeats=n_repeats, verbose=verbose)
    metrics, predictions, times, n_params, loss = result
    summarize_scores(model_name, metrics, score_type=score_type)
    times = np.array(times)
    train_t, pred_t = times[:, 0], times[:, 1]
    print('Times: train={}s +-({}), pred={}s +-({})'.format(round(np.mean(train_t), 2), round(np.std(train_t), 4),
                                                            round(np.mean(pred_t), 2), round(np.std(pred_t)), 4))

    #%%
    pred = predictions[0]
    down2 = forecast2['forecast'].dropna()
    upsample = forecast1['forecast'].dropna()
    y_true = forecast1['data'].dropna()
    score = []

    for pred in predictions:
        forecast_reconst = reconstruct(pred, train2, test2, input_cfg, steps_out=6, ss=ss)
        downsample = pd.Series(forecast_reconst, index=down2.index)
        df = pd.concat([upsample, downsample], axis=1)
        df.columns = ['upsample', 'downsample']
        df = df.interpolate(method='index', limit_direction='both')
        upsampled = df.loc[upsample.index, 'downsample']
        metric = forecast_accuracy(upsampled, y_true)
        score.append(metric['minmax'])
    print('MinMax: {}s +-({})'.format(round(np.mean(score), 4), round(np.std(score), 4)))

    #%%
    downsample = forecast2['forecast'].dropna()
    forecasts = forecast1.copy()
    forecasts = append_interpolation(forecasts, downsample)
    forecasts.rename(columns={"forecast": "original"}, inplace=True)

    # %%
    y_true = forecasts['data'].dropna()
    metrics1 = forecast_accuracy(forecasts['original'].dropna(), y_true)
    metrics2 = forecast_accuracy(forecasts['upsampled'].dropna(), y_true)
    metrics3 = forecast_accuracy(forecasts['ensemble'].dropna(), y_true)
    res = {'original-minmax': metrics1['minmax'], 'upsampled-minmax': metrics2['minmax'],
           'ensemple-minmax': metrics3['minmax']}
    print(res)

    #%%
    size = (1980, 1080 // 2)
    image_folder, plot_hist, plot_title, save_results, results_folder, verbose = unpack_in_cfg(in_cfg)
    plotly_time_series(forecasts, alphas=[1, 1, .6, .6, 1],
                       title="SERIES: " + str(input_cfg) + '<br>SERIES:' + str(input_cfg) +
                             '<br>' + name +' RES: ' + str(res), markers='lines', plot_title=plot_title,
                       file_path=[image_folder, name + "_" + 'ensemple_mixed'], save=save_results, size=size,
                       label_scale=1)


