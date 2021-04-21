import copy
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from timeseries.models.utils.metrics import summary_results
from algorithms.wavenet.func import wavenet_build, dcnn_build
from timeseries.data.lorenz.lorenz import lorenz_wrapper
from timeseries.models.lorenz.functions.harness import repeat_evaluate, summarize_scores, summarize_times
from timeseries.models.lorenz.functions.preprocessing import preprocess
from timeseries.models.lorenz.multivariate.multistep.cnn.func import cnn_multi_step_mv_predict, cnn_multi_step_mv_fit, \
    cnn_multi_step_mv_build
from timeseries.models.lorenz.multivariate.multistep.cnnlstm.func import cnnlstm_multi_step_mv_predict, \
    cnnlstm_multi_step_mv_fit, cnnlstm_multi_step_mv_build
from timeseries.models.lorenz.multivariate.multistep.convlstm.func import convlstm_multi_step_mv_predict, \
    convlstm_multi_step_mv_fit, convlstm_multi_step_mv_build
from timeseries.models.lorenz.multivariate.multistep.dcnn.func import dcnn_multi_step_mv_predict, dcnn_multi_step_mv_fit
from timeseries.models.lorenz.multivariate.multistep.wavenet.func import wavenet_multi_step_mv_predict, \
    wavenet_multi_step_mv_fit
from timeseries.models.utils.models import models_strings, get_params
from timeseries.plotly.plot import plot_multiple_scores, plot_bar_summary
import pandas as pd

if __name__ == '__main__':
    # %% GENERAL INPUTS
    detrend_ops = ['ln_return', ('ema_diff', 5), 'ln_return']
    save_folder = 'images'
    score_type = 'minmax'
    plot_title = True
    save_plots = False
    plot_hist = False
    verbose = 0
    n_steps_out = 1
    n_repeats = 2
    suffix = 'trend_pp_noise_dense_none'+str(n_steps_out)

    # MODEL AND TIME SERIES INPUTS
    input_cfg = {"variate": "multi", "granularity": 5, "noise": True, 'preprocess': True,
                 'trend': True, 'detrend': 'ln_return'}

    lorenz_df, train, test, t_train, t_test = lorenz_wrapper(input_cfg)
    train_pp, test_pp, ss = preprocess(input_cfg, train, test)

    # %%
    names = ["DCNN", "WAVENET", "CNN", "CNN-LSTM", "ConvLSTM"]
    model_cfgs = []
    model_cfgs.append({"n_steps_in": 50, "n_steps_out": n_steps_out, 'n_layers': 5, "n_filters": 50,
                       "n_kernel": 3, "n_epochs": 20, "n_batch": 32, 'reg': 'l2'})
    model_cfgs.append({"n_steps_in": 50, "n_steps_out": n_steps_out, 'n_layers': 5, "n_filters": 64,
                       "n_kernel": 3, "n_epochs": 20, "n_batch": 32, 'hidden_channels': 4})
    model_cfgs.append({"n_steps_in": 36, "n_steps_out": n_steps_out, "n_filters": 64, "n_kernel": 3,
                       "n_epochs": 20, "n_batch": 100})
    model_cfgs.append({"n_seq": 3, "n_steps_in": 12, "n_steps_out": n_steps_out, "n_filters": 64,
                       "n_kernel": 3, "n_nodes": 50, "n_epochs": 20, "n_batch": 100})
    model_cfgs.append({"n_seq": 3, "n_steps_in": 12, "n_steps_out": n_steps_out, "n_filters": 64,
                       "n_kernel": 3, "n_nodes": 50, "n_epochs": 15, "n_batch": 100})

    functions = []
    functions.append([dcnn_multi_step_mv_predict, dcnn_multi_step_mv_fit, dcnn_build])
    functions.append([wavenet_multi_step_mv_predict, wavenet_multi_step_mv_fit, wavenet_build])
    functions.append([cnn_multi_step_mv_predict, cnn_multi_step_mv_fit, cnn_multi_step_mv_build])
    functions.append([cnnlstm_multi_step_mv_predict, cnnlstm_multi_step_mv_fit, cnnlstm_multi_step_mv_build])
    functions.append([convlstm_multi_step_mv_predict, convlstm_multi_step_mv_fit, convlstm_multi_step_mv_build])

    # %% GET PARAMS
    n_params = get_params(model_cfgs, functions, names, train_pp)

    # %% RUN FORECASTS
    scores_models, model_times = [], []
    for i, model_cfg in enumerate(model_cfgs):
        print('EVALUATING {} MODEL'.format(names[i]))
        metrics, _, times = repeat_evaluate(train_pp, test_pp, train, test, input_cfg, model_cfg,
                                            functions[i][0], functions[i][1], ss=ss, n_repeats=n_repeats)
        scores, scores_m, score_std = summarize_scores(names[i], metrics, score_type, input_cfg, model_cfg, plot=False)
        train_t_m, train_t_std, pred_t_m, pred_t_std = summarize_times(names[i], times)
        model_times.append((names[i], times, train_t_m, train_t_std, pred_t_m, pred_t_std))
        scores_models.append((names[i], scores, scores_m, score_std))

    # %%
    models_info, models_name = models_strings(names, model_cfgs)
    summary, data, errors = summary_results(scores_models, model_times, n_params, score_type)
    print(summary)

    plot_bar_summary(data, errors, title="SERIES: " + str(input_cfg) + '<br>' + 'STEPS OUT: ' + str(n_steps_out),
                     file_path=[save_folder, models_name + suffix], plot_title=plot_title,
                     save=save_plots, n_cols_adj_range=data.shape[1])

    #%%
    # scores = [s[1] for s in scores_models]
    # plot_multiple_scores(scores, score_type, names, title="SERIES: "+str(input_cfg)+'<br>'+'STEPS OUT: '+str(n_steps_out),
    #                      file_path=[save_folder, models_name + suffix], plot_title=plot_title, save=save_plots)



