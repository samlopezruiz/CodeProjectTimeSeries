import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from algorithms.gpregress.prim import primitives1
from timeseries.models.lorenz.multivariate.multistep.gpregress.func import gpregress_multi_step_mv_predict, \
    gpregress_multi_step_mv_fit
from timeseries.models.lorenz.multivariate.multistep.stroganoff.func import stroganoff_multi_step_mv_predict, \
    stroganoff_multi_step_mv_fit
from timeseries.models.utils.metrics import summary_results
from algorithms.wavenet.func import wavenet_build, dcnn_build
from timeseries.data.lorenz.lorenz import lorenz_wrapper
from timeseries.models.lorenz.functions.harness import repeat_evaluate, summarize_scores, summarize_times, \
    evaluate_models
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
from timeseries.models.utils.models import models_strings, get_params, save_vars
from timeseries.plotly.plot import plot_multiple_scores, plot_bar_summary
import pandas as pd
import time
#%%

if __name__ == '__main__':
    # %% GENERAL INPUTS
    detrend_ops = ['ln_return', ('ema_diff', 5), 'ln_return']
    image_folder, results_folder = 'images', 'results'
    score_type = 'minmax'
    plot_title = True
    save_results = False
    plot_hist = False
    verbose = 0
    n_steps_out = 1
    n_repeats = 5
    suffix = 'trend_pp_noise_dense_none' + str(n_steps_out)

    # MODEL AND TIME SERIES INPUTS
    input_cfg = {"variate": "multi", "granularity": 5, "noise": True, 'preprocess': True,
                 'trend': True, 'detrend': 'ln_return'}

    lorenz_df, train, test, t_train, t_test = lorenz_wrapper(input_cfg)
    train_pp, test_pp, ss = preprocess(input_cfg, train, test)

    # %%
    names = ["DCNN", "WAVENET"] #, "CNN", "CNN-LSTM", "ConvLSTM", "STROGANOFF", "GP-REGRESS"]
    model_cfgs = []
    model_cfgs.append({"n_steps_in": 50, "n_steps_out": n_steps_out, 'n_layers': 5, "n_filters": 50,
                       "n_kernel": 3, "n_epochs": 20, "n_batch": 32, 'reg': 'l2'})
    model_cfgs.append({"n_steps_in": 50, "n_steps_out": n_steps_out, 'n_layers': 5, "n_filters": 64,
                       "n_kernel": 3, "n_epochs": 20, "n_batch": 32, 'hidden_channels': 4})
    # model_cfgs.append({"n_steps_in": 36, "n_steps_out": n_steps_out, "n_filters": 64, "n_kernel": 3,
    #                    "n_epochs": 20, "n_batch": 100})
    # model_cfgs.append({"n_seq": 3, "n_steps_in": 12, "n_steps_out": n_steps_out, "n_filters": 64,
    #                    "n_kernel": 3, "n_nodes": 50, "n_epochs": 20, "n_batch": 100})
    # model_cfgs.append({"n_seq": 3, "n_steps_in": 12, "n_steps_out": n_steps_out, "n_filters": 64,
    #                    "n_kernel": 3, "n_nodes": 50, "n_epochs": 15, "n_batch": 100})
    # model_cfgs.append({"n_steps_in": 5, "n_steps_out": n_steps_out, "n_gen": 10, "n_pop": 300, "cxpb": 0.8,
    #                    "mxpb": 0.05, "depth": 7, 'elitism_size': 2, 'selection': 'tournament', 'tour_size': 5})
    # model_cfgs.append({"n_steps_in": 10, "n_steps_out": n_steps_out, "n_gen": 10, "n_pop": 400, "cxpb": 0.7,
    #                    "mxpb": 0.1, "depth": 5, 'elitism_size': 5, 'selection': 'tournament', 'tour_size': 3,
    #                    'primitives': primitives1()})

    functions = []
    functions.append([dcnn_multi_step_mv_predict, dcnn_multi_step_mv_fit, dcnn_build])
    functions.append([wavenet_multi_step_mv_predict, wavenet_multi_step_mv_fit, wavenet_build])
    # functions.append([cnn_multi_step_mv_predict, cnn_multi_step_mv_fit, cnn_multi_step_mv_build])
    # functions.append([cnnlstm_multi_step_mv_predict, cnnlstm_multi_step_mv_fit, cnnlstm_multi_step_mv_build])
    # functions.append([convlstm_multi_step_mv_predict, convlstm_multi_step_mv_fit, convlstm_multi_step_mv_build])
    # functions.append([stroganoff_multi_step_mv_predict, stroganoff_multi_step_mv_fit])
    # functions.append([gpregress_multi_step_mv_predict, gpregress_multi_step_mv_fit])

    # %% RUN EVALUATIONS

    data = (train_pp, test_pp, train, test)
    st = time.time()
    summary, data, errors = evaluate_models(input_cfg, names, model_cfgs,
                                            functions, n_repeats, ss, score_type, data, parallel=use_parallel)
    print('Evaluation Time: {}'.format(round(time.time() - st, 2)))

    # %% SAVE RESULTS
    models_info, models_name = models_strings(names, model_cfgs, suffix)
    print(summary)
    if save_results:
        save_vars([input_cfg, model_cfgs, summary], file_path=[results_folder, models_name])

    # %%
    plot_bar_summary(data, errors, title="SERIES: " + str(input_cfg) + '<br>' + 'STEPS OUT: ' + str(n_steps_out),
                     file_path=[image_folder, models_name], plot_title=plot_title,
                     save=save_results, n_cols_adj_range=1)
