import os

import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from timeseries.data.lorenz.lorenz import lorenz_wrapper
from timeseries.experiments.lorenz.functions.harness import repeat_evaluate
from timeseries.experiments.lorenz.functions.summarize import summarize_scores
from timeseries.experiments.lorenz.functions.preprocessing import preprocess
from timeseries.experiments.lorenz.multivariate.multistep.cnnlstm.func import cnnlstm_get_multi_step_mv_funcs
#%%

if __name__ == '__main__':
    # %% GENERAL INPUTS
    score_type = 'minmax'
    n_repeats = 7
    verbose = 0

    # MODEL AND TIME SERIES INPUTS
    model_name = "CNN-LSTM"
    input_cfg = {"variate": "multi", "granularity": 2, "noise": True, 'preprocess': True,
                 'trend': True, 'detrend': 'ln_return'}
    model_cfg = {"n_steps_out": 6, "n_steps_in": 8, "n_seq": 2, "n_kernel": 3,
                 "n_filters": 64, "n_nodes": 128, "n_batch": 32, "n_epochs": 25}
    func_cfg = cnnlstm_get_multi_step_mv_funcs()

    lorenz_df, train, test, t_train, t_test = lorenz_wrapper(input_cfg)
    train_pp, test_pp, ss = preprocess(input_cfg, train, test)

    # %% EVALUATE
    result = repeat_evaluate(train_pp, test_pp, train, test, input_cfg, model_cfg, func_cfg[0],
                             func_cfg[1], ss=ss, n_repeats=n_repeats, verbose=verbose)
    metrics, predictions, times, n_params, loss = result
    summarize_scores(model_name, metrics, score_type=score_type)
    times = np.array(times)
    train_t, pred_t = times[:, 0], times[:, 1]
    print('Times: train={}s +-({}), pred={}s +-({})'.format(round(np.mean(train_t),2), round(np.std(train_t),4),
                                                          round(np.mean(pred_t),2), round(np.std(pred_t)),4))
    # %% EVALUATE
    result = repeat_evaluate(train_pp, test_pp, train, test, input_cfg, model_cfg, func_cfg[3],
                             func_cfg[4], ss=ss, n_repeats=n_repeats, verbose=verbose)
    metrics, predictions, times, n_params, loss = result
    summarize_scores(model_name, metrics, score_type=score_type)
    times = np.array(times)
    train_t, pred_t = times[:, 0], times[:, 1]
    print('Times: train={}s +-({}), pred={}s +-({})'.format(round(np.mean(train_t), 2), round(np.std(train_t), 4),
                                                            round(np.mean(pred_t), 2), round(np.std(pred_t)), 4))

    # # %% EVALUATE
    # result = repeat_evaluate(train_pp, test_pp, train, test, input_cfg, model_cfg, func_cfg[3],
    #                          func_cfg[1], ss=ss, n_repeats=n_repeats, verbose=verbose)
    # metrics, predictions, times, n_params, loss = result
    # summarize_scores(model_name, metrics, score_type=score_type)
    # times = np.array(times)
    # train_t, pred_t = times[:, 0], times[:, 1]
    # print('Times: train={}s +-({}), pred={}s +-({})'.format(round(np.mean(train_t), 2), round(np.std(train_t), 4),
    #                                                         round(np.mean(pred_t), 2), round(np.std(pred_t)), 4))