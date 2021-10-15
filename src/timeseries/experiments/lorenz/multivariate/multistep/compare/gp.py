import copy
import os
from algorithms.gpregress.prim import primitives1
from timeseries.experiments.lorenz.multivariate.multistep.gpregress.func import gpregress_multi_step_mv_predict, \
    gpregress_multi_step_mv_fit
from timeseries.experiments.lorenz.multivariate.multistep.stroganoff.func import stroganoff_multi_step_mv_predict, \
    stroganoff_multi_step_mv_fit
from timeseries.experiments.utils.metrics import summary_results
from timeseries.data.lorenz.lorenz import lorenz_wrapper
from timeseries.experiments.lorenz.functions.harness import repeat_evaluate
from timeseries.experiments.lorenz.functions.summarize import summarize_scores, summarize_times
from timeseries.experiments.lorenz.functions.preprocessing import preprocess
from timeseries.experiments.utils.models import models_strings, get_models_params
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
    verbose = 2
    n_steps_out = 2
    n_repeats = 2
    suffix = 'trend_pp_noise_dense_none' + str(n_steps_out)

    # MODEL AND TIME SERIES INPUTS
    input_cfg = {"variate": "multi", "granularity": 5, "noise": False, 'preprocess': True,
                 'trend': True, 'detrend': 'ln_return'}

    lorenz_df, train, test, t_train, t_test = lorenz_wrapper(input_cfg)
    train_pp, test_pp, ss = preprocess(input_cfg, train, test)

    # %%
    names = ["STROGANOFF", "GP-REGRESS"]
    model_cfgs = []
    model_cfgs.append(
        {"n_steps_in": 5, "n_steps_out": n_steps_out, "n_gen": 10, "n_pop": 300, "cxpb": 0.8, "mxpb": 0.05,
         "depth": 7, 'elitism_size': 2, 'selection': 'tournament', 'tour_size': 5})
    model_cfgs.append({"n_steps_in": 10, "n_steps_out": n_steps_out, "n_gen": 10, "n_pop": 400, "cxpb": 0.7, "mxpb": 0.1,
                       "depth": 5, 'elitism_size': 5, 'selection': 'tournament', 'tour_size': 3,
                       'primitives': primitives1()})

    functions = []
    functions.append([stroganoff_multi_step_mv_predict, stroganoff_multi_step_mv_fit])
    functions.append([gpregress_multi_step_mv_predict, gpregress_multi_step_mv_fit])


    # %% GET PARAMS
    n_params = get_models_params(model_cfgs, functions, names, train_pp)

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
                     file_path=[save_folder, models_name + suffix], plot_title=plot_title, save=save_plots)
    # %%
    # scores = [s[1] for s in scores_models]
    # plot_multiple_scores(scores, score_type, names, title="SERIES: "+str(input_cfg)+'<br>'+'STEPS OUT: '+str(n_steps_out),
    #                      file_path=[save_folder, models_name + suffix], plot_title=plot_title, save=save_plots)
