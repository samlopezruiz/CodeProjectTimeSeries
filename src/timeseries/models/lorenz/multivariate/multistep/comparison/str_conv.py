from timeseries.data.lorenz.lorenz import lorenz_wrapper
from timeseries.models.lorenz.functions.harness import repeat_evaluate, summarize_scores
from timeseries.models.lorenz.functions.preprocessing import preprocess
from timeseries.models.lorenz.multivariate.multistep.convlstm.func import convlstm_multi_step_mv_predict, \
    convlstm_multi_step_mv_fit
from timeseries.models.lorenz.multivariate.multistep.stroganoff.func import stroganoff_multi_step_mv_fit, \
    stroganoff_multi_step_mv_predict
from timeseries.models.utils.models import models_strings
from timeseries.plotly.plot import plot_multiple_scores
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
    n_steps_out = 6
    n_repeats = 5
    suffix = 'trend_pp_noise'

    # MODEL AND TIME SERIES INPUTS
    input_cfg = {"variate": "multi", "granularity": 5, "noise": True, 'preprocess': True,
                 'trend': True, 'detrend': 'ln_return'}

    lorenz_df, train, test, t_train, t_test = lorenz_wrapper(input_cfg)
    train_pp, test_pp, ss = preprocess(input_cfg, train, test)

    # %%
    names = ["STROGANOFF", "CONV-LSTM"]
    model_cfgs = []
    model_cfgs.append(
        {"n_steps_in": 5, "n_steps_out": n_steps_out, "n_gen": 10, "n_pop": 300, "cxpb": 0.8, "mxpb": 0.05,
         "depth": 7, 'elitism_size': 2, 'selection': 'tournament', 'tour_size': 5})

    model_cfgs.append({"n_seq": 3, "n_steps_in": 12, "n_steps_out": n_steps_out, "n_filters": 256,
                       "n_kernel": 3, "n_nodes": 200, "n_epochs": 15, "n_batch": 100})

    functions = []
    functions.append([stroganoff_multi_step_mv_predict, stroganoff_multi_step_mv_fit])
    functions.append([convlstm_multi_step_mv_predict, convlstm_multi_step_mv_fit])

    # %% RUN FORECASTS
    scores_models = []
    for i, model_cfg in enumerate(model_cfgs):
        print('EVALUATING {} MODEL'.format(names[i]))
        metrics, _ = repeat_evaluate(train_pp, test_pp, train, test, input_cfg, model_cfg,
                                     functions[i][0], functions[i][1], ss=ss, n_repeats=n_repeats)
        scores, scores_m, score_std = summarize_scores(names[i], metrics, score_type, input_cfg, model_cfg, plot=False)
        scores_models.append((scores, scores_m, score_std))

    # %%
    models_info, models_name = models_strings(names, model_cfgs)

    scores = [s[0] for s in scores_models]
    plot_multiple_scores(scores, score_type, names, title="SERIES: " + str(input_cfg) + models_info,
                         file_path=[save_folder, models_name + suffix], plot_title=plot_title, save=True)
