from timeseries.data.lorenz.lorenz import lorenz_wrapper
from timeseries.models.lorenz.functions.functions import walk_forward_step_forecast_werrs
from timeseries.models.lorenz.functions.harness import repeat_evaluate
from timeseries.models.lorenz.functions.summarize import summarize_scores
from timeseries.models.lorenz.functions.preprocessing import preprocess
from timeseries.models.lorenz.univariate.onestep.convlstm.func import convlstm_one_step_uv_predict, \
    convlstm_one_step_uv_fit
from timeseries.models.lorenz.univariate.onestep.gp.func import gp_one_step_uv_predict, gp_one_step_uv_fit
from timeseries.models.lorenz.univariate.onestep.stroganoff.func import stroganoff_one_step_uv_predict, \
    stroganoff_one_step_uv_fit
from timeseries.models.utils.models import models_strings, save_vars
from timeseries.plotly.plot import plot_multiple_scores
import joblib


if __name__ == '__main__':
    # %% GENERAL INPUTS
    model_cfgs, functions = [], []
    detrend_ops = ['ln_return', ('ema_diff', 5), 'ln_return']
    images_folder, results_folder = 'images', 'results'
    score_type = 'minmax'
    plot_title = True
    save_plots = True
    plot_hist = False
    verbose = 0
    n_repeats = 5
    suffix = 'trend_pp'

    # MODEL AND TIME SERIES INPUTS
    input_cfg = {"variate": "uni", "granularity": 5, "noise": False, 'preprocess': True,
                 'trend': True, 'detrend': 'ln_return'}

    lorenz_df, train, test, t_train, t_test = lorenz_wrapper(input_cfg)
    train_pp, test_pp, ss = preprocess(input_cfg, train, test)

    # %%
    names = ['GP', "STROGANOFF", "CONV-LSTM"]

    model_cfgs.append({"n_steps_in": 10, "ngen": 30, 'cxpb': 0.8, 'mutpb': 0.2})
    model_cfgs.append(
        {"n_steps_in": 10, "n_gen": 20, "n_pop": 300, "cxpb": 0.8, "mxpb": 0.05,
         "depth": 7, 'elitism_size': 2, 'selection': 'tournament', 'tour_size': 5})

    model_cfgs.append({"n_seq": 3, "n_steps_in": 12, "n_filters": 256,
                       "n_kernel": 3, "n_nodes": 200, "n_epochs": 15, "n_batch": 100})

    functions.append([gp_one_step_uv_predict, gp_one_step_uv_fit])
    functions.append([stroganoff_one_step_uv_predict, stroganoff_one_step_uv_fit])
    functions.append([convlstm_one_step_uv_predict, convlstm_one_step_uv_fit])

    # %% RUN FORECASTS
    scores_models = []
    for i, model_cfg in enumerate(model_cfgs):
        print('EVALUATING {} MODEL'.format(names[i]))
        if names[i] == 'GP':
            metrics, _ = repeat_evaluate(train_pp, test_pp, train, test, input_cfg, model_cfg,
                                         functions[i][0], functions[i][1], ss=ss, n_repeats=n_repeats,
                                         walkforward=walk_forward_step_forecast_werrs)
        else:
            metrics, _ = repeat_evaluate(train_pp, test_pp, train, test, input_cfg, model_cfg,
                                         functions[i][0], functions[i][1], ss=ss, n_repeats=n_repeats)
        scores, scores_m, score_std = summarize_scores(names[i], metrics, score_type, input_cfg, model_cfg, plot=False)
        scores_models.append((scores, scores_m, score_std))

    save_vars([model_cfgs, scores_models], file_path=[results_folder, '_'.join(names)])

    # %%
    models_info, models_name = models_strings(names, model_cfgs)

    scores = [s[0] for s in scores_models]
    plot_multiple_scores(scores, score_type, names, title="SERIES: " + str(input_cfg) + models_info,
                         file_path=[images_folder, models_name + suffix], plot_title=plot_title, save=True)
