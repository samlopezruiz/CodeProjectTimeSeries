import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from timeseries.models.lorenz.multivariate.multistep.configs.all import all_mv_configs
from timeseries.data.lorenz.lorenz import lorenz_wrapper
from timeseries.models.lorenz.functions.harness import evaluate_models, save_plot_results
from timeseries.models.lorenz.functions.preprocessing import preprocess
import time

# %%

if __name__ == '__main__':
    # %% GENERAL INPUTS
    in_cfg = {'steps': 1, 'n_repeats': 3, 'save_results': True, 'verbose': 1, 'score_type': 'minmax',
              'plot_title': True, 'plot_hist': False, 'image_folder': 'images', 'results_folder': 'results',
              'detrend_ops': ['ln_return', ('ema_diff', 5), 'ln_return']}
    input_cfg = {"variate": "multi", "granularity": 5, "noise": True, 'preprocess': True,
                 'trend': True, 'detrend': 'ln_return'}

    lorenz_df, train, test, t_train, t_test = lorenz_wrapper(input_cfg)
    train_pp, test_pp, ss = preprocess(input_cfg, train, test)

    # %%
    names, model_cfgs, func_cfgs = all_mv_configs(in_cfg)

    # %% RUN EVALUATIONS
    data = (train_pp, test_pp, train, test)
    st = time.time()
    summary, data, errors = evaluate_models(input_cfg, names, model_cfgs, func_cfgs,
                                            in_cfg['n_repeats'], ss, in_cfg['score_type'], data)
    print('Evaluation Time: {}'.format(round(time.time() - st, 2)))
    save_plot_results(names, summary, data, errors, input_cfg, model_cfgs, in_cfg)
