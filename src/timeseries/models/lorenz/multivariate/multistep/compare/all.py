import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from timeseries.models.lorenz.multivariate.multistep.configs.all import all_mv_configs
from timeseries.models.lorenz.functions.harness import save_plot_results, evaluate_models_series

# %%

if __name__ == '__main__':
    # %% GENERAL INPUTS
    in_cfg = {'steps': 6, 'n_series': 3, 'n_repeats': 3, 'save_results': True, 'verbose': 1, 'score_type': 'minmax',
              'plot_title': True, 'plot_hist': False, 'image_folder': 'images', 'results_folder': 'results',
              'detrend_ops': ['ln_return', ('ema_diff', 5), 'ln_return']}
    input_cfg = {"variate": "multi", "granularity": 5, "noise": True, 'preprocess': True,
                 'trend': True, 'detrend': 'ln_return'}

    # %%
    names, model_cfgs, func_cfgs = all_mv_configs(in_cfg['steps'], group='NN', ensemble=True)

    # %% RUN EVALUATIONS
    summary, data, errors = evaluate_models_series(in_cfg, input_cfg, names, model_cfgs, func_cfgs, debug=False)
    save_plot_results(names, summary, data, errors, input_cfg, model_cfgs, in_cfg)
