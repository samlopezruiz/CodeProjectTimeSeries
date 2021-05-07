import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from timeseries.models.lorenz.multivariate.multistep.configs.cnn import cnn_mv_configs
from timeseries.models.lorenz.multivariate.multistep.configs.cnnlstm import cnnlstm_mv_configs
from timeseries.models.lorenz.multivariate.multistep.configs.wavenet import wavenet_mv_configs
from timeseries.models.lorenz.multivariate.multistep.configs.convlstm import convlstm_mv_configs
from timeseries.models.lorenz.multivariate.multistep.configs.dcnn import dcnn_mv_configs
from timeseries.models.lorenz.multivariate.multistep.ensemble.func import ensemble_get_multi_step_mv_funcs, \
    get_ensemble_steps_cfgs
from timeseries.models.lorenz.functions.harness import evaluate_models_series, \
    save_plot_results, ensemble_get_names
# %%

if __name__ == '__main__':
    # %% GENERAL INPUTS
    in_cfg = {'steps': 1, 'n_series': 3, 'n_repeats': 3, 'save_results': True, 'score_type': 'minmax',
              'verbose': 1,  'plot_title': True, 'plot_hist': False,
              'image_folder': 'images', 'results_folder': 'results'}

    # MODEL AND TIME SERIES INPUTS
    model_name = "ENSEMBLE"
    input_cfg = {"variate": "multi", "granularity": 5, "noise": True, 'preprocess': True,
                 'trend': True, 'detrend': 'ln_return'}
    model_cfg = [convlstm_mv_configs(steps=in_cfg['steps']),
                 dcnn_mv_configs(steps=in_cfg['steps']),
                 wavenet_mv_configs(steps=in_cfg['steps']),
                 cnnlstm_mv_configs(steps=in_cfg['steps']),
                 cnn_mv_configs(steps=in_cfg['steps'])]
    func_cfg = ensemble_get_multi_step_mv_funcs()

    model_cfgs = get_ensemble_steps_cfgs(model_cfg)
    names = ensemble_get_names(model_cfgs)
    func_cfgs = [func_cfg for _ in range(len(model_cfgs))]

    # %% RUN EVALUATIONS
    summary, data, errors = evaluate_models_series(in_cfg, input_cfg, names, model_cfgs, func_cfgs)
    summary_sorted = summary.sort_values(['score_m'], ascending=False)
    save_plot_results(names, summary, data, errors, input_cfg, model_cfgs, in_cfg,
                      models_name='ensemble_s'+str(in_cfg['steps']))
