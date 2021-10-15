import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from timeseries.experiments.lorenz.multivariate.multistep.configs.convlstm import convlstm_mv_configs
from timeseries.experiments.lorenz.multivariate.multistep.configs.dcnn import dcnn_mv_configs
from timeseries.experiments.lorenz.multivariate.multistep.ensemble.func import ensemble_get_multi_step_mv_funcs, \
    get_ensemble_steps_cfgs
from timeseries.experiments.utils.models import save_gs_results
from timeseries.data.lorenz.lorenz import lorenz_wrapper
from timeseries.experiments.lorenz.functions.harness import grid_search
from timeseries.experiments.lorenz.functions.preprocessing import preprocess
from timeseries.plotly.plot import plot_bar_summary
import time

# %%

if __name__ == '__main__':
    # %% GENERAL INPUTS
    in_cfg = {'steps': 1, 'save_results': False, 'verbose': 1, 'plot_title': True, 'plot_hist': False,
              'image_folder': 'images', 'results_folder': 'results',
              'detrend_ops': ['ln_return', ('ema_diff', 5), 'ln_return']}
    score_type = 'minmax'

    # MODEL AND TIME SERIES INPUTS
    model_name = "ENSEMBLE"
    input_cfg = {"variate": "multi", "granularity": 5, "noise": True, 'preprocess': True,
                 'trend': True, 'detrend': 'ln_return'}
    model_cfg = [convlstm_mv_configs(steps=in_cfg['steps']),
                 dcnn_mv_configs(steps=in_cfg['steps'])]
                 # wavenet_mv_configs(steps=in_cfg['steps']),
                 # cnnlstm_mv_configs(steps=in_cfg['steps']),
                 # cnn_mv_configs(steps=in_cfg['steps'])]
    func_cfg = ensemble_get_multi_step_mv_funcs()

    gs_cfg = get_ensemble_steps_cfgs(model_cfg)
    n_repeats = 1

    # %% GRID SEARCH
    lorenz_df, train, test, t_train, t_test = lorenz_wrapper(input_cfg)
    train_pp, test_pp, ss = preprocess(input_cfg, train, test)
    data_in = (train_pp, test_pp, train, test)

    st = time.time()
    summary, data, errors = grid_search(input_cfg, gs_cfg, model_cfg, func_cfg, n_repeats, score_type,
                                        ss, data_in, comb=False, debug=False, ensemble=True)

    print('Grid Search Time: {}'.format(round(time.time() - st, 2)))

    # %% SAVE AND PLOT RESULTS
    vars = [input_cfg, gs_cfg, model_cfg, summary]
    save_gs_results(input_cfg, gs_cfg, model_cfg, in_cfg, vars)
    plot_bar_summary(data, errors, title="SERIES: " + str(input_cfg) + '<br>' + 'STEPS OUT: ' + str(in_cfg['steps']),
                     file_path=[in_cfg['image_folder'], 'file_name'], plot_title=in_cfg['plot_title'],
                     save=in_cfg['save_results'], n_cols_adj_range=1)
