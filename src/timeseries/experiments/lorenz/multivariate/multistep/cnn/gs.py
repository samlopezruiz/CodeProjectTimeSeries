import os

from timeseries.experiments.lorenz.multivariate.multistep.cnn.func import cnn_get_multi_step_mv_funcs, get_cnn_steps_cfgs
from timeseries.experiments.lorenz.multivariate.multistep.cnnlstm.func import cnnlstm_get_multi_step_mv_funcs, \
    get_cnnlstm_steps_cfgs

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from algorithms.wavenet.func import get_wnet_steps_cfgs
from timeseries.experiments.lorenz.multivariate.multistep.dcnn.func import dcnn_get_multi_step_mv_funcs
from timeseries.experiments.utils.models import save_gs_results
from timeseries.data.lorenz.lorenz import lorenz_wrapper
from timeseries.experiments.lorenz.functions.harness import grid_search
from timeseries.experiments.lorenz.functions.preprocessing import preprocess
from timeseries.plotly.plot import plot_gs_results
import time

# %%

if __name__ == '__main__':
    # %% GENERAL INPUTS
    in_cfg = {'steps': 1, 'save_results': True, 'verbose': 1, 'plot_title': True, 'plot_hist': False,
              'image_folder': 'images', 'results_folder': 'results',
              'detrend_ops': ['ln_return', ('ema_diff', 5), 'ln_return']}
    score_type = 'minmax'

    # MODEL AND TIME SERIES INPUTS
    model_name = "CNN"
    input_cfg = {"variate": "multi", "granularity": 5, "noise": True, 'preprocess': True,
                 'trend': True, 'detrend': 'ln_return'}
    model_cfg = {"n_steps_out": 1, "n_steps_in": 14, "n_kernel": 4, "n_filters": 128,
                 'n_nodes': 32, "n_batch": 128, "n_epochs": 20}
    func_cfg = cnn_get_multi_step_mv_funcs()

    searches = {}
    searches['A'] = get_cnn_steps_cfgs(in_steps_range=[6, 24, 4], k_range=[3, 6])
    searches['B'] = {'n_filters': [2**k for k in range(4, 9)], 'n_nodes': [2**k for k in range(4, 9)]}
    searches['C'] = {'n_batch': [2**k for k in range(3, 10)]}
    searches['D'] = {'n_epochs': list(range(5, 55, 5))}

    is_comb = True
    gs_cfg = searches['D']
    n_repeats = 3

    # %% GRID SEARCH
    lorenz_df, train, test, t_train, t_test = lorenz_wrapper(input_cfg)
    train_pp, test_pp, ss = preprocess(input_cfg, train, test)
    data_in = (train_pp, test_pp, train, test)

    st = time.time()
    summary, data, errors = grid_search(input_cfg, gs_cfg, model_cfg, func_cfg, n_repeats,
                                        score_type, ss, data_in, comb=is_comb, debug=False)

    print('Grid Search Time: {}'.format(round(time.time() - st, 2)))

    # %% SAVE AND PLOT RESULTS
    vars = [input_cfg, gs_cfg, model_cfg, summary]
    save_gs_results(input_cfg, gs_cfg, model_cfg, in_cfg, vars)
    plot_gs_results(input_cfg, gs_cfg, model_cfg, in_cfg, data, errors)
