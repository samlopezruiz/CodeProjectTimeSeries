import os

import numpy as np

from timeseries.experiments.lorenz.multivariate.multistep.cnn.func import cnn_get_multi_step_mv_funcs, get_cnn_steps_cfgs
from timeseries.experiments.lorenz.multivariate.multistep.cnnlstm.func import cnnlstm_get_multi_step_mv_funcs, \
    get_cnnlstm_steps_cfgs
from timeseries.experiments.lorenz.multivariate.multistep.stroganoff.func import stroganoff_get_multi_step_mv_funcs

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
    model_name = "STROGANOFF"
    input_cfg = {"variate": "multi", "granularity": 5, "noise": True, 'preprocess': True,
                 'trend': True, 'detrend': 'ln_return'}
    model_cfg = {"n_steps_out": 1, "n_steps_in": 14, "depth": 4, "n_gen": 5, "n_pop": 200,
                 "cxpb": 0.6, "mxpb": 0.07, 'elitism_size': 2, 'selection': 'tournament', 'tour_size': 5}
    func_cfg = stroganoff_get_multi_step_mv_funcs()

    searches = {}
    searches['A'] = (True, {'n_steps_in': list(range(2, 18, 4)), 'depth': list(range(2, 9, 2))})
    searches['B'] = (True, {'n_gen': list(range(5, 55, 10)), 'n_pop': list(range(100, 500, 100))})
    searches['C'] = (True, {'cxpb': list(np.round(np.arange(0.2, 1.2, 0.2), 2))})
    searches['D'] = (True, {'mxpb': list(np.round(np.arange(0.01, 0.1, 0.02), 4))})
    searches['E'] = (True, {'selection': ['tournament', 'roullete']})

    is_comb, gs_cfg = searches['E']
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
