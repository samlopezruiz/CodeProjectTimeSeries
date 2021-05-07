import os

from timeseries.models.lorenz.multivariate.multistep.cnnlstm.func import get_cnnlstm_steps_cfgs

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from timeseries.models.lorenz.multivariate.multistep.convlstm.func import convlstm_get_multi_step_mv_funcs
from timeseries.models.utils.models import save_gs_results
from timeseries.data.lorenz.lorenz import lorenz_wrapper
from timeseries.models.lorenz.functions.harness import grid_search
from timeseries.models.lorenz.functions.preprocessing import preprocess
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
    model_name = "CONV-LSTM"
    input_cfg = {"variate": "multi", "granularity": 5, "noise": True, 'preprocess': True,
                 'trend': True, 'detrend': 'ln_return'}
    model_cfg = {"n_steps_out": 1, "n_steps_in": 6, "n_seq": 2, "n_kernel": 2,
                  "n_filters": 64, "n_nodes": 64, "n_batch": 64, "n_epochs": 15}
    functions = convlstm_get_multi_step_mv_funcs()

    # A) get_cnnlstm_steps_cfgs(in_steps_range=[4, 14, 2], n_seq_range=[2, 8, 2], k_range=[2, 6])
    # B) {'n_filters': [2**k for k in range(4, 8)], 'n_nodes': [2**k for k in range(4, 8)]}
    # C) {'n_batch': [2**k for k in range(3, 10)]}
    # D) {'n_epochs': list(range(5, 55, 5))}

    is_comb = False
    gs_cfg = {'n_epochs': list(range(5, 55, 5))}
    n_repeats = 3

    # %% GRID SEARCH
    lorenz_df, train, test, t_train, t_test = lorenz_wrapper(input_cfg)
    train_pp, test_pp, ss = preprocess(input_cfg, train, test)
    data_in = (train_pp, test_pp, train, test)

    st = time.time()
    summary, data, errors = grid_search(input_cfg, gs_cfg, model_cfg, functions,
                                        n_repeats, score_type, ss, data_in, comb=is_comb, debug=False)

    print('Grid Search Time: {}'.format(round(time.time() - st, 2)))

    # %% SAVE AND PLOT RESULTS
    vars = [input_cfg, gs_cfg, model_cfg, summary]
    save_gs_results(input_cfg, gs_cfg, model_cfg, in_cfg, vars)
    plot_gs_results(input_cfg, gs_cfg, model_cfg, in_cfg, data, errors)
