import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from algorithms.wavenet.func import get_wnet_steps_cfgs


from timeseries.experiments.lorenz.univariate.multistep.dcnn.func import dcnn_get_multi_step_uv_funcs


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
    model_name = "D-CNN"
    input_cfg = {"variate": "uni", "granularity": 5, "noise": True, 'preprocess': True,
                 'trend': True, 'detrend': 'ln_return'}
    model_cfg = {"n_steps_out": 1, "n_steps_in": 17, 'n_layers': 3, "n_kernel": 4, 'reg': None,
                 "n_filters": 128, 'hidden_channels': 5, "n_batch": 128, "n_epochs": 25}
    functions = dcnn_get_multi_step_uv_funcs()

    # A) get_wnet_steps_cfgs(l_range=[2, 5], k_range=[2, 7])
    # B) {'reg': ['l2', None]}
    # C) {'n_filters': [2**k for k in range(3, 8)], 'hidden_channels': list(range(1, 10, 2))}
    # D) {'n_batch': [2**k for k in range(3, 10)]}
    # E) {'n_epochs': list(range(5, 55, 5))}

    is_comb = True
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
