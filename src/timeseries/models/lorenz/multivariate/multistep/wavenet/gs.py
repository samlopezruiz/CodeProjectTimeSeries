import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from algorithms.wavenet.func import wavenet_build
from timeseries.data.lorenz.lorenz import lorenz_wrapper
from timeseries.models.lorenz.functions.functions import walk_forward_step_forecast
from timeseries.models.lorenz.functions.harness import repeat_evaluate, summarize_scores, summarize_times, grid_search
from timeseries.models.lorenz.functions.preprocessing import preprocess, reconstruct
from timeseries.models.lorenz.multivariate.multistep.wavenet.func import wavenet_multi_step_mv_fit, \
    wavenet_multi_step_mv_predict
from timeseries.models.utils.forecast import multi_step_forecast_df
from timeseries.models.utils.metrics import forecast_accuracy, summary_results
from timeseries.plotly.plot import plotly_time_series
import time

#%%

if __name__ == '__main__':
    # %% GENERAL INPUTS
    detrend_ops = ['ln_return', ('ema_diff', 5), 'ln_return']
    image_folder, results_folder = 'images', 'results'
    score_type = 'minmax'
    plot_title = True
    save_results = False
    plot_hist = False
    verbose = 1
    n_repeats = 2
    suffix = 'trend_pp_noise_1'

    # MODEL AND TIME SERIES INPUTS
    model_name = "WAVENET"
    input_cfg = {"variate": "multi", "granularity": 5, "noise": True, 'preprocess': True,
                 'trend': True, 'detrend': 'ln_return'}
    model_cfg = {"n_steps_in": 50, "n_steps_out": 6, 'n_layers': 5, "n_filters": 50,
                 "n_kernel": 3, "n_epochs": 20, "n_batch": 32, 'hidden_channels': 4}

    lorenz_df, train, test, t_train, t_test = lorenz_wrapper(input_cfg)
    train_pp, test_pp, ss = preprocess(input_cfg, train, test)

    functions = [wavenet_multi_step_mv_predict, wavenet_multi_step_mv_fit, wavenet_build]

    # %%
    gs_cfg = {'n_epochs': list(range(1, 5)), 'n_batch': list(range(32, 50, 10)), 'hidden_channels': list(range(2))}
    gs_cfg = {'n_epochs': list(range(1, 5)), 'n_batch': list(range(32, 50, 10))}

    data_in = (train_pp, test_pp, train, test)
    st = time.time()
    summary, data, errors = grid_search(input_cfg, gs_cfg, model_cfg, functions,
                                        n_repeats, score_type,  ss, data_in)

    print('Grid Search Time: {}'.format(round(time.time() - st, 2)))



