import numpy as np

from timeseries.experiments.lorenz.functions.dataprep import split_mv_seq_multi_step
from timeseries.data.lorenz.lorenz import lorenz_wrapper

if __name__ == '__main__':
    # %% GENERAL INPUTS
    in_cfg = {'steps': 6, 'save_results': False, 'verbose': 1, 'plot_title': True, 'plot_hist': False,
              'image_folder': 'images', 'results_folder': 'results',
              'detrend_ops': ['ln_return', ('ema_diff', 5), 'ln_return']}

    # MODEL AND TIME SERIES INPUTS
    model_name = "CNN"
    input_cfg = {"variate": "multi", "granularity": 5, "noise": True, 'preprocess': True,
                 'trend': True, 'detrend': 'ln_return'}
    model_cfg = {"n_seq": 3, "n_steps_in": 12, "n_steps_out": 6, "n_filters": 64,
                 "n_kernel": 3, "n_nodes": 200, "n_epochs": 100, "n_batch": 100}

    # %% DATA
    # lorenz_df, train, test, t_train, t_test = lorenz_wrapper(input_cfg)

    train = np.array([list(range(100)), list(range(-100, 0)), list(range(100, 0, -1))]).T
    s = np.sum(train, axis=1).reshape(-1, 1)
    train = np.append(train, s, axis=1)
    X1, y = split_mv_seq_multi_step(train, model_cfg['n_steps_in'], 6)
    n_features = X1.shape[2]
    # reshape from [samples, timesteps] into [samples, subsequences, timesteps, features]
    X = X1.reshape((X1.shape[0], model_cfg['n_seq'], int(model_cfg['n_steps_in']/model_cfg['n_seq']), n_features))