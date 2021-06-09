import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np

from timeseries.models.lorenz.univariate.multistep.cnnlstm.func import cnnlstm_get_multi_step_uv_funcs

from timeseries.models.lorenz.functions.harness import eval_multi_step_forecast
from timeseries.models.lorenz.functions.preprocessing import preprocess
from timeseries.data.lorenz.lorenz import dc_lorenz_wrapper

# %%

if __name__ == '__main__':
    # %% GENERAL INPUTS
    in_cfg = {'steps': 1, 'save_results': False, 'verbose': 1, 'plot_title': True, 'plot_hist': False,
              'image_folder': 'images', 'results_folder': 'results',
              'detrend_ops': ['ln_return', ('ema_diff', 5), 'ln_return']}

    # MODEL AND TIME SERIES INPUTS
    name = "CNN-LSTM"
    input_cfg = {"variate": "multi", "granularity": 1, 'delta_y': 5, "noise": True, 'preprocess': True,
                 'trend': True, 'detrend': 'ln_return'}
    model_cfg = {"n_steps_out": 1, "n_steps_in": 10, "n_seq": 5, "n_kernel": 3,
                 "n_filters": 32, "n_nodes": 16, "n_batch": 32, "n_epochs": 40}
    func_cfg = cnnlstm_get_multi_step_uv_funcs()

    # %% DATA
    lorenz_dc, train, test, t_train, t_test = dc_lorenz_wrapper(input_cfg)

    #%%
    ix_train = int(lorenz_dc.shape[0] * 0.8)
    train = lorenz_dc.iloc[:ix_train, :]
    test = lorenz_dc.iloc[ix_train:, :]

    #%%
    var = 'dc'
    train_x, test_x = np.array(train[var]), np.array(test[var])
    train_pp, test_pp, ss = preprocess(input_cfg, train_x, test_x)
    data_in = (train_pp, test_pp, train_x, test_x, None, None)
    metrics, forecast = eval_multi_step_forecast(name, input_cfg, model_cfg, func_cfg, in_cfg, data_in, ss)


