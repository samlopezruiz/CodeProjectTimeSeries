from timeseries.experiments.lorenz.multivariate.multistep.cnnconvlstm.func import cnnconvlstm_get_multi_step_mv_funcs
from timeseries.experiments.lorenz.multivariate.multistep.cnnlstm.func import cnnlstm_get_multi_step_mv_funcs


def cnnconvlstm_mv_configs(steps=1):
    keys = {6: 0, 3: 1, 1: 2}
    good_configs = []

    # %% CNN-LSTM: 0.9467 minmax  (+/- 0.0023) STEPS=6
    model_name = "CNN-CONVLSTM"
    input_cfg = {"variate": "multi", "granularity": 5, "noise": True, 'preprocess': True,
                 'trend': True, 'detrend': 'ln_return'}
    model_cfg = {"n_steps_out": 6, "n_steps_in": 8, "n_seq": 2, "n_kernel": 3,
                 "n_filters": 64, "n_nodes": 128, "n_batch": 32, "n_epochs": 25}
    func_cfg = cnnconvlstm_get_multi_step_mv_funcs()
    good_configs.append((model_name, input_cfg, model_cfg, func_cfg))

    # %% CNN-LSTM: 0.9491 minmax  (+/- 0.0025) STEPS=3
    model_name = "CNN-CONVLSTM"
    input_cfg = {"variate": "multi", "granularity": 5, "noise": True, 'preprocess': True,
                 'trend': True, 'detrend': 'ln_return'}
    model_cfg = {"n_steps_out": 3, "n_steps_in": 8, "n_seq": 2, "n_kernel": 4,
                 "n_filters": 64, "n_nodes": 64, "n_batch": 32, "n_epochs": 25}
    func_cfg = cnnconvlstm_get_multi_step_mv_funcs()
    good_configs.append((model_name, input_cfg, model_cfg, func_cfg))

    # %% CNN-LSTM: 0.9622 minmax  (+/- 0.0009) STEPS=1
    model_name = "CNN-CONVLSTM"
    input_cfg = {"variate": "multi", "granularity": 5, "noise": True, 'preprocess': True,
                 'trend': True, 'detrend': 'ln_return'}
    model_cfg = {"n_steps_out": 1, "n_steps_in": 8, "n_seq": 4, "n_kernel": 2,
                 "n_filters": 32, "n_nodes": 16, "n_batch": 32, "n_epochs": 25}
    func_cfg = cnnconvlstm_get_multi_step_mv_funcs()
    good_configs.append((model_name, input_cfg, model_cfg, func_cfg))

    return good_configs[keys[steps]]