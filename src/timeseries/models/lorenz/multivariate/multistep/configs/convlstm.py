from timeseries.models.lorenz.multivariate.multistep.convlstm.func import convlstm_get_multi_step_mv_funcs


def convlstm_mv_configs(steps=1):
    keys = {6: 0, 3: 1, 1: 2}
    good_configs = []

    # %% CONV-LSTM: 0.9229 minmax  (+/- 0.0031)  STEPS=6
    model_name = "ConvLSTM"
    input_cfg = {"variate": "multi", "granularity": 5, "noise": True, 'preprocess': True,
                 'trend': True, 'detrend': 'ln_return'}
    model_cfg = {"n_steps_out": 6, "n_steps_in": 8, "n_seq": 2, "n_kernel": 3,
                 "n_filters": 64, "n_nodes": 64, "n_batch": 32, "n_epochs": 35}
    functions = convlstm_get_multi_step_mv_funcs()
    good_configs.append((model_name, input_cfg, model_cfg, functions))

    # %% CONV-LSTM: 0.9547 minmax  (+/- 0.0017) STEPS=3
    model_name = "ConvLSTM"
    input_cfg = {"variate": "multi", "granularity": 5, "noise": True, 'preprocess': True,
                 'trend': True, 'detrend': 'ln_return'}
    model_cfg = {"n_steps_out": 3, "n_steps_in": 6, "n_seq": 2, "n_kernel": 2,
                 "n_filters": 32, "n_nodes": 64, "n_batch": 64, "n_epochs": 30}
    functions = convlstm_get_multi_step_mv_funcs()
    good_configs.append((model_name, input_cfg, model_cfg, functions))

    # %% CONV-LSTM: 0.9574 minmax  (+/- 0.001) STEPS=1
    model_name = "ConvLSTM"
    input_cfg = {"variate": "multi", "granularity": 5, "noise": True, 'preprocess': True,
                 'trend': True, 'detrend': 'ln_return'}
    model_cfg = {"n_steps_out": 1, "n_steps_in": 6, "n_seq": 2, "n_kernel": 2,
                 "n_filters": 64, "n_nodes": 64, "n_batch": 64, "n_epochs": 15}
    functions = convlstm_get_multi_step_mv_funcs()
    good_configs.append((model_name, input_cfg, model_cfg, functions))

    return good_configs[keys[steps]]