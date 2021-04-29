from timeseries.models.lorenz.multivariate.multistep.convlstm.func import convlstm_get_multi_step_mv_funcs


def convlstm_mv_configs(steps=1):
    keys = {6: 0, 3: 1, 1: 2}
    good_configs = []

    # %% D-CNN: 0.9405 minmax  (+/- 0.0025)  STEPS=6
    model_name = "ConvLSTM"
    input_cfg = {"variate": "multi", "granularity": 5, "noise": True, 'preprocess': True,
                 'trend': True, 'detrend': 'ln_return'}
    model_cfg = {"n_seq": 3, "n_steps_in": 12, "n_steps_out": 6, "n_filters": 64,
                       "n_kernel": 3, "n_nodes": 50, "n_epochs": 15, "n_batch": 100}
    functions = convlstm_get_multi_step_mv_funcs()
    good_configs.append((model_name, input_cfg, model_cfg, functions))

    # %% D-CNN: 0.9477 minmax  (+/- 0.0015) STEPS=3
    model_name = "ConvLSTM"
    input_cfg = {"variate": "multi", "granularity": 5, "noise": True, 'preprocess': True,
                 'trend': True, 'detrend': 'ln_return'}
    model_cfg = {"n_seq": 3, "n_steps_in": 12, "n_steps_out": 3, "n_filters": 64,
                       "n_kernel": 3, "n_nodes": 50, "n_epochs": 15, "n_batch": 100}
    functions = convlstm_get_multi_step_mv_funcs()
    good_configs.append((model_name, input_cfg, model_cfg, functions))

    # %% WAVENET: 0.9568 minmax  (+/- 0.0012) STEPS=1
    model_name = "ConvLSTM"
    input_cfg = {"variate": "multi", "granularity": 5, "noise": True, 'preprocess': True,
                 'trend': True, 'detrend': 'ln_return'}
    model_cfg = {"n_seq": 3, "n_steps_in": 12, "n_steps_out": 1, "n_filters": 64,
                       "n_kernel": 3, "n_nodes": 50, "n_epochs": 15, "n_batch": 100}
    functions = convlstm_get_multi_step_mv_funcs()
    good_configs.append((model_name, input_cfg, model_cfg, functions))

    return good_configs[keys[steps]]