from timeseries.models.lorenz.multivariate.multistep.dcnn.func import dcnn_get_multi_step_mv_funcs
from timeseries.models.lorenz.univariate.multistep.dcnn.func import dcnn_get_multi_step_uv_funcs


def dcnn_uv_configs(steps=1):
    keys = {6: 0, 3: 1, 1: 2}
    good_configs = []

    # %% D-CNN: 0.9405 minmax  (+/- 0.0025)  STEPS=6
    model_name = "D-CNN"
    input_cfg = {"variate": "uni", "granularity": 5, "noise": True, 'preprocess': True,
                 'trend': True, 'detrend': 'ln_return'}
    model_cfg = {"n_steps_out": 6, "n_steps_in": 9, 'n_layers': 2, "n_kernel": 4, 'reg': None,
                 "n_filters": 64, 'hidden_channels': 5, "n_batch": 32, "n_epochs": 25}
    functions = dcnn_get_multi_step_uv_funcs()
    good_configs.append((model_name, input_cfg, model_cfg, functions))

    # %% D-CNN: 0.9477 minmax  (+/- 0.0015) STEPS=3
    model_name = "D-CNN"
    input_cfg = {"variate": "uni", "granularity": 5, "noise": True, 'preprocess': True,
                 'trend': True, 'detrend': 'ln_return'}
    model_cfg = {"n_steps_out": 3, "n_steps_in": 5, 'n_layers': 2, "n_kernel": 2, 'reg': None,
                 "n_filters": 32, 'hidden_channels': 5, "n_batch": 16, "n_epochs": 20}
    functions = dcnn_get_multi_step_uv_funcs()
    good_configs.append((model_name, input_cfg, model_cfg, functions))

    # %% D-CNN: 0.9606 minmax  (+/- 0.0005) STEPS=1
    model_name = "D-CNN"
    input_cfg = {"variate": "uni", "granularity": 5, "noise": True, 'preprocess': True,
                 'trend': True, 'detrend': 'ln_return'}
    model_cfg = {"n_steps_out": 1, "n_steps_in": 17, 'n_layers': 3, "n_kernel": 4, 'reg': None,
                 "n_filters": 32, 'hidden_channels': 5, "n_batch": 128, "n_epochs": 20}
    # model_cfg = {"n_steps_out": 1, "n_steps_in": 5, 'n_layers': 2, "n_kernel": 2, 'reg': None,
    #              "n_filters": 32, 'hidden_channels': 5, "n_batch": 256, "n_epochs": 25}
    functions = dcnn_get_multi_step_uv_funcs()
    good_configs.append((model_name, input_cfg, model_cfg, functions))

    return good_configs[keys[steps]]