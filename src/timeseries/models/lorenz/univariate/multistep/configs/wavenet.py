from algorithms.wavenet.func import wavenet_build
from timeseries.models.lorenz.multivariate.multistep.wavenet.func import wavenet_multi_step_mv_predict, \
    wavenet_multi_step_mv_fit, wavenet_get_multi_step_mv_funcs
from timeseries.models.lorenz.univariate.multistep.wavenet.func import wavenet_get_multi_step_uv_funcs


def wavenet_uv_configs(steps=1):
    keys = {6: 0, 3: 1, 1: 2}
    good_configs = []

    # %% WAVENET: 0.9369 minmax  (+/- 0.0037) STEPS=6
    model_name = "WAVENET"
    input_cfg = {"variate": "uni", "granularity": 5, "noise": True, 'preprocess': True,
                 'trend': True, 'detrend': 'ln_return'}
    model_cfg = {"n_steps_in": 11, "n_steps_out": 6, 'n_layers': 2, "n_filters": 64,
                 "n_kernel": 5, "n_epochs": 20, "n_batch": 100, 'hidden_channels': 7}
    functions = wavenet_get_multi_step_uv_funcs()
    good_configs.append((model_name, input_cfg, model_cfg, functions))

    # %% WAVENET: 0.9513 minmax  (+/- 0.0016) STEPS=3
    model_name = "WAVENET"
    input_cfg = {"variate": "uni", "granularity": 5, "noise": True, 'preprocess': True,
                 'trend': True, 'detrend': 'ln_return'}
    model_cfg = {"n_steps_in": 7, "n_steps_out": 3, 'n_layers': 2, "n_filters": 35,
                 "n_kernel": 3, "n_epochs": 35, "n_batch": 100, 'hidden_channels': 7}
    functions = wavenet_get_multi_step_uv_funcs()
    good_configs.append((model_name, input_cfg, model_cfg, functions))

    # %% WAVENET: 0.9568 minmax  (+/- 0.0012) STEPS=1
    model_name = "WAVENET"
    input_cfg = {"variate": "uni", "granularity": 5, "noise": True, 'preprocess': True,
                 'trend': True, 'detrend': 'ln_return'}
    model_cfg = {"n_steps_out": 1, "n_steps_in": 11, 'n_layers': 2, "n_kernel": 5,
                 "n_filters": 70, 'hidden_channels': 8, "n_batch": 64, "n_epochs": 15, }
    functions = wavenet_get_multi_step_uv_funcs()
    good_configs.append((model_name, input_cfg, model_cfg, functions))


    return good_configs[keys[steps]]