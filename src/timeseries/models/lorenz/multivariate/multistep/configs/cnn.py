from algorithms.wavenet.func import wavenet_build
from timeseries.models.lorenz.multivariate.multistep.cnn.func import cnn_get_multi_step_mv_funcs
from timeseries.models.lorenz.multivariate.multistep.dcnn.func import dcnn_get_multi_step_mv_funcs
from timeseries.models.lorenz.multivariate.multistep.wavenet.func import wavenet_multi_step_mv_predict, \
    wavenet_multi_step_mv_fit


def cnn_mv_configs(steps=1):
    keys = {6: 0, 3: 1, 1: 2}
    good_configs = []

    # %% CNN: 0.9254 minmax  (+/- 0.0015)
    model_name = "CNN"
    input_cfg = {"variate": "multi", "granularity": 5, "noise": True, 'preprocess': True,
                 'trend': True, 'detrend': 'ln_return'}
    model_cfg = {"n_steps_out": 6, "n_steps_in": 10, "n_kernel": 5, "n_filters": 128,
                 'n_nodes': 256, "n_batch": 128, "n_epochs": 20}
    func_cfg = cnn_get_multi_step_mv_funcs()
    good_configs.append((model_name, input_cfg, model_cfg, func_cfg))

    # %% CNN: 0.9454 minmax  (+/- 0.0009)
    model_name = "CNN"
    input_cfg = {"variate": "multi", "granularity": 5, "noise": True, 'preprocess': True,
                 'trend': True, 'detrend': 'ln_return'}
    model_cfg = {"n_steps_out": 3, "n_steps_in": 10, "n_kernel": 4, "n_filters": 64,
                 'n_nodes': 16, "n_batch": 64, "n_epochs": 20}
    func_cfg = cnn_get_multi_step_mv_funcs()
    good_configs.append((model_name, input_cfg, model_cfg, func_cfg))

    # %% CNN: 0.9602 minmax  (+/- 0.0008)
    model_name = "CNN"
    input_cfg = {"variate": "multi", "granularity": 5, "noise": True, 'preprocess': True,
                 'trend': True, 'detrend': 'ln_return'}
    model_cfg = {"n_steps_out": 1, "n_steps_in": 14, "n_kernel": 4, "n_filters": 128,
                 'n_nodes': 32, "n_batch": 128, "n_epochs": 20}
    func_cfg = cnn_get_multi_step_mv_funcs()
    good_configs.append((model_name, input_cfg, model_cfg, func_cfg))

    return good_configs[keys[steps]]
