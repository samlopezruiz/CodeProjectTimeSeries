from timeseries.experiments.lorenz.univariate.multistep.cnn.func import cnn_get_multi_step_uv_funcs


def cnn_uv_configs(steps=1):
    keys = {6: 0, 3: 1, 1: 2}
    good_configs = []

    model_name = "CNN"
    input_cfg = {"variate": "uni", "granularity": 5, "noise": True, 'preprocess': True,
                 'trend': True, 'detrend': 'ln_return'}
    func_cfg = cnn_get_multi_step_uv_funcs()
    
    # %%
    model_cfg = {"n_steps_out": 6, "n_steps_in": 10, "n_kernel": 5, "n_filters": 128,
                 'n_nodes': 256, "n_batch": 128, "n_epochs": 20}
    good_configs.append((model_name, input_cfg, model_cfg, func_cfg))

    # %%
    model_cfg = {"n_steps_out": 3, "n_steps_in": 10, "n_kernel": 4, "n_filters": 64,
                 'n_nodes': 16, "n_batch": 64, "n_epochs": 20}
    good_configs.append((model_name, input_cfg, model_cfg, func_cfg))

    # %%
    model_cfg = {"n_steps_out": 1, "n_steps_in": 14, "n_kernel": 4, "n_filters": 128,
                 'n_nodes': 32, "n_batch": 128, "n_epochs": 20}
    good_configs.append((model_name, input_cfg, model_cfg, func_cfg))

    return good_configs[keys[steps]]
