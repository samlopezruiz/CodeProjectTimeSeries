from timeseries.experiments.lorenz.univariate.onestep.arima.func import arima_get_one_step_uv_funcs


def arima_uv_configs(steps=1):
    keys = {1: 0}
    good_configs = []

    # %%
    model_name = "ARIMA"
    input_cfg = {"variate": "uni", "granularity": 5, "noise": True, 'preprocess': True,
                 'trend': True, 'detrend': 'ln_return'}
    model_cfg = {"n_steps_out": 1, 'order': (6, 0, 2)}
    func_cfg = arima_get_one_step_uv_funcs()
    good_configs.append((model_name, input_cfg, model_cfg, func_cfg))

    return good_configs[keys[steps]]
