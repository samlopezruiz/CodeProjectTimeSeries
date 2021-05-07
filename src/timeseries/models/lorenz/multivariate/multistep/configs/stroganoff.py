from algorithms.wavenet.func import wavenet_build
from timeseries.models.lorenz.multivariate.multistep.dcnn.func import dcnn_get_multi_step_mv_funcs
from timeseries.models.lorenz.multivariate.multistep.stroganoff.func import stroganoff_get_multi_step_mv_funcs
from timeseries.models.lorenz.multivariate.multistep.wavenet.func import wavenet_multi_step_mv_predict, \
    wavenet_multi_step_mv_fit


def stroganoff_mv_configs(steps=1):
    keys = {6: 0, 3: 1, 1: 2}
    good_configs = []

    # %% STROGANOFF: 0.89 minmax  (+/- 0.0028)
    model_name = "STROGANOFF"
    input_cfg = {"variate": "multi", "granularity": 5, "noise": True, 'preprocess': True,
                 'trend': True, 'detrend': 'ln_return'}
    model_cfg = {"n_steps_out": 6, "n_steps_in": 6, "depth": 4, "n_gen": 15, "n_pop": 400,
                 "cxpb": 0.6, "mxpb": 0.03, 'elitism_size': 2, 'selection': 'tournament', 'tour_size': 5}
    functions = stroganoff_get_multi_step_mv_funcs()
    good_configs.append((model_name, input_cfg, model_cfg, functions))

    # %% STROGANOFF: 0.9232 minmax  (+/- 0.0011) STEPS=3
    model_name = "STROGANOFF"
    input_cfg = {"variate": "multi", "granularity": 5, "noise": True, 'preprocess': True,
                 'trend': True, 'detrend': 'ln_return'}
    model_cfg = {"n_steps_out": 3, "n_steps_in": 10, "depth": 8, "n_gen": 35, "n_pop": 400,
                 "cxpb": 0.6, "mxpb": 0.05, 'elitism_size': 2, 'selection': 'roullete', 'tour_size': 5}
    functions = stroganoff_get_multi_step_mv_funcs()
    good_configs.append((model_name, input_cfg, model_cfg, functions))

    # %% STROGANOFF: 0.9544 minmax  (+/- 0.0008) STEPS=1
    model_name = "STROGANOFF"
    input_cfg = {"variate": "multi", "granularity": 5, "noise": True, 'preprocess': True,
                 'trend': True, 'detrend': 'ln_return'}
    model_cfg = {"n_steps_out": 1, "n_steps_in": 14, "depth": 4, "n_gen": 5, "n_pop": 200,
                 "cxpb": 0.6, "mxpb": 0.07, 'elitism_size': 2, 'selection': 'tournament', 'tour_size': 5}
    functions = stroganoff_get_multi_step_mv_funcs()
    good_configs.append((model_name, input_cfg, model_cfg, functions))

    return good_configs[keys[steps]]
