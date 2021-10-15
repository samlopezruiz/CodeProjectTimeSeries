from algorithms.gpregress.prim import primitives1
from timeseries.experiments.lorenz.multivariate.multistep.gpregress.func import gpregress_get_multi_step_mv_funcs


def gpregress_mv_configs(steps=1):
    keys = {6: 0, 3: 1, 1: 2}
    good_configs = []

    # %% GP-REGRESS: 0.8855 minmax  (+/- 0.001)  STEPS=6
    model_name = "GP-REGRESS"
    input_cfg = {"variate": "multi", "granularity": 5, "noise": True, 'preprocess': True,
                 'trend': True, 'detrend': 'ln_return'}
    model_cfg = {"n_steps_out": 6, "n_steps_in": 14, "depth": 8, "n_gen": 35, "n_pop": 300,
                 "cxpb": 0.6, "mxpb": 0.05, 'elitism_size': 5, 'selection': 'roullete', 'tour_size': 3,
                 'primitives': primitives1()}
    functions = gpregress_get_multi_step_mv_funcs()
    good_configs.append((model_name, input_cfg, model_cfg, functions))

    # %% GP-REGRESS: 0.9334 minmax  (+/- 0.0009) STEPS=3
    model_name = "GP-REGRESS"
    input_cfg = {"variate": "multi", "granularity": 5, "noise": True, 'preprocess': True,
                 'trend': True, 'detrend': 'ln_return'}
    model_cfg = {"n_steps_out": 3, "n_steps_in": 6, "depth": 4, "n_gen": 5, "n_pop": 400,
                 "cxpb": 0.2, "mxpb": 0.09, 'elitism_size': 5, 'selection': 'roullete', 'tour_size': 3,
                 'primitives': primitives1()}
    functions = gpregress_get_multi_step_mv_funcs()
    good_configs.append((model_name, input_cfg, model_cfg, functions))

    # %% GP-REGRESS: 0.9519 minmax  (+/- 0.0009) STEPS=1
    model_name = "GP-REGRESS"
    input_cfg = {"variate": "multi", "granularity": 5, "noise": True, 'preprocess': True,
                 'trend': True, 'detrend': 'ln_return'}
    model_cfg = {"n_steps_out": 1, "n_steps_in": 6, "depth": 8, "n_gen": 25, "n_pop": 200,
                 "cxpb": 0.2, "mxpb": 0.09, 'elitism_size': 5, 'selection': 'tournament', 'tour_size': 3,
                 'primitives': primitives1()}
    functions = gpregress_get_multi_step_mv_funcs()
    good_configs.append((model_name, input_cfg, model_cfg, functions))

    return good_configs[keys[steps]]
