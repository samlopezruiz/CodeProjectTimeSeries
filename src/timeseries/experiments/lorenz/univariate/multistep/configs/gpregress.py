from algorithms.gpregress.prim import primitives1
from timeseries.experiments.lorenz.multivariate.multistep.gpregress.func import gpregress_get_multi_step_mv_funcs
from timeseries.experiments.lorenz.univariate.onestep.gpregress.func import gpregress_get_one_step_uv_funcs


def gpregress_uv_configs(steps=1):
    keys = {1: 0}
    good_configs = []

    # %% GP-REGRESS: 0.8855 minmax  (+/- 0.001)  STEPS=6
    model_name = "GP-REGRESS"
    input_cfg = {"variate": "uni", "granularity": 5, "noise": True, 'preprocess': True,
                 'trend': True, 'detrend': 'ln_return'}
    model_cfg = {"n_steps_out": 1, "n_steps_in": 6, "depth": 8, "n_gen": 25, "n_pop": 200,
                 "cxpb": 0.2, "mxpb": 0.09, 'elitism_size': 5, 'selection': 'tournament', 'tour_size': 3,
                 'primitives': primitives1()}
    functions = gpregress_get_one_step_uv_funcs()
    good_configs.append((model_name, input_cfg, model_cfg, functions))

    return good_configs[keys[steps]]
