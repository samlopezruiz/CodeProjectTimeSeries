from algorithms.gpregress.prim import primitives1
from algorithms.wavenet.func import wavenet_build
from timeseries.models.lorenz.multivariate.multistep.dcnn.func import dcnn_get_multi_step_mv_funcs
from timeseries.models.lorenz.multivariate.multistep.gpregress.func import gpregress_get_multi_step_mv_funcs
from timeseries.models.lorenz.multivariate.multistep.stroganoff.func import stroganoff_get_multi_step_mv_funcs
from timeseries.models.lorenz.multivariate.multistep.wavenet.func import wavenet_multi_step_mv_predict, \
    wavenet_multi_step_mv_fit


def gpregress_mv_configs(steps=1):
    keys = {6: 0, 3: 1, 1: 2}
    good_configs = []

    # %% D-CNN: 0.9405 minmax  (+/- 0.0025)  STEPS=6
    model_name = "GP-REGRESS"
    input_cfg = {"variate": "multi", "granularity": 5, "noise": True, 'preprocess': True,
                 'trend': True, 'detrend': 'ln_return'}
    model_cfg = {"n_steps_in": 10, "n_steps_out": 6, "n_gen": 10, "n_pop": 400, "cxpb": 0.7,
                 "mxpb": 0.1, "depth": 5, 'elitism_size': 5, 'selection': 'tournament', 'tour_size': 3,
                 'primitives': primitives1()}
    functions = gpregress_get_multi_step_mv_funcs()
    good_configs.append((model_name, input_cfg, model_cfg, functions))

    # %% D-CNN: 0.9477 minmax  (+/- 0.0015) STEPS=3
    model_name = "GP-REGRESS"
    input_cfg = {"variate": "multi", "granularity": 5, "noise": True, 'preprocess': True,
                 'trend': True, 'detrend': 'ln_return'}
    model_cfg = {"n_steps_in": 10, "n_steps_out": 3, "n_gen": 10, "n_pop": 400, "cxpb": 0.7,
                 "mxpb": 0.1, "depth": 5, 'elitism_size': 5, 'selection': 'tournament', 'tour_size': 3,
                 'primitives': primitives1()}
    functions = gpregress_get_multi_step_mv_funcs()
    good_configs.append((model_name, input_cfg, model_cfg, functions))

    # %% WAVENET: 0.9568 minmax  (+/- 0.0012) STEPS=1
    model_name = "GP-REGRESS"
    input_cfg = {"variate": "multi", "granularity": 5, "noise": True, 'preprocess': True,
                 'trend': True, 'detrend': 'ln_return'}
    model_cfg = {"n_steps_in": 10, "n_steps_out": 1, "n_gen": 10, "n_pop": 400, "cxpb": 0.7,
                 "mxpb": 0.1, "depth": 5, 'elitism_size': 5, 'selection': 'tournament', 'tour_size': 3,
                 'primitives': primitives1()}
    functions = gpregress_get_multi_step_mv_funcs()
    good_configs.append((model_name, input_cfg, model_cfg, functions))

    return good_configs[keys[steps]]
