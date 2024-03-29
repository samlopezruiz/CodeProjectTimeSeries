import os

from algorithms.gpregress.prim import primitives1
from timeseries.experiments.lorenz.multivariate.multistep.gpregress.func import gpregress_get_multi_step_mv_funcs
from timeseries.experiments.lorenz.multivariate.multistep.stroganoff.func import stroganoff_get_multi_step_mv_funcs

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from timeseries.experiments.lorenz.multivariate.multistep.cnn.func import cnn_get_multi_step_mv_funcs
from timeseries.data.lorenz.lorenz import lorenz_wrapper
from timeseries.experiments.lorenz.functions.harness import repeat_evaluate
from timeseries.experiments.lorenz.functions.summarize import summarize_scores
from timeseries.experiments.lorenz.functions.preprocessing import preprocess
#%%

if __name__ == '__main__':
    # %% GENERAL INPUTS
    score_type = 'minmax'
    n_repeats = 5
    verbose = 0

    # MODEL AND TIME SERIES INPUTS
    model_name = "GP-REGRESS"
    input_cfg = {"variate": "multi", "granularity": 5, "noise": True, 'preprocess': True,
                 'trend': True, 'detrend': 'ln_return'}
    model_cfg = {"n_steps_out": 1, "n_steps_in": 6, "depth": 8, "n_gen": 25, "n_pop": 200,
                 "cxpb": 0.2, "mxpb": 0.09, 'elitism_size': 5, 'selection': 'tournament', 'tour_size': 3,
                 'primitives': primitives1()}
    func_cfg = gpregress_get_multi_step_mv_funcs()

    lorenz_df, train, test, t_train, t_test = lorenz_wrapper(input_cfg)
    train_pp, test_pp, ss = preprocess(input_cfg, train, test)

    # %% EVALUATE
    result = repeat_evaluate(train_pp, test_pp, train, test, input_cfg, model_cfg, func_cfg[0],
                             func_cfg[1], ss=ss, n_repeats=n_repeats, verbose=verbose)
    metrics, predictions, times, n_params, loss = result
    summarize_scores(model_name, metrics, score_type=score_type)
