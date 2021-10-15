import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from timeseries.data.lorenz.lorenz import lorenz_wrapper
from timeseries.experiments.lorenz.functions.harness import repeat_evaluate
from timeseries.experiments.lorenz.functions.summarize import summarize_scores
from timeseries.experiments.lorenz.functions.preprocessing import preprocess
from timeseries.experiments.lorenz.multivariate.multistep.dcnn.func import dcnn_get_multi_step_mv_funcs
#%%

if __name__ == '__main__':
    # %% GENERAL INPUTS
    score_type = 'minmax'
    n_repeats = 8
    verbose = 0

    # MODEL AND TIME SERIES INPUTS
    name = "D-CNN"
    input_cfg = {"variate": "multi", "granularity": 5, "noise": True, 'preprocess': True,
                 'trend': True, 'detrend': 'ln_return'}
    model_cfg = {"n_steps_out": 1, "n_steps_in": 5, 'n_layers': 2, "n_kernel": 2, 'reg': None,
                 "n_filters": 32, 'hidden_channels': 5, "n_batch": 256, "n_epochs": 25}
    func_cfg = dcnn_get_multi_step_mv_funcs()

    lorenz_df, train, test, t_train, t_test = lorenz_wrapper(input_cfg)
    train_pp, test_pp, ss = preprocess(input_cfg, train, test)

    # %% EVALUATE
    result = repeat_evaluate(train_pp, test_pp, train, test, input_cfg, model_cfg, func_cfg[0],
                             func_cfg[1], ss=ss, n_repeats=n_repeats, verbose=verbose)
    metrics, predictions, times, n_params, loss = result
    summarize_scores(name, metrics, score_type=score_type)
