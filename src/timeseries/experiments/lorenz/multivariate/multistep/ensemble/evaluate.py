import os

from timeseries.experiments.lorenz.multivariate.multistep.configs.convlstm import convlstm_mv_configs
from timeseries.experiments.lorenz.multivariate.multistep.configs.dcnn import dcnn_mv_configs
from timeseries.experiments.lorenz.multivariate.multistep.configs.wavenet import wavenet_mv_configs

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from timeseries.experiments.lorenz.multivariate.multistep.configs.cnn import cnn_mv_configs
from timeseries.experiments.lorenz.multivariate.multistep.configs.cnnlstm import cnnlstm_mv_configs
from timeseries.experiments.lorenz.multivariate.multistep.ensemble.func import ensemble_get_multi_step_mv_funcs
from timeseries.data.lorenz.lorenz import lorenz_wrapper
from timeseries.experiments.lorenz.functions.harness import repeat_evaluate
from timeseries.experiments.lorenz.functions.summarize import summarize_scores
from timeseries.experiments.lorenz.functions.preprocessing import preprocess

# %%

if __name__ == '__main__':
    # %% GENERAL INPUTS
    score_type = 'minmax'
    n_repeats = 3
    verbose = 0
    steps = 6
    # MODEL AND TIME SERIES INPUTS
    name = "ENSEMBLE"
    input_cfg = {"variate": "multi", "granularity": 5, "noise": True, 'preprocess': True,
                 'trend': True, 'detrend': 'ln_return'}
    model_cfg = [dcnn_mv_configs(steps=6),
                 cnnlstm_mv_configs(steps=6)]
    func_cfg = ensemble_get_multi_step_mv_funcs()

    lorenz_df, train, test, t_train, t_test = lorenz_wrapper(input_cfg)
    train_pp, test_pp, ss = preprocess(input_cfg, train, test)

    # %% EVALUATE
    result = repeat_evaluate(train_pp, test_pp, train, test, input_cfg, model_cfg, func_cfg[0],
                             func_cfg[1], ss=ss, n_repeats=n_repeats, verbose=verbose)
    metrics, predictions, times, n_params, loss = result
    summarize_scores(name, metrics, score_type=score_type)
