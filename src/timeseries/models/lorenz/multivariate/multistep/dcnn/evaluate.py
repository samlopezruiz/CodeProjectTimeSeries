from timeseries.data.lorenz.lorenz import univariate_lorenz, multivariate_lorenz, lorenz_wrapper
from timeseries.models.lorenz.functions.harness import repeat_evaluate, summarize_scores
from timeseries.models.lorenz.functions.preprocessing import preprocess
from timeseries.models.lorenz.multivariate.multistep.convlstm.func import convlstm_multi_step_mv_predict, \
    convlstm_multi_step_mv_fit
from timeseries.models.lorenz.multivariate.multistep.dcnn.func import dcnn_multi_step_mv_predict, dcnn_multi_step_mv_fit
from timeseries.models.lorenz.multivariate.multistep.wavenet.func import wavenet_multi_step_mv_predict, \
    wavenet_multi_step_mv_fit
from timeseries.models.lorenz.multivariate.onestep.convlstm.func import convlstm_one_step_mv_predict, \
    convlstm_one_step_mv_fit
from timeseries.models.lorenz.univariate.onestep.convlstm.func import convlstm_one_step_uv_fit, \
    convlstm_one_step_uv_predict


if __name__ == '__main__':
    # %% GENERAL INPUTS
    detrend_ops = ['ln_return', ('ema_diff', 5), 'diff']
    save_folder = 'images'
    score_type = 'minmax'
    n_repeats = 5
    verbose = 0

    # MODEL AND TIME SERIES INPUTS
    name = "DCNN"
    input_cfg = {"variate": "multi", "granularity": 5, "noise": True, 'preprocess': True,
                 'trend': True, 'detrend': 'ln_return'}
    model_cfg = {"n_steps_in": 50, "n_steps_out": 6, 'n_layers': 5, "n_filters": 50,
                 "n_kernel": 3, "n_epochs": 20, "n_batch": 32, 'reg': 'l2'}

    lorenz_df, train, test, t_train, t_test = lorenz_wrapper(input_cfg)
    train_pp, test_pp, ss = preprocess(input_cfg, train, test)

    # %% EVALUATE
    metrics, preds = repeat_evaluate(train_pp, test_pp, train, test, input_cfg, model_cfg,
                                     dcnn_multi_step_mv_predict, dcnn_multi_step_mv_fit,
                                     ss=ss, n_repeats=n_repeats, verbose=verbose)

    summarize_scores(name, metrics, score_type=score_type, input_cfg=input_cfg, model_cfg=model_cfg)