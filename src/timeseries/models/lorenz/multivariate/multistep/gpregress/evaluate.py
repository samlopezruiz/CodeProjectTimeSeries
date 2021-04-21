from timeseries.data.lorenz.lorenz import lorenz_wrapper
from timeseries.models.lorenz.functions.harness import repeat_evaluate, summarize_scores
from timeseries.models.lorenz.functions.preprocessing import preprocess
from timeseries.models.lorenz.multivariate.multistep.stroganoff.func import stroganoff_multi_step_mv_predict, \
    stroganoff_multi_step_mv_fit
from timeseries.models.lorenz.univariate.onestep.stroganoff.func import stroganoff_one_step_uv_predict, \
    stroganoff_one_step_uv_fit


if __name__ == '__main__':
    # %% GENERAL INPUTS
    detrend_ops = ['ln_return', ('ema_diff', 5), 'diff']
    save_folder = 'images'
    score_type = 'minmax'
    n_repeats = 3
    verbose = 0

    # MODEL AND TIME SERIES INPUTS
    name = "STROGANOFF"
    input_cfg = {"variate": "multi", "granularity": 5, "noise": True, 'preprocess': True,
                 'trend': True, 'detrend': 'ln_return'}
    model_cfg = {"n_steps_in": 5, "n_steps_out": 2, "n_gen": 10, "n_pop": 300, "cxpb": 0.8, "mxpb": 0.05,
                 "depth": 7, 'elitism_size': 2, 'selection': 'tournament', 'tour_size': 5}

    lorenz_df, train, test, t_train, t_test = lorenz_wrapper(input_cfg)
    train_pp, test_pp, ss = preprocess(input_cfg, train, test)

    # %% EVALUATE
    metrics, preds = repeat_evaluate(train_pp, test_pp, train, test, input_cfg, model_cfg,
                                     stroganoff_multi_step_mv_predict, stroganoff_multi_step_mv_fit,
                                     ss=ss, n_repeats=n_repeats)

    summarize_scores(name, metrics, score_type=score_type, input_cfg=input_cfg, model_cfg=model_cfg)