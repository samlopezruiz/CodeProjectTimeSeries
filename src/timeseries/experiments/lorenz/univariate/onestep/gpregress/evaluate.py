from timeseries.data.lorenz.lorenz import lorenz_wrapper
from timeseries.experiments.lorenz.functions.harness import repeat_evaluate
from timeseries.experiments.lorenz.functions.summarize import summarize_scores
from timeseries.experiments.lorenz.functions.preprocessing import preprocess
from timeseries.experiments.lorenz.univariate.onestep.stroganoff.func import stroganoff_one_step_uv_predict, \
    stroganoff_one_step_uv_fit


if __name__ == '__main__':
    # %% GENERAL INPUTS
    detrend_ops = ['ln_return', ('ema_diff', 5), 'diff']
    save_folder = 'images'
    score_type = 'minmax'
    n_repeats = 10
    verbose = 0

    # MODEL AND TIME SERIES INPUTS
    name = "STROGANOFF"
    input_cfg = {"variate": "uni", "granularity": 5, "noise": False, 'preprocess': False,
                 'trend': False, 'detrend': 'diff'}
    model_cfg = {"n_steps_in": 10, "n_gen": 20, "n_pop": 200, "cxpb": 0.6, "mxpb": 0.1,
                 "depth": 5, 'elitism_size': 2, 'selection': 'roullete'}

    lorenz_df, train, test, t_train, t_test = lorenz_wrapper(input_cfg)
    train_pp, test_pp, ss = preprocess(input_cfg, train, test)

    # %% EVALUATE
    metrics, preds = repeat_evaluate(train_pp, test_pp, train, test, input_cfg, model_cfg,
                                     stroganoff_one_step_uv_predict, stroganoff_one_step_uv_fit,
                                     ss=ss, n_repeats=n_repeats)

    summarize_scores(name, metrics, score_type=score_type, input_cfg=input_cfg, model_cfg=model_cfg)