from timeseries.data.lorenz.lorenz import multivariate_lorenz
from timeseries.models.lorenz.functions.harness import repeat_evaluate
from timeseries.models.lorenz.functions.summarize import summarize_scores
from timeseries.models.lorenz.multivariate.multistep.cnn.func import cnn_multi_step_mv_predict, cnn_multi_step_mv_fit
from timeseries.models.lorenz.multivariate.onestep.cnn.func import cnn_one_step_mv_predict, cnn_one_step_mv_fit

if __name__ == '__main__':
    # %% INPUTS
    lorenz_df, train, test, t_train, t_test = multivariate_lorenz(granularity=5)
    name = "CNN"
    cfg = {"n_steps_in": 36, "n_steps_out": 10, "n_filters": 256, "n_kernel": 3,
           "n_epochs": 100, "n_batch": 100}

    # %% EVALUATE
    scores, preds = repeat_evaluate(train, test, cfg, cnn_multi_step_mv_predict, cnn_multi_step_mv_fit,
                                    steps=cfg['n_steps_out'], n_repeats=10)
    summarize_scores(name, scores)