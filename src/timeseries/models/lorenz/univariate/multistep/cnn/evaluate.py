from timeseries.data.lorenz.lorenz import univariate_lorenz
from timeseries.models.lorenz.functions.harness import repeat_evaluate, summarize_scores
from timeseries.models.lorenz.univariate.multistep.cnn.func import cnn_multi_step_uv_predict, cnn_multi_step_uv_fit

if __name__ == '__main__':
    # %% INPUTS
    lorenz_df, train, test, t_train, t_test = univariate_lorenz(granularity=5)
    name = "CNN"
    cfg = {"n_steps_in": 36, "n_steps_out": 7, "n_filters": 256,
           "n_kernel": 3, "n_epochs": 100, "n_batch": 100}

    # %% EVALUATE
    scores, preds = repeat_evaluate(train, test, cfg, cnn_multi_step_uv_predict, cnn_multi_step_uv_fit,
                                    steps=cfg["n_steps_out"], n_repeats=10)
    summarize_scores(name, scores)