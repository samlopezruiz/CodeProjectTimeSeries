from timeseries.data.lorenz.lorenz import univariate_lorenz
from timeseries.models.lorenz.functions.harness import repeat_evaluate, summarize_scores
from timeseries.models.lorenz.univariate.multistep.mlp.func import mlp_multi_step_uv_fit, mlp_multi_step_uv_predict

if __name__ == '__main__':
    # %% INPUTS
    lorenz_df, train, test, t_train, t_test = univariate_lorenz(granularity=5)
    name = "MLP"
    cfg = {"n_steps_in": 24, "n_steps_out": 10, "n_nodes": 500,
           "n_epochs": 100, "n_batch": 100}

    #%% EVALUATE
    scores, preds = repeat_evaluate(train, test, cfg, mlp_multi_step_uv_predict, mlp_multi_step_uv_fit,
                                    n_repeats=10, steps=cfg["n_steps_out"])
    summarize_scores(name, scores)
