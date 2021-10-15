from timeseries.data.lorenz.lorenz import multivariate_lorenz
from timeseries.experiments.lorenz.functions.harness import repeat_evaluate
from timeseries.experiments.lorenz.functions.summarize import summarize_scores
from timeseries.experiments.lorenz.multivariate.multistep.mlp.func import mlp_multi_step_mv_predict, mlp_multi_step_mv_fit

if __name__ == '__main__':
    #%% INPUTS
    lorenz_df, train, test, t_train, t_test = multivariate_lorenz(granularity=5)
    name = "MLP"
    cfg = {"n_steps_in": 24, "n_steps_out": 10, "n_nodes": 500,
           "n_epochs": 100, "n_batch": 100}

    #%% EVALUATE
    scores, preds = repeat_evaluate(train, test, cfg, mlp_multi_step_mv_predict, mlp_multi_step_mv_fit,
                                       steps=cfg['n_steps_out'], n_repeats=10)
    summarize_scores(name, scores)