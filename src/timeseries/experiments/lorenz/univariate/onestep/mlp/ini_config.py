from timeseries.data.lorenz.lorenz import univariate_lorenz
from timeseries.experiments.lorenz.functions.harness import repeat_evaluate
from timeseries.experiments.lorenz.functions.summarize import summarize_scores
from timeseries.experiments.lorenz.univariate.onestep.mlp.func import mlp_one_step_uv_predict, mlp_one_step_uv_fit


if __name__ == '__main__':
    lorenz_df, train, test, t_train, t_test = univariate_lorenz()
    # n_nput, n_nodes, n_epochs, n_batch = architectures
    cfg = (24, 500, 100, 100)
    scores = repeat_evaluate(train, test, cfg, mlp_one_step_uv_predict, mlp_one_step_uv_fit)
    summarize_scores('persistence', scores)