from timeseries.data.lorenz.lorenz import univariate_lorenz
from timeseries.models.lorenz.functions.harness import repeat_evaluate, summarize_scores
from timeseries.models.lorenz.univariate.multistep.cnnlstm.func import cnnlstm_multi_step_uv_predict, \
    cnnlstm_multi_step_uv_fit
from timeseries.models.lorenz.univariate.onestep.cnnlstm.func import cnnlstm_one_step_uv_predict, \
    cnnlstm_one_step_uv_fit
from timeseries.models.lorenz.univariate.onestep.mlp.func import mlp_one_step_uv_predict, mlp_one_step_uv_fit

if __name__ == '__main__':
    # %% INPUT
    lorenz_df, train, test, t_train, t_test = univariate_lorenz(granularity=5)
    name = "CNN-LSTM"
    cfg = {"n_seq": 3, "n_steps_in": 12, "n_steps_out": 10, "n_filters": 64,
           "n_kernel": 3, "n_nodes": 200, "n_epochs": 100, "n_batch": 100}

    # %% EVALUATE
    scores, preds = repeat_evaluate(train, test, cfg, cnnlstm_multi_step_uv_predict, cnnlstm_multi_step_uv_fit,
                                    steps=cfg["n_steps_out"], n_repeats=10)
    summarize_scores(name, scores)