from timeseries.data.lorenz.lorenz import univariate_lorenz
from timeseries.experiments.lorenz.functions.harness import repeat_evaluate
from timeseries.experiments.lorenz.functions.summarize import summarize_scores
from timeseries.experiments.lorenz.univariate.onestep.convlstm.func import convlstm_one_step_uv_fit, \
    convlstm_one_step_uv_predict

if __name__ == '__main__':
    # %% INPUT
    lorenz_df, train, test, t_train, t_test = univariate_lorenz(granularity=5)
    name = "CONV-LSTM"
    cfg = {"n_seq": 3, "n_steps_in": 12, "n_filters": 256,
           "n_kernel": 3, "n_nodes": 200, "n_epochs": 100, "n_batch": 100}

    #%% EVALUATE
    scores, preds = repeat_evaluate(train, test, cfg, convlstm_one_step_uv_predict, convlstm_one_step_uv_fit,
                                    n_repeats=10)
    summarize_scores(name, scores)