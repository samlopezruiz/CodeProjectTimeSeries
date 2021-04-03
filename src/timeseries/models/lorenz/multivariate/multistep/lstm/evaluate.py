from timeseries.data.lorenz.lorenz import univariate_lorenz, multivariate_lorenz
from timeseries.models.lorenz.functions.harness import repeat_evaluate, summarize_scores
from timeseries.models.lorenz.multivariate.multistep.lstm.func import lstm_multi_step_mv_predict, lstm_multi_step_mv_fit
from timeseries.models.lorenz.multivariate.onestep.lstm.func import lstm_one_step_mv_fit, lstm_one_step_mv_predict
from timeseries.models.lorenz.univariate.onestep.lstm.func import lstm_one_step_uv_predict, lstm_one_step_uv_fit

if __name__ == '__main__':
    # %% INPUT
    lorenz_df, train, test, t_train, t_test = multivariate_lorenz(granularity=5)
    name = "LSTM"
    cfg = {"n_steps_in": 36, "n_steps_out": 10, "n_nodes": 50, "n_epochs": 100, "n_batch": 100}

    # %% EVALUATE
    scores, preds = repeat_evaluate(train, test, cfg, lstm_multi_step_mv_predict, lstm_multi_step_mv_fit,
                                    steps=cfg['n_steps_out'], n_repeats=10)
    summarize_scores(name, scores)