from timeseries.data.lorenz.lorenz import univariate_lorenz
from timeseries.models.lorenz.functions.harness import repeat_evaluate, summarize_scores
from timeseries.models.lorenz.univariate.multistep.lstm.func import lstm_multi_step_uv_predict, lstm_multi_step_uv_fit


if __name__ == '__main__':
    name = "LSTM"
    lorenz_df, train, test, t_train, t_test = univariate_lorenz(granularity=5)
    # n_nput, n_nodes, n_epochs, n_batch = config
    cfg = (36, 7, 50, 100, 100)
    scores, preds = repeat_evaluate(train, test, cfg, lstm_multi_step_uv_predict,
                                    lstm_multi_step_uv_fit, steps=cfg[1], n_repeats=10)
    summarize_scores(name, scores)