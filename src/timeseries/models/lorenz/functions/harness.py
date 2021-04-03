import time
from numpy import mean
from numpy import std
from matplotlib import pyplot
from timeseries.data.lorenz.lorenz import univariate_lorenz
from timeseries.models.lorenz.functions.functions import walk_forward_step_forecast
from timeseries.models.lorenz.functions.preprocessing import reconstruct
from timeseries.models.lorenz.univariate.onestep.simple.func import simple_forecast, simple_fit
from timeseries.models.utils.metrics import forecast_accuracy
from timeseries.plotly.plot import plot_scores


def difference(data, interval):
    return [data[i] - data[i - interval] for i in range(interval, len(data))]


def repeat_evaluate(train_pp, test_pp, train, test, input_cfg, cfg, model_forecast, model_fit, ss, n_repeats=30):
    metrics = []
    predictions = []
    start_t = time.time()
    for i in range(n_repeats):
        forecast = walk_forward_step_forecast(train_pp, test_pp, cfg, model_forecast,
                                              model_fit, steps=cfg.get('n_steps_out', 1))
        forecast_reconst = reconstruct(forecast, train, test, input_cfg, cfg, ss=ss)
        metric = forecast_accuracy(forecast_reconst, test[:, -1])
        metrics.append(metric)
        predictions.append(forecast)
        print("{}/{} - ({}%) in {}s".format(i+1, n_repeats, round(100*(i+1)/n_repeats),
                                            round(time.time() - start_t)), end="\r")
    return metrics, predictions


def summarize_scores(name, metrics, score_type='rmse', input_cfg=None, model_cfg=None):
    scores = [m[score_type] for m in metrics]
    scores_m, score_std = mean(scores), std(scores)
    print('{}: {} {}  (+/- {})'.format(name, round(scores_m, 4), score_type, round(score_std, 4)))
    plot_scores(scores, score_type=score_type, title="SERIES: " + str(input_cfg) + '<br>' + name + ': ' + str(model_cfg))
    # pyplot.boxplot(scores)
    # pyplot.show()


if __name__ == '__main__':
    lorenz_df, train, test, t_train, t_test = univariate_lorenz()
    cfg = (1, 1, 'persist')
    scores = repeat_evaluate(train, test, cfg, simple_forecast, simple_fit)
    summarize_scores('persistence', scores)
