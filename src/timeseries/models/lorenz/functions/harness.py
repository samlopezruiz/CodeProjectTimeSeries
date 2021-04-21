import time
from numpy import mean
from numpy import std
from matplotlib import pyplot
from timeseries.data.lorenz.lorenz import univariate_lorenz
from timeseries.models.lorenz.functions.functions import walk_forward_step_forecast
from timeseries.models.lorenz.functions.preprocessing import reconstruct
from timeseries.models.lorenz.univariate.onestep.simple.func import simple_forecast, simple_fit
from timeseries.models.utils.metrics import forecast_accuracy, summary_results
from timeseries.plotly.plot import plot_scores
import numpy as np
from itertools import product
from copy import copy

from multiprocessing import cpu_count
from warnings import catch_warnings, filterwarnings
from joblib import Parallel, delayed
from collections import defaultdict
from tqdm import tqdm

#%%
# # patch joblib progress callback
class BatchCompletionCallBack(object):
    completed = defaultdict(int)

    def __init__(self, time, index, parallel):
        self.index = index
        self.parallel = parallel

    def __call__(self, index):
        BatchCompletionCallBack.completed[self.parallel] += 1
        print("Parallel execution done with work: {}".format(BatchCompletionCallBack.completed[self.parallel]))
        if self.parallel._original_iterator is not None:
            self.parallel.dispatch_next()

import joblib
joblib.parallel.BatchCompletionCallBack = BatchCompletionCallBack

#%%
def difference(data, interval):
    return [data[i] - data[i - interval] for i in range(interval, len(data))]


def repeat_evaluate_old(train_pp, test_pp, train, test, input_cfg, cfg, model_forecast, model_fit, ss, n_repeats=30,
                        walkforward=walk_forward_step_forecast, verbose=0, parallel=True):
    metrics, predictions, times, n_params = [], [], [], 0
    start_t = time.time()
    for i in range(n_repeats):
        forecast, train_t, pred_t, n_params = walkforward(train_pp, test_pp, cfg, model_forecast,
                                                          model_fit, steps=cfg.get('n_steps_out', 1), verbose=verbose)
        forecast_reconst = reconstruct(forecast, train, test, input_cfg, cfg, ss=ss)
        metric = forecast_accuracy(forecast_reconst, test[:, -1] if input_cfg['variate'] == 'multi' else test)
        metrics.append(metric)
        predictions.append(forecast)
        times.append((train_t, pred_t))
        print("{}/{} - ({}%) in {}s".format(i + 1, n_repeats, round(100 * (i + 1) / n_repeats),
                                            round(time.time() - start_t)), end="\r")
    return metrics, predictions, times, n_params


def repeat_evaluate(train_pp, test_pp, train, test, input_cfg, cfg, model_forecast, model_fit, ss, n_repeats=30,
                    walkforward=walk_forward_step_forecast, verbose=0, parallel=True):
    metrics, predictions, times, n_params = [], [], [], 0
    start_t = time.time()

    results = parallel_walkforward(cfg, model_fit, model_forecast, n_repeats, parallel, test_pp, train_pp, verbose,
                                   walkforward)
    i = 0
    for forecast, train_t, pred_t, n_params in results:
        forecast_reconst = reconstruct(forecast, train, test, input_cfg, cfg, ss=ss)
        metric = forecast_accuracy(forecast_reconst, test[:, -1] if input_cfg['variate'] == 'multi' else test)
        metrics.append(metric)
        predictions.append(forecast)
        times.append((train_t, pred_t))
        print("{}/{} - ({}%) in {}s".format(i + 1, n_repeats, round(100 * (i + 1) / n_repeats),
                                            round(time.time() - start_t)), end="\r")
        i += 1
    return metrics, predictions, times, n_params

def parallel_walkforward(cfg, model_fit, model_forecast, n_repeats, parallel, test_pp, train_pp, verbose, walkforward):
    if parallel:
        executor = Parallel(n_jobs=cpu_count(), backend='multiprocessing')
        tasks = (delayed(walkforward)(train_pp, test_pp, cfg, model_forecast,
                                      model_fit, steps=cfg.get('n_steps_out', 1), verbose=verbose) for _ in
                 tqdm(range(n_repeats)))
        results = executor(tasks)
    else:
        results = [walkforward(train_pp, test_pp, cfg, model_forecast, model_fit,
                               steps=cfg.get('n_steps_out', 1), verbose=verbose) for _ in tqdm(range(n_repeats))]
    return results


def evaluate_models(input_cfg, names, model_cfgs, functions, n_repeats, ss, score_type, data_in):
    parallel = len(model_cfgs) > n_repeats
    results = parallel_repeat_eval(model_cfgs, parallel, data_in, ss, input_cfg, functions, n_repeats)
    scores, times, params = consolidate_summary(results, names, score_type)
    summary, data, errors = summary_results(scores, times, params, score_type)
    return summary, data, errors


def parallel_repeat_eval(cfgs, parallel, data_in, ss, input_cfg, functions, n_repeats):
    train_pp, test_pp, train, test = data_in
    if parallel:
        executor = Parallel(n_jobs=cpu_count(), backend='multiprocessing')
        tasks = (delayed(repeat_evaluate)(train_pp, test_pp, train, test, input_cfg, cfg, functions[i][0], functions[i][1],
                                          ss=ss, n_repeats=n_repeats, parallel=not parallel) for i, cfg in tqdm(enumerate(cfgs)))
        results = executor(tasks)
    else:
        results = [repeat_evaluate(train_pp, test_pp, train, test, input_cfg, cfg, functions[i][0],
                                   functions[i][1], ss=ss, n_repeats=n_repeats, parallel=not parallel) for i, cfg in tqdm(enumerate(cfgs))]
    return results


def gs_get_cfgs(gs_cfg, model_cfg):
    keys = list(gs_cfg.keys())
    ranges = list(gs_cfg.values())
    combinations = list(product(*ranges))
    name_cfg = copy(gs_cfg)

    cfgs_gs, names = [], []
    for comb in combinations:
        new_cfg = copy(model_cfg)
        for k, key in enumerate(keys):
            new_cfg[key] = comb[k]
            name_cfg[key] = comb[k]
        cfgs_gs.append(new_cfg)
        names.append(str(name_cfg))

    return cfgs_gs, names


def grid_search(input_cfg, gs_cfg, model_cfg, function, n_repeats, score_type, ss, data_in):
    cfgs_gs, names = gs_get_cfgs(gs_cfg, model_cfg)
    parallel = len(cfgs_gs) > n_repeats
    functions = [function for _ in range(len(cfgs_gs))]
    results = parallel_repeat_eval(cfgs_gs, parallel, data_in, ss, input_cfg, functions, n_repeats)
    cfg_scores, cfg_times, cfg_params = consolidate_summary(results, names, score_type)
    summary, data, errors = summary_results(cfg_scores, cfg_times, cfg_params, score_type)
    return summary, data, errors


def consolidate_summary(results, names, score_type):
    cfg_scores, cfg_times, cfg_params = [], [], []
    for i, res in enumerate(results):
        metrics, _, times, n_params = res
        scores, scores_m, score_std = summarize_scores(names[i], metrics, score_type)
        train_t_m, train_t_std, pred_t_m, pred_t_std = summarize_times(names[i], times)
        cfg_times.append((names[i], times, train_t_m, train_t_std, pred_t_m, pred_t_std))
        cfg_scores.append((names[i], scores, scores_m, score_std))
        cfg_params.append((names[i], n_params))

    return cfg_scores, cfg_times, cfg_params


def summarize_scores(name, metrics, score_type='rmse'):
    scores = [m[score_type] for m in metrics]
    scores_m, score_std = mean(scores), std(scores)
    print('{}: {} {}  (+/- {})'.format(name, round(scores_m, 4), score_type, round(score_std, 4)))
    return scores, scores_m, score_std


def summarize_times(name, times):
    times = np.array(times)
    train_t, pred_t = times[:, 0], times[:, 1]
    train_t_m, train_t_std = mean(train_t), std(train_t)
    pred_t_m, pred_t_std = mean(pred_t), std(pred_t)
    print('{}: train = {} s  (+/- {}), pred = {} s  (+/- {})'.format(name, round(train_t_m, 1), round(train_t_std, 4),
                                                                     round(pred_t_m, 4), round(pred_t_std, 4)))
    return train_t_m, train_t_std, pred_t_m, pred_t_std


if __name__ == '__main__':
    lorenz_df, train, test, t_train, t_test = univariate_lorenz()
    cfg = (1, 1, 'persist')
    scores = repeat_evaluate(train, test, cfg, simple_forecast, simple_fit)
    summarize_scores('persistence', scores)
