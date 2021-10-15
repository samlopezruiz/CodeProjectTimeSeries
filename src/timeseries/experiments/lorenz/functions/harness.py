import time

from timeseries.data.lorenz.lorenz import lorenz_wrapper
from timeseries.experiments.lorenz.functions.functions import walk_forward_step_forecast, ismv
from timeseries.experiments.lorenz.functions.preprocessing import reconstruct, preprocess
from timeseries.experiments.lorenz.functions.summarize import consolidate_summary, consolidate_series_summaries
from timeseries.experiments.utils.config import unpack_in_cfg
from timeseries.experiments.utils.forecast import multi_step_forecast_df, multi_step_forecasts_df
from timeseries.experiments.utils.metrics import forecast_accuracy, summary_results
from timeseries.experiments.utils.models import models_strings, save_vars, get_suffix
from timeseries.plotly.plot import plotly_time_series, plot_bar_summary
from itertools import product
from copy import copy
from multiprocessing import cpu_count
from joblib import Parallel, delayed
from collections import defaultdict
from tqdm import tqdm


# %%
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


# %%
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
                    walkforward=walk_forward_step_forecast, verbose=0, parallel=True, debug=False, n_steps_out=None):
    if debug: print(model_forecast)
    results = parallel_walkforward(cfg, model_fit, model_forecast, n_repeats, parallel, test_pp, train_pp, verbose,
                                   walkforward, debug=debug, n_steps_out=n_steps_out)
    if n_steps_out is None:
        n_steps_out = cfg[0][2].get('n_steps_out', 1) if isinstance(cfg, list) else cfg.get('n_steps_out', 1)
    metrics, predictions, times, n_params, loss = [], [], [], 0, []
    for forecast, train_t, pred_t, n_params, train_loss in results:
        forecast_reconst = reconstruct(forecast, train, test, input_cfg, n_steps_out, ss=ss)
        metric = forecast_accuracy(forecast_reconst, test[:, -1] if input_cfg['variate'] == 'multi' else test)
        metrics.append(metric)
        predictions.append(forecast)
        times.append((train_t, pred_t))
        loss.append(train_loss)
    return metrics, predictions, times, n_params, loss


def parallel_walkforward(cfg, model_fit, model_forecast, n_repeats, parallel, test_pp,
                         train_pp, verbose, walkforward, debug=False, n_steps_out=None):
    if n_steps_out is None:
        n_steps_out = cfg[0][2].get('n_steps_out', 1) if isinstance(cfg, list) else cfg.get('n_steps_out', 1)
    if parallel:
        executor = Parallel(n_jobs=cpu_count(), backend='multiprocessing')
        tasks = (delayed(walkforward)(train_pp, test_pp, cfg, model_forecast,
                                      model_fit, steps=n_steps_out, verbose=verbose) for _ in
                 tqdm(range(n_repeats)))
        results = executor(tasks)
    else:
        if debug:
            results = [walkforward(train_pp, test_pp, cfg, model_forecast, model_fit,
                                   steps=n_steps_out, verbose=verbose) for _ in range(n_repeats)]
        else:
            results = [walkforward(train_pp, test_pp, cfg, model_forecast, model_fit,
                                   steps=n_steps_out, verbose=verbose) for _ in tqdm(range(n_repeats))]
    return results


def evaluate_models(input_cfg, names, model_cfgs, functions, n_repeats, ss, score_type, data_in, debug=False):
    parallel = len(model_cfgs) > n_repeats and not debug
    results = parallel_repeat_eval(model_cfgs, parallel, data_in, ss, input_cfg, functions, n_repeats, debug=debug)
    consolidate = consolidate_summary(results, names, score_type)
    summary, data, errors = summary_results(consolidate, score_type)
    return results, summary, data, errors


def evaluate_models_series(in_cfg, input_cfg, names, model_cfgs, func_cfgs, debug=False):
    results = []
    for _ in range(in_cfg['n_series']):
        lorenz_df, train, test, t_train, t_test = lorenz_wrapper(input_cfg)
        train_pp, test_pp, ss = preprocess(input_cfg, train, test)

        data_in = (train_pp, test_pp, train, test)
        st = time.time()
        res, _, _, _ = evaluate_models(input_cfg, names, model_cfgs, func_cfgs,
                                       in_cfg['n_repeats'], ss, in_cfg['score_type'], data_in, debug=debug)
        print('Evaluation Time: {}'.format(round(time.time() - st, 2)))
        results.append(res)

    consolidate = consolidate_series_summaries(results, names, in_cfg['score_type'])
    summary, data, errors = summary_results(consolidate, in_cfg['score_type'])
    return summary, data, errors


def parallel_repeat_eval(cfgs, parallel, data_in, ss, input_cfg, functions, n_repeats, debug=False):
    train_pp, test_pp, train, test = data_in
    parallel_in = not parallel and not debug
    if parallel:
        executor = Parallel(n_jobs=cpu_count(), backend='multiprocessing')
        tasks = (
            delayed(repeat_evaluate)(train_pp, test_pp, train, test, input_cfg, cfg, functions[i][0], functions[i][1],
                                     ss=ss, n_repeats=n_repeats, parallel=parallel_in) for i, cfg in
            tqdm(enumerate(cfgs)))
        results = executor(tasks)
    else:
        if debug:
            results = [
                repeat_evaluate(train_pp, test_pp, train, test, input_cfg, cfg, functions[i][0], functions[i][1], ss=ss,
                                n_repeats=n_repeats, parallel=parallel_in, debug=debug) for i, cfg in enumerate(cfgs)]
        else:
            results = [repeat_evaluate(train_pp, test_pp, train, test, input_cfg, cfg, functions[i][0], functions[i][1],
                                       ss=ss, n_repeats=n_repeats, parallel=parallel_in) for i, cfg in tqdm(enumerate(cfgs))]
    return results


def ensemble_get_names(gs_cfgs):
    names = []
    for mod_cfg in gs_cfgs:
        name = str([cfg[0] for cfg in mod_cfg])
        names.append(name)
    return names


def gs_get_cfgs(gs_cfg, model_cfg, comb=True):
    keys = list(gs_cfg.keys())
    ranges = list(gs_cfg.values())
    combinations = list(product(*ranges))
    name_cfg = copy(gs_cfg)

    cfgs_gs, names = [], []
    if comb:
        for comb in combinations:
            new_cfg = copy(model_cfg)
            for k, key in enumerate(keys):
                new_cfg[key] = comb[k]
                name_cfg[key] = comb[k]
            cfgs_gs.append(new_cfg)
            names.append(str(name_cfg))
    else:
        for i in range(len(gs_cfg[keys[0]])):
            new_cfg = copy(model_cfg)
            for key in keys:
                new_cfg[key] = gs_cfg[key][i]
                name_cfg[key] = gs_cfg[key][i]
            cfgs_gs.append(new_cfg)
            names.append(str(name_cfg))

    return cfgs_gs, names


def ensemble_search(input_cfg, gs_cfg, model_cfg, function, n_repeats, score_type,
                    ss, data_in, less_is_better=False, comb=True, debug=False):
    names = ensemble_get_names(gs_cfg)
    parallel = len(gs_cfg) > n_repeats and not debug
    functions = [function for _ in range(len(gs_cfg))]
    results = parallel_repeat_eval(gs_cfg, parallel, data_in, ss, input_cfg, functions, n_repeats, debug=debug)
    consolidate = consolidate_summary(results, names, score_type)
    summary, data, errors = summary_results(consolidate, score_type, less_is_better)
    return summary, data, errors


def grid_search(input_cfg, gs_cfg, model_cfg, function, n_repeats, score_type,
                ss, data_in, less_is_better=False, comb=True, debug=False, ensemble=False):
    cfgs_gs, names = (gs_cfg, ensemble_get_names(gs_cfg)) if ensemble else gs_get_cfgs(gs_cfg, model_cfg, comb)
    parallel = len(cfgs_gs) > n_repeats and not debug
    functions = [function for _ in range(len(cfgs_gs))]
    results = parallel_repeat_eval(cfgs_gs, parallel, data_in, ss, input_cfg, functions, n_repeats, debug=debug)
    consolidate = consolidate_summary(results, names, score_type)
    summary, data, errors = summary_results(consolidate, score_type, less_is_better)
    return summary, data, errors


def eval_multi_step_forecast(name, input_cfg, model_cfg, functions, in_cfg, data_in, ss, label_scale=1,
                             size=(1980, 1080), train_prev_steps=200, plot=True):
    train_pp, test_pp, train, test, t_train, t_test = data_in
    if len(train.shape) > 1:
        if train_pp.shape[1] > 1:
            train_y, test_y = train[:, -1], test[:, -1]
        else:
            train_y, test_y = train, test
    else:
        train_y, test_y = train, test
    image_folder, plot_hist, plot_title, save_results, results_folder, verbose = unpack_in_cfg(in_cfg)

    forecast, _, _, _, _ = walk_forward_step_forecast(train_pp, test_pp, model_cfg, functions[0], functions[1],
                                                      steps=in_cfg['steps'], plot_hist=plot_hist,
                                                      verbose=verbose)

    forecast_reconst = reconstruct(forecast, train, test, input_cfg, in_cfg['steps'], ss=ss)
    metrics = forecast_accuracy(forecast_reconst, test_y)

    df = multi_step_forecast_df(train_y, test_y, forecast_reconst, t_train, t_test, train_prev_steps=200)
    if plot:
        suffix = get_suffix(input_cfg, in_cfg['steps'])
        model_title = {'n_steps_out': model_cfg[0][2]['n_steps_out']} if isinstance(model_cfg, list) else model_cfg
        plotly_time_series(df,
                           title="SERIES: " + str(input_cfg) + '<br>' + name + ': ' + str(model_title) +
                                 '<br>RES: ' + str(metrics), markers='lines', plot_title=plot_title,
                           file_path=[image_folder, name + "_" + suffix], save=save_results, size=size, label_scale=label_scale)
    print(metrics)
    return metrics, df


def view_multi_step_forecasts(names, input_cfg, model_cfgs, func_cfgs, in_cfg, data_in, ss, alphas):
    train_pp, test_pp, train, test, t_train, t_test = data_in
    image_folder, plot_hist, plot_title, save_results, results_folder, verbose = unpack_in_cfg(in_cfg)
    forecasts = []
    metrics_res = []
    for name, model_cfg, func_cfg in zip(names, model_cfgs, func_cfgs):
        forecast, _, _, _, _ = walk_forward_step_forecast(train_pp, test_pp, model_cfg, func_cfg[0], func_cfg[1],
                                                          steps=in_cfg['steps'], plot_hist=plot_hist,
                                                          verbose=verbose)
        forecast_reconst = reconstruct(forecast, train, test, input_cfg, in_cfg['steps'], ss=ss)
        metrics = forecast_accuracy(forecast_reconst, test[:, -1])
        forecasts.append(forecast_reconst)
        metrics_res.append(metrics)

    df = multi_step_forecasts_df(train[:, -1], test[:, -1], names, forecasts, t_train, t_test, train_prev_steps=200)

    result_title = {}
    for name, metric in zip(names, metrics_res):
        result_title[name] = metric[in_cfg['score_type']]
    name = '_'.join(names)

    suffix = get_suffix(input_cfg, in_cfg['steps'])
    plotly_time_series(df, title="SERIES: " + str(input_cfg) +
                                 '<br>' + in_cfg['score_type'].upper() + ': ' + str(result_title) +
                                 '<br>' + 'STEPS OUT: ' + str(in_cfg['steps']),
                       markers='lines', plot_title=plot_title, save=save_results,
                       file_path=[image_folder, name + "_" + suffix], alphas=alphas)
    return metrics_res, df


def run_multi_step_forecast(name, input_cfg, model_cfg, functions, in_cfg, data_in, ss):
    train_pp, test_pp, train, test, t_train, t_test = data_in
    is_mv = ismv(train_pp)
    image_folder, plot_hist, plot_title, save_results, results_folder, verbose = unpack_in_cfg(in_cfg)
    model, _, _ = functions[1](train_pp, model_cfg, verbose=1)
    # history doesn't contain last y column
    history = train_pp[:, :-1] if is_mv else train_pp
    forecast = functions[0](model, history, model_cfg)
    forecast_reconst = reconstruct(forecast, train, test, input_cfg, in_cfg['steps'], ss=ss)
    df = multi_step_forecast_df(train[:, -1] if is_mv else train,
                                test[:in_cfg['steps'], -1] if is_mv else train[:in_cfg['steps']],
                                forecast_reconst, t_train, t_test[:in_cfg['steps']], train_prev_steps=500)
    suffix = get_suffix(input_cfg, in_cfg['steps'])
    plotly_time_series(df, title="SERIES: " + str(input_cfg) + '<br>' + name + ': ' + str(model_cfg),
                       file_path=[image_folder, name + '_SR_' + suffix], plot_title=plot_title, save=save_results)
    return model, forecast_reconst, df


def save_plot_results(names, summary, data, errors, input_cfg, model_cfgs, in_cfg, models_name=None):
    image_folder, plot_hist, plot_title, save_results, results_folder, verbose = unpack_in_cfg(in_cfg)
    #  SAVE RESULTS
    if models_name is None:
        models_info, models_name = models_strings(names, model_cfgs, get_suffix(input_cfg, in_cfg['steps']))
    print(summary)
    save_vars([in_cfg, input_cfg, names, model_cfgs, summary], [results_folder, models_name], save_results)

    cfg = {'n_steps_out': in_cfg['steps'], 'n_series': in_cfg['n_series'], 'n_repeats': in_cfg['n_repeats']}
    plot_bar_summary(data, errors, title="SERIES: " + str(input_cfg) + '<br>' + 'CONFIG: ' + str(cfg),
                     file_path=[image_folder, models_name], plot_title=plot_title, showlegend=False,
                     save=save_results, n_cols_adj_range=data.shape[1])

# if __name__ == '__main__':
#     lorenz_df, train, test, t_train, t_test = univariate_lorenz()
#     cfg = (1, 1, 'persist')
#     scores = repeat_evaluate(train, test, cfg, simple_forecast, simple_fit)
#     summarize_scores('persistence', scores)