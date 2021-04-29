import time
from timeseries.models.lorenz.functions.functions import walk_forward_step_forecast
from timeseries.models.lorenz.functions.preprocessing import reconstruct
from timeseries.models.lorenz.functions.summarize import consolidate_summary
from timeseries.models.utils.config import unpack_in_cfg
from timeseries.models.utils.forecast import multi_step_forecast_df
from timeseries.models.utils.metrics import forecast_accuracy, summary_results
from timeseries.models.utils.models import models_strings, save_vars, get_suffix
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
                    walkforward=walk_forward_step_forecast, verbose=0, parallel=True, debug=False):
    results = parallel_walkforward(cfg, model_fit, model_forecast, n_repeats, parallel, test_pp, train_pp, verbose,
                                   walkforward, debug=debug)

    metrics, predictions, times, n_params, loss = [], [], [], 0, []
    for forecast, train_t, pred_t, n_params, train_loss in results:
        forecast_reconst = reconstruct(forecast, train, test, input_cfg, cfg, ss=ss)
        metric = forecast_accuracy(forecast_reconst, test[:, -1] if input_cfg['variate'] == 'multi' else test)
        metrics.append(metric)
        predictions.append(forecast)
        times.append((train_t, pred_t))
        loss.append(train_loss)
    return metrics, predictions, times, n_params, loss


def parallel_walkforward(cfg, model_fit, model_forecast, n_repeats, parallel, test_pp,
                         train_pp, verbose, walkforward, debug=False):
    if parallel:
        executor = Parallel(n_jobs=cpu_count(), backend='multiprocessing')
        tasks = (delayed(walkforward)(train_pp, test_pp, cfg, model_forecast,
                                      model_fit, steps=cfg.get('n_steps_out', 1), verbose=verbose) for _ in
                 tqdm(range(n_repeats)))
        results = executor(tasks)
    else:
        if debug:
            results = [walkforward(train_pp, test_pp, cfg, model_forecast, model_fit,
                                   steps=cfg.get('n_steps_out', 1), verbose=verbose) for _ in range(n_repeats)]
        else:
            results = [walkforward(train_pp, test_pp, cfg, model_forecast, model_fit,
                                   steps=cfg.get('n_steps_out', 1), verbose=verbose) for _ in tqdm(range(n_repeats))]
    return results


def evaluate_models(input_cfg, names, model_cfgs, functions, n_repeats, ss, score_type, data_in, debug=False):
    parallel = len(model_cfgs) > n_repeats and not debug
    results = parallel_repeat_eval(model_cfgs, parallel, data_in, ss, input_cfg, functions, n_repeats, debug=debug)
    consolidate = consolidate_summary(results, names, score_type)
    summary, data, errors = summary_results(consolidate, score_type)
    return summary, data, errors


def parallel_repeat_eval(cfgs, parallel, data_in, ss, input_cfg, functions, n_repeats, debug=False):
    train_pp, test_pp, train, test = data_in
    parallel_in = not parallel and not debug
    if parallel:
        executor = Parallel(n_jobs=cpu_count(), backend='multiprocessing')
        tasks = (
        delayed(repeat_evaluate)(train_pp, test_pp, train, test, input_cfg, cfg, functions[i][0], functions[i][1],
                                 ss=ss, n_repeats=n_repeats, parallel=parallel_in) for i, cfg in tqdm(enumerate(cfgs)))
        results = executor(tasks)
    else:
        if debug:
            results = [
                repeat_evaluate(train_pp, test_pp, train, test, input_cfg, cfg, functions[i][0], functions[i][1], ss=ss,
                                n_repeats=n_repeats, parallel=parallel_in, debug=debug) for i, cfg in enumerate(cfgs)]
        else:
            results = [repeat_evaluate(train_pp, test_pp, train, test, input_cfg, cfg, functions[i][0], functions[i][1],
                                       ss=ss, n_repeats=n_repeats, parallel=parallel_in) for i, cfg in
                       tqdm(enumerate(cfgs))]
    return results


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


def grid_search(input_cfg, gs_cfg, model_cfg, function, n_repeats, score_type,
                ss, data_in, less_is_better=False, comb=True, debug=False):
    cfgs_gs, names = gs_get_cfgs(gs_cfg, model_cfg, comb)
    parallel = len(cfgs_gs) > n_repeats and not debug
    functions = [function for _ in range(len(cfgs_gs))]
    results = parallel_repeat_eval(cfgs_gs, parallel, data_in, ss, input_cfg, functions, n_repeats, debug=debug)
    consolidate = consolidate_summary(results, names, score_type)
    summary, data, errors = summary_results(consolidate, score_type, less_is_better)
    return summary, data, errors


def eval_multi_step_forecast(name, input_cfg, model_cfg, functions, in_cfg, data_in, ss):
    train_pp, test_pp, train, test, t_train, t_test = data_in
    image_folder, plot_hist, plot_title, save_results, results_folder, verbose = unpack_in_cfg(in_cfg)

    forecast, _, _, _, _ = walk_forward_step_forecast(train_pp, test_pp, model_cfg, functions[0], functions[1],
                                                      steps=model_cfg['n_steps_out'], plot_hist=plot_hist,
                                                      verbose=verbose)

    forecast_reconst = reconstruct(forecast, train, test, input_cfg, model_cfg, ss=ss)
    metrics = forecast_accuracy(forecast_reconst, test[:, -1])

    df = multi_step_forecast_df(train[:, -1], test[:, -1], forecast_reconst, t_train, t_test, train_prev_steps=200)
    suffix = get_suffix(input_cfg, model_cfg)
    plotly_time_series(df,
                       title="SERIES: " + str(input_cfg) + '<br>' + name + ': ' + str(
                           model_cfg) + '<br>RES: ' + str(
                           metrics),
                       markers='lines',
                       file_path=[image_folder, name + "_" + suffix], plot_title=plot_title, save=save_results)
    print(metrics)
    return metrics, df


def run_multi_step_forecast(name, input_cfg, model_cfg, functions, in_cfg, data_in, ss):
    train_pp, test_pp, train, test, t_train, t_test = data_in
    image_folder, plot_hist, plot_title, save_results, results_folder, verbose = unpack_in_cfg(in_cfg)
    model, _, _ = functions[1](train_pp, model_cfg, verbose=1)
    # history doesn't contain last y column
    history = train_pp[:, :-1]
    forecast = functions[0](model, history, model_cfg)
    forecast_reconst = reconstruct(forecast, train, test, input_cfg, model_cfg, ss=ss)
    df = multi_step_forecast_df(train[:, -1], test[:model_cfg['n_steps_out'], -1], forecast_reconst, t_train,
                              t_test[:model_cfg['n_steps_out']], train_prev_steps=500)
    suffix = get_suffix(input_cfg, model_cfg)
    plotly_time_series(df, title="SERIES: " + str(input_cfg) + '<br>' + name + ': ' + str(model_cfg),
                       file_path=[image_folder, name+'_SR_'+suffix], plot_title=plot_title, save=save_results)
    return model, forecast_reconst, df


def save_plot_results(names, summary, data, errors, input_cfg, model_cfgs, in_cfg):
    image_folder, plot_hist, plot_title, save_results, results_folder, verbose = unpack_in_cfg(in_cfg)
    #  SAVE RESULTS
    models_info, models_name = models_strings(names, model_cfgs, get_suffix(input_cfg, model_cfgs[0]))
    print(summary)
    save_vars([input_cfg, model_cfgs, summary], [results_folder, models_name], save_results)

    #
    plot_bar_summary(data, errors, title="SERIES: " + str(input_cfg) + '<br>' + 'STEPS OUT: ' + str(in_cfg['steps']),
                     file_path=[image_folder, models_name], plot_title=plot_title, showlegend=False,
                     save=save_results, n_cols_adj_range=data.shape[1])

# if __name__ == '__main__':
#     lorenz_df, train, test, t_train, t_test = univariate_lorenz()
#     cfg = (1, 1, 'persist')
#     scores = repeat_evaluate(train, test, cfg, simple_forecast, simple_fit)
#     summarize_scores('persistence', scores)
