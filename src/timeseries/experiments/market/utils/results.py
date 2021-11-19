from copy import copy

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import logging

from algorithms.moo.utils.indicators import get_hypervolume
from timeseries.experiments.market.plot_forecasts import group_forecasts
from timeseries.experiments.market.utils.preprocessing import reconstruct_forecasts
from timeseries.plotly.plot import plotly_time_series, plotly_time_series_bars_hist, plotly_multiple


def subset_results(metrics, metric_names=['rmse', 'minmax']):
    results = np.zeros((len(metrics), len(metric_names)))
    for i, metric in enumerate(metrics):
        for m, name in enumerate(metric_names):
            results[i][m] = metrics[i][name]

    return pd.DataFrame(results, columns=metric_names)


def get_results(metrics, model_cfg, test_x_pp, metric_names=['rmse', 'minmax'], plot_=True, print_=True):
    results = subset_results(metrics, metric_names=metric_names)
    if model_cfg['use_regimes']:
        results['regime'] = [np.mean(np.argmax(prob.to_numpy(), axis=1)) for x, prob in test_x_pp]
    if print_:
        for m in metric_names:
            print('Test {}: {} +-({})'.format(m, round(np.mean(results[m]), 2),
                                              round(np.std(results[m]), 4)))

    if plot_:
        rows = list(range(3 if model_cfg['use_regimes'] else 2))
        type_plot = ['bar' for _ in range(3 if model_cfg['use_regimes'] else 2)]
        plotly_time_series(results, rows=rows, xaxis_title='test subset', type_plot=type_plot, plot_ytitles=True)
        # plot_corr_df(results)
    if model_cfg['use_regimes']:
        results['reg_round'] = np.round(results['regime'])
    return results

def results_by_state(all_forecast_df):
    groupby_state = all_forecast_df.groupby('state')
    score_states = pd.DataFrame()
    score_states['count'] = groupby_state.count()['rse']
    score_states['mean'] = groupby_state.mean()['rse']
    score_states['std'] = groupby_state.std()['rse']
    score_states['perc'] = score_states['count'] / sum(score_states['count'])
    score_states.index.name = 'state'
    return score_states


def plot_multiple_results_forecast(all_forecast_df, forecast_dfs, use_regimes, results, max_subplots=15, n_plots=2,
                                   save=False, file_path=None):

    if use_regimes:
        file_path0 = copy(file_path)
        file_path0[-1] = file_path0[-1] + '_hist'
        plotly_time_series_bars_hist(all_forecast_df, features=['data', 'forecast', 'rse', 'state'],
                                     color_col='state', bars_cols=['rse'], save=save, file_path=file_path0)
        results_state = results_by_state(all_forecast_df).round(decimals=3)

        n_states = len(pd.unique(all_forecast_df['state']))
        subsets_state = []
        for i in range(n_states):
            subsets_state.append(list(results.loc[results['reg_round'] == i, 'reg_round'].index))

        file_path[-1] = file_path[-1] + '_plt' + str(0)
        for r, subsets in enumerate(subsets_state):
            chosen = sorted(np.random.choice(subsets, size=min(max_subplots, len(subsets))))
            dfs = [forecast_dfs[i] for i in chosen]
            title = 'Regime {}: {}'.format(r, str(results_state.iloc[r, :].to_dict()))
            file_path[-1] = file_path[-1][:-1] + str(r)
            plotly_multiple(dfs, features=['data', 'forecast'], title=title, save=save, file_path=file_path)
    else:
        file_path[-1] = file_path[-1] + '_plt' + str(0)
        chosen = sorted(np.random.choice(list(range(len(results))), size=min(max_subplots * n_plots, len(results))))
        for i in range(n_plots):
            chosen_plot = chosen[i * max_subplots:i * max_subplots + max_subplots]
            dfs = [forecast_dfs[i] for i in chosen_plot]
            title = 'Randomly selected forecasts'
            file_path[-1] = file_path[-1][:-1] + str(i)
            plotly_multiple(dfs, features=['data', 'forecast'], title=title, save=save, file_path=file_path)


def confusion_mat(y_true, y_pred, plot_=True, self_change=False):
    hit_rate = pd.DataFrame()
    hit_rate['up_down'] = y_true > y_true.shift(1)
    if not self_change:
        hit_rate['up_down_pred'] = y_pred > y_true.shift(1)
    else:
        hit_rate['up_down_pred'] = y_pred > y_pred.shift(1)
    hit_rate['hit_rate'] = hit_rate['up_down'] == hit_rate['up_down_pred']
    cm = confusion_matrix(hit_rate['up_down'], hit_rate['up_down_pred'], normalize='all')

    tn, fp, fn, tp = cm.ravel()
    if plot_:
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)  # , display_labels = clf.classes_)
        disp.plot()
        plt.show()

    return cm, {'tn': round(tn, 4), 'fp': round(fp, 4), 'fn': round(fn, 4), 'tp': round(tp, 4)}


def hit_rate_from_forecast(results, n_output_steps, plot_=True):
    target_col = results['target']

    forecasts = results['reconstructed_forecasts'] if results['target'] else results['forecasts']
    if target_col:
        label = '{} t+{}'.format(target_col, 1)
    else:
        label = 't+{}'.format(1)

    cm, cm_metrics = confusion_mat(y_true=forecasts['targets'][label],
                                   y_pred=forecasts['p50'][label],
                                   plot_=plot_)

    grouped = group_forecasts(forecasts, n_output_steps, target_col)

    confusion_mats = []
    for identifier, ss_df in grouped['targets'].items():
        try:
            confusion_mats.append(confusion_mat(y_true=ss_df[label],
                                                y_pred=grouped['p50'][identifier][label],
                                                plot_=False)[0])
        except:
            pass

    confusion_mats = np.stack(confusion_mats)

    return {
        'global_hit_rate': (cm, cm_metrics),
        'grouped_by_id_hit_rate': confusion_mats
    }

def post_process_results(results, formatter, experiment_cfg, plot_=True):
    model_params = formatter.get_default_model_params()
    n_output_steps = model_params['total_time_steps'] - model_params['num_encoder_steps']

    print('Reconstructing forecasts...')
    if results['target']:
        results['reconstructed_forecasts'] = reconstruct_forecasts(formatter, results['forecasts'])
    results['hit_rates'] = hit_rate_from_forecast(results, n_output_steps, plot_=plot_)
    results['experiment_cfg'] = experiment_cfg


def compile_multiple_results(moo_results, experiment_labels, hv_ref=[10] * 2):
    results = {}
    for q_lbl, bound in zip(['lower quantile', 'upper quantile'], ['lq', 'uq']):
        results[q_lbl] = {}
        results[q_lbl]['risks'] = {}
        results[q_lbl]['history'] = {}
        results[q_lbl]['hv_hist'] = {}
        for experiment, exp_lbl in zip(moo_results, experiment_labels):
            results[q_lbl]['risks'][exp_lbl] = [e[bound]['F'] for e in experiment]
            results[q_lbl]['history'][exp_lbl] = [e[bound]['pop_hist'] for e in experiment]
            results[q_lbl]['hv_hist'][exp_lbl] = [[get_hypervolume(F, hv_ref) for F in hist] for hist
                                                  in [e[bound]['pop_hist'] for e in experiment] if hist is not None]

    return results