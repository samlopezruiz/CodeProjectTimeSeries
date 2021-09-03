import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import logging

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


def plot_multiple_results_forecast(all_forecast_df, forecast_dfs, use_regimes, results, max_subplots=15, n_plots=2):
    if use_regimes:
        plotly_time_series_bars_hist(all_forecast_df, features=['data', 'forecast', 'rse', 'state'],
                                     color_col='state', bars_cols=['rse'])
        results_state = results_by_state(all_forecast_df).round(decimals=3)

        n_states = len(pd.unique(all_forecast_df['state']))
        subsets_state = []
        for i in range(n_states):
            subsets_state.append(list(results.loc[results['reg_round'] == i, 'reg_round'].index))

        for r, subsets in enumerate(subsets_state):
            chosen = sorted(np.random.choice(subsets, size=min(max_subplots, len(subsets))))
            dfs = [forecast_dfs[i] for i in chosen]
            title = 'Regime {}: {}'.format(r, str(results_state.iloc[r, :].to_dict()))
            plotly_multiple(dfs, features=['data', 'forecast'], title=title)
    else:
        chosen = sorted(np.random.choice(list(range(len(results))), size=min(max_subplots * n_plots, len(results))))
        for i in range(n_plots):
            chosen_plot = chosen[i * max_subplots:i * max_subplots + max_subplots]
            dfs = [forecast_dfs[i] for i in chosen_plot]
            title = 'Randomly selected forecasts'
            plotly_multiple(dfs, features=['data', 'forecast'], title=title)


def confusion_mat(all_forecast_df, plot_=True):
    all_forecast_df['up_down'] = all_forecast_df['data'] > all_forecast_df['data'].shift(1)
    all_forecast_df['up_down_pred'] = all_forecast_df['forecast'] > all_forecast_df['data'].shift(1)
    all_forecast_df['hit_rate'] = all_forecast_df['up_down'] == all_forecast_df['up_down_pred']
    cm = confusion_matrix(all_forecast_df['up_down'], all_forecast_df['up_down_pred'], normalize='all')
    tn, fp, fn, tp = cm.ravel()
    if plot_:
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)  # , display_labels = clf.classes_)
        disp.plot()
        plt.show()

    return cm, {'tn': round(tn, 4), 'fp': round(fp, 4), 'fn': round(fn, 4), 'tp': round(tp, 4)}

