import os

import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from algorithms.compare.winners import Winners
from timeseries.experiments.market.plot.plot import plot_2D_moo_dual_results, plot_2D_moo_results_equal_w
from timeseries.experiments.market.utils.filename import get_result_folder
from timeseries.experiments.market.utils.results import compile_multiple_results_q, compile_multiple_results, \
    get_hv_results_from_runs
from timeseries.experiments.utils.files import save_vars
from timeseries.utils.utils import array_from_lists, mean_std_from_array, write_latex_from_scores, write_text_file, \
    latex_table

sns.set_theme('poster')


def get_average_of_runs(F_runs, x_lim, labels, step_size=0.01):
    right_intervals = np.arange(0, x_lim, step_size)
    left_intervals = np.roll(right_intervals, 1)[1:]
    right_intervals = right_intervals[1:]
    x = (left_intervals + right_intervals) / 2

    average_run = {}

    for i, F_run in enumerate(F_runs):
        average = []
        for l, r in zip(left_intervals, right_intervals):
            mean_interval = []
            for F in F_run:
                mask = np.logical_and(l <= F[:, 0], F[:, 0] < r)
                f = F[mask, 1]
                if len(f) > 0:
                    mean_interval.append(np.mean(f))
            if len(mean_interval) > 0:
                average.append(np.mean(mean_interval))
            else:
                average.append(np.nan)
        average_run[labels[i]] = average

    return average_run, x


if __name__ == "__main__":
    # %%
    general_cfg = {'comparison_name': 'multiple_training',
                   'save_results': True,
                   'save_plot': True,
                   'plot_title': False,
                   }

    results_cfg = {'formatter': 'snp'}

    weights_files = [
        ('60t_ema_q159', 'TFTModel_ES_ema_r_q159_NSGA2_g100_p100_s0_c1_eq1_dual_wmoo_repeat3'),
        ('60t_ema_q159_1', 'TFTModel_ES_ema_r_q159_NSGA2_g100_p100_s0_c1_eq1_dual_wmoo_repeat3'),
        ('60t_ema_q159_2', 'TFTModel_ES_ema_r_q159_NSGA2_g100_p100_s0_c1_eq1_dual_wmoo_repeat3'),
        ('60t_ema_q159_3', 'TFTModel_ES_ema_r_q159_NSGA2_g100_p100_s0_c1_eq1_dual_wmoo_repeat3'),
        ('60t_ema_q159_4', 'TFTModel_ES_ema_r_q159_NSGA2_g100_p100_s0_c1_eq1_dual_wmoo_repeat3'),
        ('60t_ema_q258', 'TFTModel_ES_ema_r_q258_NSGA2_g100_p100_s0_c1_eq1_dual_wmoo_repeat3'),
        ('60t_ema_q258_1', 'TFTModel_ES_ema_r_q258_NSGA2_g100_p100_s0_c1_eq1_dual_wmoo_repeat3'),
        ('60t_ema_q258_2', 'TFTModel_ES_ema_r_q258_NSGA2_g100_p100_s0_c1_eq1_dual_wmoo_repeat3'),
        ('60t_ema_q258_3', 'TFTModel_ES_ema_r_q258_NSGA2_g100_p100_s0_c1_eq1_dual_wmoo_repeat3'),
        ('60t_ema_q258_4', 'TFTModel_ES_ema_r_q258_NSGA2_g100_p100_s0_c1_eq1_dual_wmoo_repeat3'),
        ('60t_ema_q357', 'TFTModel_ES_ema_r_q357_NSGA2_g100_p100_s0_c1_eq1_dual_wmoo_repeat3'),
        ('60t_ema_q357_1', 'TFTModel_ES_ema_r_q357_NSGA2_g100_p100_s0_c1_eq1_dual_wmoo_repeat3'),
        ('60t_ema_q357_2', 'TFTModel_ES_ema_r_q357_NSGA2_g100_p100_s0_c1_eq1_dual_wmoo_repeat3'),
        ('60t_ema_q357_3', 'TFTModel_ES_ema_r_q357_NSGA2_g100_p100_s0_c1_eq1_dual_wmoo_repeat3'),
        ('60t_ema_q357_4', 'TFTModel_ES_ema_r_q357_NSGA2_g100_p100_s0_c1_eq1_dual_wmoo_repeat3'),
    ]

    bounds = ['lq', 'uq']
    results_folder = os.path.join(get_result_folder(results_cfg), 'compare', general_cfg['comparison_name'])
    moo_results = [joblib.load(os.path.join(get_result_folder(results_cfg), file[0], 'moo', file[1]) + '.z') for file in weights_files]

    # experiment_labels = [
    #     'q: {}'.format('-'.join((np.array(experiment[0]['lq']['quantiles']) * 10).astype(int).astype(str)))
    #     for experiment in moo_results]
    experiment_labels = [weights_file[0][8:] for weights_file in weights_files]
    results = compile_multiple_results(moo_results, experiment_labels,  hv_ref=[10] * 2)

    # %% merge if necesary
    experiment_labels = ['q159', 'q258', 'q357']
    merge_keys = [[key for key in results[list(results.keys())[0]]['risks'].keys() if ss in key] for ss in experiment_labels]

    merged_results = {}
    for q_lbl, q_res in results.items():
        merged_results[q_lbl] = {}
        for metric_lbl, metric_res in q_res.items():
            metric = {}
            for keys, ss in zip(merge_keys, experiment_labels):
                merged_metric = []
                for key in keys:
                    merged_metric += metric_res[key]

                metric[ss] = merged_metric

            merged_results[q_lbl][metric_lbl] = metric

    results = merged_results
    # %%
    q_F_runs = []
    for q_lbl, q_res in results.items():
        F_runs = [np.stack(exp_res) for exp_lbl, exp_res in q_res['risks'].items()]
        q_F_runs.append(F_runs)

    # %%
    x_lim = 1
    average_q_F = {}
    for i, F_runs in enumerate(q_F_runs):
        average_run, x = get_average_of_runs(F_runs, x_lim, experiment_labels, step_size=0.01)
        average_q_F[list(results.keys())[i]] = average_run

    quantiles_losses = [[np.vstack([x, F]).T for _, F in q_F.items()] for _, q_F in average_q_F.items()]

    plot_2D_moo_dual_results(quantiles_losses,
                             save=general_cfg['save_plot'],
                             file_path=os.path.join(results_folder,
                                                    'img',
                                                    '{}_plot'.format(general_cfg['comparison_name'])),

                             col_titles=['lower quantile', 'upper quantile'],
                             legend_labels=experiment_labels,
                             figsize=(15, 15),
                             plot_title=general_cfg['plot_title'],
                             title='Multi objective optimization results',
                             markersize=5)

    # %% HV results
    q_exp_hvs, hvs_df = get_hv_results_from_runs(results, experiment_labels)

    # %% Winners
    metric = np.negative(np.mean(q_exp_hvs, axis=2))
    winners = Winners(metric, experiment_labels)
    scores = winners.score(q_exp_hvs, alternative='greater')

    #%%
    if general_cfg['save_results']:
        results['results_cfg'] = results_cfg
        results['weights_files'] = weights_files
        save_vars(results, os.path.join(results_folder,
                                        '{}'.format(general_cfg['comparison_name'])))

        write_latex_from_scores(scores,
                                os.path.join(results_folder,
                                             'txt',
                                             '{}_scores'.format(general_cfg['comparison_name'])))

        write_text_file(os.path.join(results_folder,
                                     'txt',
                                     '{}'.format(general_cfg['comparison_name'])),
                        latex_table('Hypervolume for quantiles', hvs_df.to_latex()))