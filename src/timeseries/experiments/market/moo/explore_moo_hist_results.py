import os

import joblib
import numpy as np
import seaborn as sns

from algorithms.moo.utils.plot import plot_runs
from timeseries.experiments.market.utils.filename import get_result_folder
from timeseries.experiments.market.utils.results import compile_multiple_results
from timeseries.experiments.utils.files import save_vars

sns.set_theme('poster')
sns.set_style("dark")

if __name__ == "__main__":
    # %%
    general_cfg = {'save_plot': True,
                   'test_name': 'moo_methods',
                   'show_title': True,
                   'comparison_name': 'moo_methods',
                   'save_results': True}

    results_cfg = {'formatter': 'snp',
                   'experiment_name': '60t_ema_q159'}

    weights_files = ['TFTModel_ES_ema_r_q159_NSGA2_g10_p100_s1_dual_wmoo_repeat2',
                     'TFTModel_ES_ema_r_q159_NSGA3_g10_p100_s1_dual_wmoo_repeat2']

    results_folder = os.path.join(get_result_folder(results_cfg), 'compare')
    moo_results = [joblib.load(os.path.join(get_result_folder(results_cfg), 'moo', file) + '.z') for file in
                   weights_files]

    experiment_labels = [experiment[0]['lq']['moo_method'] for experiment in moo_results]
    results = compile_multiple_results(moo_results, experiment_labels, hv_ref=[10] * 2)

    # %%

    for q_lbl, q_res in results.items():
        for exp_lbl, exp_res in q_res['hv_hist'].items():
            y_runs = np.array(exp_res)

            filename = '{}_{}_runs'.format(exp_lbl, q_lbl.replace(" ", "_"))
            plot_runs(y_runs,
                      mean_run=np.mean(exp_res, axis=0),
                      x_label='Generation',
                      y_label='Hypervolume',
                      title='{} HV history for {}'.format(exp_lbl, q_lbl),
                      size=(15, 9),
                      file_path=os.path.join(results_folder, 'img', filename),
                      save=general_cfg['save_plot'],
                      legend_labels=None,
                      show_grid=True,
                      show_title=general_cfg['show_title'])

    # %%
    for q_lbl, q_res in results.items():
        y_runs = []
        for exp_lbl, exp_res in q_res['hv_hist'].items():
            y_runs.append(np.mean(exp_res, axis=0))

        filename = '{}_{}_comparison'.format(general_cfg['comparison_name'], q_lbl.replace(" ", "_"))
        exp_runs = np.array(y_runs)
        plot_runs(exp_runs,
                  mean_run=None,
                  x_label='Generation',
                  y_label='Hypervolume',
                  title='HV history for {}'.format(q_lbl),
                  size=(15, 9),
                  file_path=os.path.join(results_folder, 'img', filename),
                  save=general_cfg['save_plot'],
                  legend_labels=experiment_labels,
                  show_grid=True,
                  show_title=general_cfg['show_title'])

    # %%
    y_runs, plot_labels = [], []
    for q_lbl, q_res in results.items():
        for exp_lbl, exp_res in q_res['hv_hist'].items():
            y_runs.append(np.mean(exp_res, axis=0))
            plot_labels.append('{} for {}'.format(exp_lbl, q_lbl))

    filename = '{}_comparison'.format(general_cfg['comparison_name'])
    exp_runs = np.array(y_runs)
    plot_runs(exp_runs,
              mean_run=None,
              x_label='Generation',
              y_label='Hypervolume',
              title='HV history for {}'.format(q_lbl),
              size=(15, 9),
              file_path=os.path.join(results_folder, 'img', filename),
              save=general_cfg['save_plot'],
              legend_labels=plot_labels,
              show_grid=True,
              show_title=general_cfg['show_title'])

    if general_cfg['save_results']:
        results['results_cfg'] = results_cfg
        results['weights_files'] = weights_files
        save_vars(results, os.path.join(results_folder,
                                        '{}'.format(general_cfg['comparison_name'])))
