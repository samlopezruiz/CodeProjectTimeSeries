import os

import joblib
import numpy as np
import seaborn as sns

from timeseries.experiments.market.plot.plot import plot_2D_moo_dual_results, plot_2D_moo_results_equal_w, \
    plot_2D_dual_moo_results_equal_w
from timeseries.experiments.market.utils.filename import get_result_folder
from timeseries.experiments.market.utils.results import compile_multiple_results_q

sns.set_theme('poster')

if __name__ == "__main__":
    # %%
    general_cfg = {'test_name': 'vary_quantiles',
                   'save_plot': False,
                   'plot_selected': False,
                   'plot_title': False,
                   'plot_tolerance': True
                   }

    results_cfg = {'formatter': 'snp'}

    weights_files = [
        ('60t_ema_q159', 'TFTModel_ES_ema_r_q159_NSGA2_g100_p100_s0_c1_eq0_dual_wmoo'),
        ('60t_ema_q258', 'TFTModel_ES_ema_r_q258_NSGA2_g100_p100_s0_c1_eq0_dual_wmoo'),
        ('60t_ema_q357', 'TFTModel_ES_ema_r_q357_NSGA2_g100_p100_s0_c1_eq0_dual_wmoo'),
    ]

    bounds = ['lq', 'uq']
    results_folder = get_result_folder(results_cfg)
    moo_results = [joblib.load(os.path.join(results_folder, file[0], 'moo', file[1]) + '.z') for file in weights_files]
    # legend_labels_suffix = ['s: {}, g:{}, p:{}'.format(int(moo_result['algo_cfg']['use_sampling']),
    #                                                    moo_result['algo_cfg']['termination'][1],
    #                                                    moo_result['algo_cfg']['pop_size']) for moo_result in moo_results]

    # legend_labels_suffix = ['q: {}'.format(moo_result['quantiles']) for moo_result in moo_results]
    # legend_labels = ['Algorithm: {}'.format(moo_result['lq']['moo_method']) for moo_result in moo_results]
    # experiment_labels = [experiment['lq']['moo_method'] for experiment in moo_results]
    experiment_labels = [
        'q: {}'.format('-'.join((np.array(experiment['lq']['quantiles']) * 10).astype(int).astype(str))) for experiment
        in moo_results]
    results = compile_multiple_results_q(moo_results, experiment_labels=experiment_labels)

    # %%
    xaxis_limit = 1.
    add_risk = 0.05
    qcr_risk_selected = [0.2, 0.2]
    risk_selected = [1.1, 1.1]

    quantiles_losses, eq_quantiles_losses, original_ixs, selected_ixs = [], [], [], []
    for i, bound in enumerate(bounds):
        x_masks = [moo_result[bound]['F'][:, 0] < xaxis_limit for moo_result in moo_results]
        quantiles_losses.append([moo_result[bound]['F'] for moo_result in moo_results])
        eq_quantiles_losses.append([moo_result[bound]['eq_F'] for moo_result in moo_results])
        original_ixs.append([np.argmin(np.sum(np.abs(moo_result[bound]['F'] - moo_result[bound]['original_losses']),
                                              axis=1)) for moo_result in moo_results])

        selected_ixs.append([np.argmin(np.abs(moo_result[bound]['eq_F'][:, 0] - qcr_risk_selected[i]))
                             for x_mask, moo_result in zip(x_masks, moo_results)])
        # selected_ixs.append([np.argmin(np.abs(np.sum(moo_result[bound]['eq_F'], axis=1)[x_mask] - risk_selected[i]))
        #                      for x_mask, moo_result in zip(x_masks, moo_results)])

    filename = '{}_{}_moo_results'.format(general_cfg['test_name'],
                                          moo_results[0]['lq']['experiment_cfg']['vars_definition'])

    # %%
    plot_2D_moo_dual_results(quantiles_losses,
                             save=general_cfg['save_plot'],
                             file_path=os.path.join(os.path.dirname(results_folder),
                                                    'img',
                                                    filename),
                             original_ixs=original_ixs,
                             selected_ixs=selected_ixs if general_cfg['plot_selected'] else None,
                             col_titles=['lower quantile', 'upper quantile'],
                             legend_labels=experiment_labels,
                             figsize=(15, 15),
                             add_risk=add_risk if general_cfg['plot_tolerance'] else None,
                             plot_title=general_cfg['plot_title'],
                             title='Multi objective optimization results',
                             xaxis_limit=xaxis_limit,
                             markersize=5)

    for i, bound in enumerate(['lower quantile', 'upper quantile']):
        risk_lbl = '_eqcr{}'.format(int(qcr_risk_selected[i] * 10)) if general_cfg['plot_selected'] else ''
        plot_2D_moo_results_equal_w(quantiles_losses[i], eq_quantiles_losses[i],
                                    save=general_cfg['save_plot'],
                                    file_path=os.path.join(os.path.dirname(results_folder),
                                                           'img',
                                                           '{}_{}{}'.format(filename,
                                                                            bounds[i],
                                                                            risk_lbl)),
                                    original_ixs=original_ixs[i],
                                    selected_ixs=selected_ixs[i] if general_cfg['plot_selected'] else None,
                                    legend_labels=experiment_labels,
                                    figsize=(15, 15),
                                    add_risk=add_risk if general_cfg['plot_tolerance'] else None,
                                    plot_title=general_cfg['plot_title'],
                                    title='Multi objective optimization results for {}'.format(bound),
                                    xaxis_limit=xaxis_limit,
                                    markersize=5)
    #%%
    plot_2D_dual_moo_results_equal_w(quantiles_losses, eq_quantiles_losses,
                                     save=True, #general_cfg['save_plot'],
                                     file_path=os.path.join(os.path.dirname(results_folder),
                                                            'img',
                                                            '{}_{}_eqW'.format(filename,
                                                                             risk_lbl)),
                                     original_ixs=original_ixs,
                                     selected_ixs=selected_ixs if general_cfg['plot_selected'] else None,
                                     legend_labels=experiment_labels,
                                     figsize=(15, 7),
                                     add_risk=add_risk if general_cfg['plot_tolerance'] else None,
                                     plot_title=general_cfg['plot_title'],
                                     title='Multi objective optimization results',
                                     xaxis_limit=0.8,
                                     markersize=5)

    print(np.array(selected_ixs))
