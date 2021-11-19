import os

import joblib
import numpy as np
import seaborn as sns

from timeseries.experiments.market.plot.plot import plot_2D_pareto_front, plot_2D_moo_results
from timeseries.experiments.market.utils.filename import get_output_folder, get_result_folder

sns.set_theme('poster')

if __name__ == "__main__":
    # %%
    general_cfg = {'save_plot': True,
                   'test_name': 'moo_methods'}

    results_cfg = {'formatter': 'snp'}

    weights_files = [('60t_ema_q159', 'TFTModel_ES_ema_r_q159_NSGA2_g100_p100_s1_dual_wmoo_1'),
                     ('60t_ema_q159', 'TFTModel_ES_ema_r_q159_NSGA3_g100_p100_s1_dual_wmoo')]

    results_folder = get_result_folder(results_cfg)
    moo_results = [joblib.load(os.path.join(results_folder, file[0], 'moo', file[1]) + '.z') for file in weights_files]
    # legend_labels_suffix = ['s: {}, g:{}, p:{}'.format(int(moo_result['algo_cfg']['use_sampling']),
    #                                                    moo_result['algo_cfg']['termination'][1],
    #                                                    moo_result['algo_cfg']['pop_size']) for moo_result in moo_results]

    # legend_labels_suffix = ['q: {}'.format(moo_result['quantiles']) for moo_result in moo_results]
    legend_labels_suffix = ['Algorithm: {}'.format(moo_result['moo_method']) for moo_result in moo_results]

    quantiles_losses = [moo_result['F'] for moo_result in moo_results]
    eq_quantiles_losses = [moo_result['eq_F'] for moo_result in moo_results]
    original_losses = [moo_result['original_losses'] for moo_result in moo_results]
    original_ixs = [np.argmin(np.sum(np.abs(quantiles_loss - moo_result['original_losses']), axis=1))
                    for quantiles_loss, moo_result in zip(quantiles_losses, moo_results)]

    filename = '{}_{}_moo_results'.format(general_cfg['test_name'],
                                          moo_results[0]['experiment_cfg']['vars_definition'])

    eq_quantiles_losses = None
    plot_2D_moo_results(quantiles_losses, eq_quantiles_losses,
                        save=general_cfg['save_plot'],
                        file_path=os.path.join(os.path.dirname(results_folder),
                                               'img',
                                               filename),
                        original_ixs=original_ixs,
                        legend_labels_suffix=legend_labels_suffix,
                        figsize=(15, 15),
                        title='Multi objective optimization results',
                        xaxis_limit=.5)
