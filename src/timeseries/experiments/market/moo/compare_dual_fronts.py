import os

import joblib
import numpy as np
import seaborn as sns

from timeseries.experiments.market.plot.plot import plot_2D_moo_dual_results
from timeseries.experiments.market.utils.filename import get_result_folder

sns.set_theme('poster')

if __name__ == "__main__":
    # %%
    general_cfg = {'save_plot': True,
                   'test_name': 'moo_methods'}

    results_cfg = {'formatter': 'snp'}

    weights_files = [('60t_ema_q159', 'TFTModel_ES_ema_r_q159_NSGA2_g100_p100_s1_dual_wmoo'),
                     ('60t_ema_q159', 'TFTModel_ES_ema_r_q159_MOEAD_g100_p100_s1_dual_wmoo'),
                     ('60t_ema_q159', 'TFTModel_ES_ema_r_q159_NSGA3_g100_p100_s1_dual_wmoo')]

    results_folder = get_result_folder(results_cfg)
    moo_results = [joblib.load(os.path.join(results_folder, file[0], 'moo', file[1]) + '.z') for file in weights_files]
    # legend_labels_suffix = ['s: {}, g:{}, p:{}'.format(int(moo_result['algo_cfg']['use_sampling']),
    #                                                    moo_result['algo_cfg']['termination'][1],
    #                                                    moo_result['algo_cfg']['pop_size']) for moo_result in moo_results]

    # legend_labels_suffix = ['q: {}'.format(moo_result['quantiles']) for moo_result in moo_results]
    legend_labels = ['Algorithm: {}'.format(moo_result['lq']['moo_method']) for moo_result in moo_results]

    # %%
    quantiles_losses, original_ixs = [], []
    for bound in ['lq', 'uq']:
        quantiles_losses.append([moo_result[bound]['F'] for moo_result in moo_results])
        original_ixs.append([np.argmin(np.sum(np.abs(moo_result[bound]['F'] - moo_result[bound]['original_losses']),
                                              axis=1)) for moo_result in moo_results])

    filename = '{}_{}_moo_results'.format(general_cfg['test_name'],
                                          moo_results[0]['lq']['experiment_cfg']['vars_definition'])

    plot_2D_moo_dual_results(quantiles_losses,
                             save=general_cfg['save_plot'],
                             file_path=os.path.join(os.path.dirname(results_folder),
                                                    'img',
                                                    filename),
                             original_ixs=original_ixs,
                             col_titles=['lower quantile', 'upper quantile'],
                             legend_labels=legend_labels,
                             figsize=(15, 15),
                             title='Multi objective optimization results',
                             xaxis_limit=0.6)
