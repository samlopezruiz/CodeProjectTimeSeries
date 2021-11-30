import os

import joblib
import numpy as np
import seaborn as sns

from timeseries.experiments.market.plot.plot import plot_2D_moo_results_equal_w
from timeseries.experiments.market.utils.filename import get_result_folder

sns.set_theme('poster')

if __name__ == "__main__":
    # %%
    general_cfg = {'save_plot': False,
                   'test_name': 'moo_methods'}

    results_cfg = {'formatter': 'snp'}

    weights_files = [
        ('60t_ema_q357', 'TFTModel_ES_ema_r_q357_NSGA3_g250_p250_s1_k2_wmoo'),
        ('60t_ema_q258', 'TFTModel_ES_ema_r_q258_NSGA3_g250_p250_s1_k2_wmoo'),
        ('60t_ema_q159', 'TFTModel_ES_ema_r_q159_NSGA3_g250_p250_s1_k2_wmoo'),
    ]

    results_folder = get_result_folder(results_cfg)
    moo_results = [joblib.load(os.path.join(results_folder, file[0], 'moo', 'old', file[1]) + '.z') for file in
                   weights_files]
    # legend_labels_suffix = ['s: {}, g:{}, p:{}'.format(int(moo_result['algo_cfg']['use_sampling']),
    #                                                    moo_result['algo_cfg']['termination'][1],
    #                                                    moo_result['algo_cfg']['pop_size']) for moo_result in moo_results]

    legend_label = ['q: {}'.format(moo_result['quantiles']) for moo_result in moo_results]
    # legend_labels_suffix = ['{}'.format(moo_result['moo_method']) for moo_result in moo_results]

    quantiles_losses = [moo_result['F'] for moo_result in moo_results]
    eq_quantiles_losses = [moo_result['eq_F'] for moo_result in moo_results]
    original_losses = [moo_result['original_losses'] for moo_result in moo_results]
    original_ixs = [np.argmin(np.sum(np.abs(quantiles_loss - moo_result['original_losses']), axis=1))
                    for quantiles_loss, moo_result in zip(quantiles_losses, moo_results)]

    filename = '{}_{}_moo_results'.format(general_cfg['test_name'],
                                          moo_results[0]['experiment_cfg']['vars_definition'])

    # eq_quantiles_losses = None
    plot_2D_moo_results_equal_w(quantiles_losses, eq_quantiles_losses,
                                save=general_cfg['save_plot'],
                                file_path=os.path.join(os.path.dirname(results_folder),
                                                       'img',
                                                       filename),
                                original_ixs=original_ixs,
                                legend_labels=legend_label,
                                figsize=(20, 15),
                                title='Multi objective optimization results',
                                xaxis_limit=.5)
