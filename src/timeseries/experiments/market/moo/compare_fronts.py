import os

import joblib
import numpy as np
import seaborn as sns

from timeseries.experiments.market.plot.plot import plot_2D_pareto_front, plot_2D_moo_results
from timeseries.experiments.market.utils.filename import get_output_folder, get_result_folder

sns.set_theme('poster')

if __name__ == "__main__":
    # %%
    general_cfg = {'save_forecast': True,
                   'save_plot': True,
                   'use_all_data': True,
                   'use_moo_weights': True}

    results_cfg = {'formatter': 'snp'}

    weights_files = [('60t_ema_q159', 'ES_ema_r_q159_moo_weights'),
                     ('60t_ema', 'ES_ema_r_q258_moo_weights_1'),
                     ('60t_ema_q357', 'ES_ema_r_q357_moo_weights')]

    results_folder = get_result_folder(results_cfg)
    moo_results = [joblib.load(os.path.join(results_folder, file[0], 'moo', file[1]) + '.z') for file in weights_files]

    quantiles_losses = [moo_result['F'] for moo_result in moo_results]
    eq_quantiles_losses = [moo_result['eq_F'] for moo_result in moo_results]
    original_losses = [moo_result['original_losses'] for moo_result in moo_results]
    original_ixs = [np.argmin(np.sum(np.abs(quantiles_loss - moo_result['original_losses']), axis=1))
                    for quantiles_loss, moo_result in zip(quantiles_losses, moo_results)]

    filename = '{}_pareto_fronts'.format(moo_results[0]['experiment_cfg']['vars_definition'])

    legend_labels_suffix = ['q: {}'.format(moo_result['quantiles']) for moo_result in moo_results]

    plot_2D_moo_results(quantiles_losses, eq_quantiles_losses,
                        save=general_cfg['save_plot'],
                        file_path=os.path.join(os.path.dirname(results_folder),
                                               'img',
                                               filename),
                        original_ixs=original_ixs,
                        legend_labels_suffix=legend_labels_suffix,
                        figsize=(20, 15),
                        title='Multi objective optimization results',
                        xaxis_limit=0.6)
