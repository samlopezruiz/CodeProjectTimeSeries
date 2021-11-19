import os

import joblib
import numpy as np
import seaborn as sns
import tensorflow as tf

from algorithms.tft2.harness.train_test import compute_moo_q_loss
from timeseries.experiments.market.moo.utils.model import get_new_weights
from timeseries.experiments.market.moo.utils.utils import get_selected_ix
from timeseries.experiments.market.plot.plot import plot_2D_moo_dual_results
from timeseries.experiments.market.utils.filename import get_result_folder, quantiles_name, termination_name, risk_name
from timeseries.experiments.market.utils.harness import load_predict_model, get_model_data_config
from timeseries.experiments.market.utils.results import post_process_results
from timeseries.experiments.utils.files import save_vars

sns.set_theme('poster')

if __name__ == "__main__":
    # %%
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    print("Device Name: ", tf.test.gpu_device_name())
    print('TF eager execution: {}'.format(tf.executing_eagerly()))

    general_cfg = {'save_forecast': True,
                   'save_plot': True,
                   'use_all_data': True,
                   'use_moo_weights': False}

    results_cfg = {'formatter': 'snp',
                   'experiment_name': '60t_ema_q159',
                   'results': 'TFTModel_ES_ema_r_q159_NSGA3_g100_p100_s1_dual_wmoo'
                   }

    moo_results = joblib.load(os.path.join(get_result_folder(results_cfg), 'moo', results_cfg['results'] + '.z'))

    config, formatter, model_folder = get_model_data_config(moo_results['lq']['experiment_cfg'],
                                                            moo_results['lq']['model_cfg'],
                                                            moo_results['lq']['fixed_cfg'])

    experiment_cfg = moo_results['lq']['experiment_cfg']

    risk_selected = {'qerl': 0.5,
                     'qcru': 0.5}
    # risk_selected = {'qer-l': 0.5,
    #                  'qer-u': 0.5}

    basename = '{}_{}_q{}_{}_{}_p{}_s{}_{}_'.format(experiment_cfg['architecture'],
                                                    experiment_cfg['vars_definition'],
                                                    quantiles_name(moo_results['lq']['quantiles']),
                                                    moo_results['lq']['moo_method'],
                                                    termination_name(moo_results['lq']['algo_cfg']['termination']),
                                                    moo_results['lq']['algo_cfg']['pop_size'],
                                                    int(moo_results['lq']['algo_cfg']['use_sampling']),
                                                    risk_name(risk_selected) if general_cfg['use_moo_weights'] else '',
                                                    )

    labels, quantiles_losses, original_ixs, selected_ixs = [], [], [], []
    selected_weights, original_weights = {}, {}
    for bound, moo_result in moo_results.items():
        weights, quantiles_loss, eq_quantiles_loss = moo_result['X'], moo_result['F'], moo_result['eq_F']
        original_ix = np.argmin(np.sum(np.abs(quantiles_loss - moo_result['original_losses']), axis=1))
        selected_ix = get_selected_ix(quantiles_loss, risk_selected, upper=bound == 'uq')

        moo_results[bound]['original_ix'] = original_ix
        moo_results[bound]['selected_ix'] = selected_ix
        labels.append('upper quantile' if bound == 'uq' else 'lower quantile')
        quantiles_losses.append(quantiles_loss)
        original_ixs.append(original_ix)
        selected_ixs.append(selected_ix)
        selected_weights[bound] = weights[selected_ix, :]
        original_weights[bound] = weights[original_ix, :]

    # %%
    xaxis_limit = 1.
    img_path = os.path.join(config.results_folder,
                            experiment_cfg['experiment_name'],
                            'img',
                            '{}pf'.format(basename))

    plot_2D_moo_dual_results(quantiles_losses,
                             selected_ix=selected_ixs if general_cfg['use_moo_weights'] else None,
                             save=general_cfg['save_plot'],
                             file_path=img_path,
                             original_ixs=original_ixs,
                             figsize=(20, 15),
                             xaxis_limit=xaxis_limit,
                             col_titles=labels,
                             legend_labels=None,
                             title='MOO using {} for quantiles: {}'.format(moo_result['moo_method'],
                                                                           moo_result['quantiles']))

    # print('original quantiles loss: {}'.format(moo_result['original_losses']))
    # print('selected quantiles loss: {}'.format(quantiles_loss[selected_ix, :]))

    # %%
    new_weights = get_new_weights(moo_results['lq']['original_weights'], selected_weights) if \
        general_cfg['use_moo_weights'] else None

    results, data = load_predict_model(use_gpu=True,
                                       architecture=experiment_cfg['architecture'],
                                       model_folder=model_folder,
                                       data_config=config.data_config,
                                       data_formatter=formatter,
                                       use_all_data=general_cfg['use_all_data'],
                                       last_layer_weights=new_weights,
                                       exclude_p50=True)

    post_process_results(results, formatter, experiment_cfg, plot_=False)

    q_losses = compute_moo_q_loss(results['quantiles'], results['forecasts'])
    print('lower quantile risk: {} \nupper quantile risk: {}'.format(q_losses[0, :], q_losses[2, :]))

    results['data'] = data
    results['objective_space'] = q_losses
    if general_cfg['save_forecast']:
        save_vars(results, os.path.join(config.results_folder,
                                        experiment_cfg['experiment_name'],
                                        'moo',
                                        '{}{}pred'.format(basename,
                                                          'all_' if general_cfg['use_all_data'] else '')))
