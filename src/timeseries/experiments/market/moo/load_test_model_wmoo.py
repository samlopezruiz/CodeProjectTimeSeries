import os

import joblib
import numpy as np
import tensorflow as tf

from algorithms.tft2.harness.train_test import compute_moo_q_loss
from timeseries.experiments.market.expt_settings.configs import ExperimentConfig
from timeseries.experiments.market.plot.plot import plot_2D_pareto_front, plot_2D_moo_results
from timeseries.experiments.market.utils.filename import get_result_folder, quantiles_name
from timeseries.experiments.market.utils.harness import load_predict_model, get_model_data_config
from timeseries.experiments.market.utils.results import post_process_results
from timeseries.experiments.utils.files import save_vars
import seaborn as sns

sns.set_theme('poster')

if __name__ == "__main__":
    # %%
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    print("Device Name: ", tf.test.gpu_device_name())
    print('TF eager execution: {}'.format(tf.executing_eagerly()))

    general_cfg = {'save_forecast': True,
                   'save_plot': True,
                   'use_all_data': True,
                   'use_moo_weights': True}

    results_cfg = {'formatter': 'snp',
                   'experiment_name': '60t_ema_q357',
                   'results': 'ES_ema_r_q357_moo_weights'
                   }

    # moo_result = joblib.load(os.path.join(get_result_folder(results_cfg), results_cfg['results'] + '.z'))
    moo_result = joblib.load(os.path.join(get_result_folder(results_cfg), 'moo', results_cfg['results'] + '.z'))

    config, formatter, model_folder = get_model_data_config(moo_result['experiment_cfg'],
                                                            moo_result['model_cfg'],
                                                            moo_result['fixed_cfg'])

    experiment_cfg = moo_result['experiment_cfg']

    weights, quantiles_loss, eq_quantiles_loss = moo_result['X'], moo_result['F'], moo_result['eq_F']

    # %%
    original_ix = np.argmin(np.sum(np.abs(quantiles_loss - moo_result['original_losses']), axis=1))
    xaxis_limit = 0.6
    total_selected_error = 1.25

    Fs_x_plot_masks = quantiles_loss[:, 0] < xaxis_limit
    selected_ix = np.argmin(np.abs(np.sum(eq_quantiles_loss[Fs_x_plot_masks, :], axis=1) - total_selected_error))
    # selected_ix = np.searchsorted(quantiles_loss[:, 0], qcp_selected_error, side='left',)

    filename = '{}{}_q{}_pf'.format(experiment_cfg['vars_definition'],
                                    '_ix_{}'.format(selected_ix) if general_cfg['use_moo_weights'] else '',
                                    quantiles_name(moo_result['quantiles']))

    plot_2D_moo_results(quantiles_loss, eq_quantiles_loss,
                        selected_ix=selected_ix if general_cfg['use_moo_weights'] else None,
                        save=general_cfg['save_plot'],
                        file_path=os.path.join(config.results_folder,
                                               experiment_cfg['experiment_name'],
                                               'img',
                                               filename),
                        original_ixs=original_ix,
                        figsize=(20, 15),
                        xaxis_limit=xaxis_limit,
                        title='Multi objective optimization for quantiles: {}'.format(moo_result['quantiles']))

    # %%
    selected_weights = weights[selected_ix, :] if general_cfg['use_moo_weights'] else None
    results, data = load_predict_model(use_gpu=True,
                                       architecture=experiment_cfg['architecture'],
                                       model_folder=model_folder,
                                       data_config=config.data_config,
                                       data_formatter=formatter,
                                       use_all_data=general_cfg['use_all_data'],
                                       last_layer_weights=selected_weights,
                                       exclude_p50=True)

    post_process_results(results, formatter, experiment_cfg, plot_=False)
    loss_to_obj = lambda x: np.mean(x, axis=0)  # moo_result['loss_to_obj'] #
    q_losses = compute_moo_q_loss(results['quantiles'], results['forecasts'])
    obj = loss_to_obj(q_losses)
    print('objective space: {}'.format(obj))

    results['data'] = data
    results['objective_space'] = obj
    if general_cfg['save_forecast']:
        filename = '{}_{}{}_q{}{}_pred'.format(experiment_cfg['architecture'],
                                               'all_' if general_cfg['use_all_data'] else '',
                                               experiment_cfg['vars_definition'],
                                               quantiles_name(results['quantiles']),
                                               '_moo_ix{}'.format(selected_ix) if general_cfg[
                                                   'use_moo_weights'] else '')

        save_vars(results, os.path.join(config.results_folder,
                                        experiment_cfg['experiment_name'],
                                        'moo',
                                        filename))
