import os

import joblib
import numpy as np
import seaborn as sns
import tensorflow as tf

from algorithms.tft2.harness.train_test import compute_moo_q_loss
from timeseries.experiments.market.moo.utils.utils import get_loss_to_obj_function, rank_solutions, aggregate_qcd_qee
from timeseries.experiments.market.plot.plot import plot_2D_moo_results_equal_w
from timeseries.experiments.market.utils.filename import get_result_folder, quantiles_name, termination_name
from timeseries.experiments.market.utils.harness import load_predict_model, get_model_data_config
from timeseries.experiments.market.utils.results import post_process_results
from timeseries.experiments.utils.files import save_vars
from timeseries.plotly.plot import plot_4D

sns.set_theme('poster')

if __name__ == "__main__":
    # %%
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    print("Device Name: ", tf.test.gpu_device_name())
    print('TF eager execution: {}'.format(tf.executing_eagerly()))

    general_cfg = {'save_forecast': False,
                   'save_plot': False,
                   'use_all_data': True,
                   'use_moo_weights': True}

    results_cfg = {'formatter': 'snp',
                   'experiment_name': '60t_ema_q159',
                   'results': 'TFTModel_ES_ema_r_q159_NSGA2_g250_p100_s1_k2_wmoo'
                   }

    moo_result = joblib.load(os.path.join(get_result_folder(results_cfg), 'moo', results_cfg['results'] + '.z'))

    config, formatter, model_folder = get_model_data_config(moo_result['experiment_cfg'],
                                                            moo_result['model_cfg'],
                                                            moo_result['fixed_cfg'])

    experiment_cfg = moo_result['experiment_cfg']

    weights, quantiles_loss, eq_quantiles_loss = moo_result['X'], moo_result['F'], moo_result['eq_F']

    basename = '{}_{}_q{}_{}_{}_p{}_s{}_k{}_'.format(experiment_cfg['architecture'],
                                                     experiment_cfg['vars_definition'],
                                                     quantiles_name(moo_result['quantiles']),
                                                     moo_result['moo_method'],
                                                     termination_name(moo_result['algo_cfg']['termination']),
                                                     moo_result['algo_cfg']['pop_size'],
                                                     int(moo_result['algo_cfg']['use_sampling']),
                                                     quantiles_loss.shape[1],
                                                     )
    # %%
    selected_ix = None
    original_ix = np.argmin(np.sum(np.abs(quantiles_loss - moo_result['original_losses']), axis=1))

    filter_thold = 1.
    total_selected_error = 1.25  # for 2k
    sort_weights = [0.2, 0.2, 0.6, 0.2]  # for 4k
    camera_position = np.array([0.3, 1.25, .6]) * 1.5

    xaxis_limit = filter_thold
    if quantiles_loss.shape[1] == 2:
        Fs_x_plot_masks = quantiles_loss[:, 0] < xaxis_limit
        selected_ix = np.argmin(np.abs(np.sum(eq_quantiles_loss[Fs_x_plot_masks, :], axis=1) - total_selected_error))

        quantiles_loss_2k = quantiles_loss
        eq_quantiles_loss_2k = eq_quantiles_loss

    elif quantiles_loss.shape[1] == 4:

        tholds = [filter_thold] * 4
        mask = np.all(np.vstack([quantiles_loss[:, i] < thold for i, thold in enumerate(tholds)]).T, axis=1)
        filtered_quantiles = quantiles_loss[mask, :]

        ranked_ix = rank_solutions(filtered_quantiles, sort_weights)
        ranked_losses = filtered_quantiles[ranked_ix, :]

        selected_filtered_ix = ranked_ix[0]
        selected_ix = np.argmax(np.all(quantiles_loss == filtered_quantiles[selected_filtered_ix, :], axis=1))
        plot_ranked_losses = [x.reshape(1, -1) if len(x.shape) == 1 else x for x in ranked_losses]

        quantiles_loss_2k = aggregate_qcd_qee(quantiles_loss)
        eq_quantiles_loss_2k = aggregate_qcd_qee(eq_quantiles_loss)

    img_path = os.path.join(config.results_folder,
                            experiment_cfg['experiment_name'],
                            'img',
                            '{}{}_pf'.format(basename,
                                             '_ix_{}'.format(selected_ix) if general_cfg['use_moo_weights'] else '')),
    if quantiles_loss.shape[1] == 4:
        axis_labels = ['QCR lower bound', 'QER lower bound',
                       'QCR upper bound', 'QER upper bound']

        plot_4D(filtered_quantiles,
                color_col=3,
                ranked_F=plot_ranked_losses,
                original_point=moo_result['original_losses'],
                selected_point=quantiles_loss[selected_ix, :],
                save=general_cfg['save_plot'],
                axis_labels=axis_labels,
                file_path=img_path,
                label_scale=1,
                size=(1980, 1080),
                save_png=False,
                title='',
                camera_position=camera_position
                )

    plot_2D_moo_results_equal_w(quantiles_loss_2k, eq_quantiles_loss_2k,
                                selected_ixs=selected_ix if general_cfg['use_moo_weights'] else None,
                                save=general_cfg['save_plot'],
                                file_path=img_path,
                                original_ixs=original_ix,
                                figsize=(20, 15),
                                xaxis_limit=xaxis_limit,
                                title='MOO using {} for quantiles: {}'.format(moo_result['moo_method'],
                                                                      moo_result['quantiles']))

    print('original quantiles loss: {}'.format(moo_result['original_losses']))
    print('selected quantiles loss: {}'.format(quantiles_loss[selected_ix, :]))

    # %%
    # selected_weights = weights[selected_ix, :] if general_cfg['use_moo_weights'] else None
    # results, data = load_predict_model(use_gpu=True,
    #                                    architecture=experiment_cfg['architecture'],
    #                                    model_folder=model_folder,
    #                                    data_config=config.data_config,
    #                                    data_formatter=formatter,
    #                                    use_all_data=general_cfg['use_all_data'],
    #                                    last_layer_weights=selected_weights,
    #                                    exclude_p50=True)
    #
    # post_process_results(results, formatter, experiment_cfg, plot_=False)
    #
    # q_losses = compute_moo_q_loss(results['quantiles'], results['forecasts'])
    # obj = get_loss_to_obj_function(moo_result['loss_to_obj_type'])(q_losses)
    # print('objective space: {}'.format(obj))
    #
    # results['data'] = data
    # results['objective_space'] = obj
    # if general_cfg['save_forecast']:
    #     save_vars(results, os.path.join(config.results_folder,
    #                                     experiment_cfg['experiment_name'],
    #                                     'img',
    #                                     '{}{}{}pf'.format(basename,
    #                                                       'ix{}_'.format(selected_ix) if general_cfg[
    #                                                           'use_moo_weights'] else '',
    #                                                       'all_' if general_cfg['use_all_data'] else '')))
