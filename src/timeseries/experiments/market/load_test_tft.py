# Lint as: python3
"""Trains TFT based on a defined set of parameters.

Uses default parameters supplied from the configs file to train a TFT model from
scratch.

Usage:
python3 script_train_fixed_params {expt_name} {output_folder}

Command line args:
  expt_name: Name of dataset/experiment to train.
  output_folder: Root folder in which experiment is saved


"""

import os
import tensorflow as tf

from timeseries.experiments.market.expt_settings.configs import ExperimentConfig
from timeseries.experiments.market.utils.harness import load_predict_model
from timeseries.experiments.market.utils.results import post_process_results
from timeseries.experiments.utils.files import save_vars

if __name__ == "__main__":
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    print("Device Name: ", tf.test.gpu_device_name())
    print('TF eager execution: {}'.format(tf.executing_eagerly()))

    general_cfg = {'save_forecast': True,
                   'use_all_data': True}

    model_cfg = {'total_time_steps': 24 + 5,
                 'num_encoder_steps': 24,
                 'num_heads': 5,
                 'hidden_layer_size': 50,
                 }

    experiment_cfg = {'formatter': 'snp',
                      'experiment_name': '60t_macd',
                      'dataset_config': 'ES_60t_regime_ESc_r_ESc_macd_T10Y2Y_VIX_2015-01_to_2021-06_macd',
                      # 'dataset_config': 'ES_60t_regime_ESc_r_ESc_macd_T10Y2Y_VIX_2018-01_to_2021-06',
                      'vars_definition': 'ES_macd_vol'
                      }

    config = ExperimentConfig(experiment_cfg['formatter'], experiment_cfg)
    formatter = config.make_data_formatter()
    formatter.update_model_params(model_cfg)
    model_folder = os.path.join(config.model_folder, experiment_cfg['experiment_name'])

    results, data = load_predict_model(use_gpu=True,
                                       model_folder=model_folder,
                                       data_config=config.data_config,
                                       data_formatter=formatter,
                                       use_all_data=general_cfg['use_all_data'])

    post_process_results(results, formatter, experiment_cfg)

    results['data'] = data
    if general_cfg['save_forecast']:
        save_vars(results, os.path.join(config.results_folder,
                                        experiment_cfg['experiment_name'],
                                        '{}{}_forecasts'.format('all_' if general_cfg['use_all_data'] else '',
                                                                experiment_cfg['vars_definition'])))
