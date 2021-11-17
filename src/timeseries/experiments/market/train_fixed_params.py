import math
import os

import numpy as np
import tensorflow as tf
import telegram_send
from timeseries.experiments.market.expt_settings.configs import ExperimentConfig
from timeseries.experiments.market.utils.filename import quantiles_name
from timeseries.experiments.market.utils.harness import train_test_model, get_model_data_config
from timeseries.experiments.market.utils.results import post_process_results
from timeseries.experiments.utils.files import save_vars

# tensorboard --logdir src/timeseries/experiments/market/outputs/saved_models/snp/60t_ema/logs/fit
if __name__ == "__main__":
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    print("Device Name: ", tf.test.gpu_device_name())
    print('TF eager execution: {}'.format(tf.executing_eagerly()))

    general_cfg = {'save_results': True,
                   'send_notifications': True}

    model_cfg = {'total_time_steps': 48 + 5,
                 'num_encoder_steps': 48,
                 'num_heads': 4,
                 'hidden_layer_size': 128,
                 'learning_rate': 0.01,
                 }

    fixed_cfg = {'quantiles': [0.1, 0.5, 0.9],
                 }

    # experiment_cfg = {'formatter': 'snp',
    #                   'experiment_name': '5t_ema_test',
    #                   'dataset_config': 'ES_5t_regime_2015-01_to_2021-06_ema_r',
    #                   'vars_definition': 'ES_slow_ema_r',
    #                   'architecture': 'TFTModel'
    #                   }

    experiment_cfg = {'formatter': 'snp',
                      'experiment_name': '60t_ema_q159',
                      'dataset_config': 'ES_60t_regime_2015-01_to_2021-06_ema',
                      'vars_definition': 'ES_ema_r',
                      'architecture': 'TFTModel'
                      }

    # experiment_cfg = {'formatter': 'snp',
    #                   'experiment_name': '60t_macd_irace',
    #                   'dataset_config': 'ES_60t_regime_ESc_r_ESc_macd_T10Y2Y_VIX_2015-01_to_2021-06_macd',
    #                   'vars_definition': 'ES_fast_macd_vol',
    #                   'architecture': 'TFTModel'
    #                   }

    config, formatter, model_folder = get_model_data_config(experiment_cfg, model_cfg, fixed_cfg)

    results = train_test_model(use_gpu=True,
                               architecture=experiment_cfg['architecture'],
                               prefetch_data=False,
                               model_folder=model_folder,
                               data_config=config.data_config,
                               data_formatter=formatter,
                               use_testing_mode=False,
                               predict_eval=True,
                               tb_callback=True,
                               use_best_params=False,
                               indicators_use_time_subset=True
                               )
    filename = '{}_{}_q{}_lr{}_pred'.format(experiment_cfg['architecture'],
                                            experiment_cfg['vars_definition'],
                                            quantiles_name(results['quantiles']),
                                            str(results['learning_rate'])[2:],
                                            )
    if general_cfg['send_notifications']:
        telegram_send.send(messages=["training for {} completed".format(filename)])

    post_process_results(results, formatter, experiment_cfg)
    if general_cfg['save_results']:
        results['model_cfg'] = model_cfg
        results['fixed_cfg'] = fixed_cfg
        results['experiment_cfg'] = experiment_cfg
        save_vars(results, os.path.join(config.results_folder,
                                        experiment_cfg['experiment_name'],
                                        filename))

    print(results['hit_rates']['global_hit_rate'][1])
