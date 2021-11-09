import os

import tensorflow as tf

from timeseries.experiments.market.expt_settings.configs import ExperimentConfig
from timeseries.experiments.market.utils.harness import train_test_model
from timeseries.experiments.market.utils.results import post_process_results
from timeseries.experiments.utils.files import save_vars

# tensorboard --logdir src/timeseries/experiments/market/outputs/saved_models/snp/60t_macd/logs/fit
if __name__ == "__main__":
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    print("Device Name: ", tf.test.gpu_device_name())
    print('TF eager execution: {}'.format(tf.executing_eagerly()))

    general_cfg = {'save_forecast': True}

    model_cfg = {'total_time_steps': 36 + 5,
                 'num_encoder_steps': 36,
                 'num_heads': 5,
                 'hidden_layer_size': 50,
                 }

    fixed_cfg = {'num_epochs': 3
                 }

    experiment_cfg = {'formatter': 'snp',
                      'experiment_name': '60t_macd',
                      'dataset_config': 'ES_60t_regime_ESc_r_ESc_macd_T10Y2Y_VIX_2015-01_to_2021-06_macd',
                      # 'dataset_config': 'ES_60t_regime_ESc_r_ESc_macd_T10Y2Y_VIX_2018-01_to_2021-06',
                      'vars_definition': 'ES_fast_macd',
                      'architecture': 'TFTModel'
                      }

    config = ExperimentConfig(experiment_cfg['formatter'], experiment_cfg)
    formatter = config.make_data_formatter()
    formatter.update_model_params(model_cfg)
    formatter.update_fixed_params(fixed_cfg)
    model_folder = os.path.join(config.model_folder, experiment_cfg['experiment_name'])

    results = train_test_model(use_gpu=True,
                               architecture=experiment_cfg['architecture'],
                               prefetch_data=True,
                               model_folder=model_folder,
                               data_config=config.data_config,
                               data_formatter=formatter,
                               use_testing_mode=False,
                               predict_eval=True,
                               tb_callback=False,
                               use_best_params=False,
                               indicators_use_time_subset=False
                               )

    post_process_results(results, formatter, experiment_cfg)

    if general_cfg['save_forecast']:
        save_vars(results, os.path.join(config.results_folder,
                                        experiment_cfg['experiment_name'],
                                        '{}_{}_forecasts'.format(experiment_cfg['architecture'],
                                                                 experiment_cfg['vars_definition'])))

    print(results['hit_rates']['global_hit_rate'][1])
