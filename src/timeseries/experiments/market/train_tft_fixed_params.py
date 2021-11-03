import os

import tensorflow as tf

from timeseries.experiments.market.expt_settings.configs import ExperimentConfig
from timeseries.experiments.market.utils.harness import train_test_tft
from timeseries.experiments.market.utils.preprocessing import reconstruct_forecasts
from timeseries.experiments.utils.files import save_vars

# tensorboard --logdir src/timeseries/experiments/market/outputs/saved_models/snp/60t/logs/fit
if __name__ == "__main__":
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    print("Device Name: ", tf.test.gpu_device_name())
    print('TF eager execution: {}'.format(tf.executing_eagerly()))


    general_cfg = {'save_forecast': True}

    model_cfg = {'total_time_steps': 50 + 6,
                 'num_encoder_steps': 50,
                 'num_heads': 5,
                 'hidden_layer_size': 160,
                 }

    experiment_cfg = {'formatter': 'snp',
                      'experiment_name': '60t',
                      'dataset_config': 'ES_60t_regime_ESc_r_ESc_macd_T10Y2Y_VIX',
                      'vars_definition': 'ES_r_all'
                      }

    config = ExperimentConfig(experiment_cfg['formatter'], experiment_cfg)
    formatter = config.make_data_formatter()
    formatter.update_model_params(model_cfg)
    model_folder = os.path.join(config.model_folder, experiment_cfg['experiment_name'])

    results = train_test_tft(use_gpu=True,
                             prefetch_data=False,
                             model_folder=model_folder,
                             data_config=config.data_config,
                             data_formatter=formatter,
                             use_testing_mode=True,
                             predict_eval=True,
                             tb_callback=True,
                             use_best_params=False
                             )

    print('Reconstructing forecasts...')
    results['reconstructed_forecasts'] = reconstruct_forecasts(formatter, results['forecasts'])
    results['experiment_cfg'] = experiment_cfg

    if general_cfg['save_forecast']:
        save_vars(results, os.path.join(config.results_folder, experiment_cfg['experiment_name'], 'test_forecasts'))
