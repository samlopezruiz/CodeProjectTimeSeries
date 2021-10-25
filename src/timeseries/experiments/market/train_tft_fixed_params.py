import os

import tensorflow as tf

from timeseries.data.market.utils.names import get_inst_ohlc_names
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
                      'market_file': 'split_ES_minute_60T_dwn_smpl_2018-01_to_2021-06_g12week_r25_4',
                      'additional_file': 'subset_NQ_minute_60T_dwn_smpl_2012-01_to_2021-07',
                      'regime_file': 'regime_ESc_r_ESc_macd_T10Y2Y_VIX',
                      'macd_vars': ['ESc'],
                      'returns_vars': get_inst_ohlc_names('ES'),
                      'add_prefix_col': 'NQ',
                      'add_macd_vars': ['NQc'],
                      'add_returns_vars': get_inst_ohlc_names('NQ'),
                      'true_target': 'ESc'
                      }

    config = ExperimentConfig(experiment_cfg['formatter'], experiment_cfg)
    formatter = config.make_data_formatter()
    formatter.update_model_params(model_cfg)

    results = train_test_tft(use_gpu=True,
                             prefetch_data=False,
                             model_folder=os.path.join(config.model_folder, experiment_cfg['experiment_name']),
                             data_config=config.data_config,
                             data_formatter=formatter,
                             use_testing_mode=True)

    print('Reconstructing forecasts...')
    results['reconstructed_forecasts'] = reconstruct_forecasts(formatter, results['forecasts'])
    results['experiment_cfg'] = experiment_cfg

    if general_cfg['save_forecast']:
        save_vars(results, os.path.join(config.results_folder, experiment_cfg['experiment_name'], 'test_forecasts'))
