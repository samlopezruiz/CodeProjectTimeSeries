import os

import pandas as pd
import tensorflow as tf

from timeseries.experiments.market.expt_settings.configs import ExperimentConfig
from timeseries.experiments.market.utils.harness import train_test_model
from timeseries.experiments.market.utils.results import post_process_results
from timeseries.experiments.utils.files import save_vars

# tensorboard --logdir src/timeseries/experiments/market/outputs/saved_models/snp/60t_macd/logs/fit
from timeseries.plotly.plot import plotly_time_series
from timeseries.utils.parallel import repeat

if __name__ == "__main__":
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    print("Device Name: ", tf.test.gpu_device_name())
    print('TF eager execution: {}'.format(tf.executing_eagerly()))

    general_cfg = {'save_forecast': False}

    model_cfg = {'total_time_steps': 48 + 5,
                 'num_encoder_steps': 48,
                 'num_heads': 6,
                 'hidden_layer_size': 128,
                 }

    fixed_cfg = {'num_epochs': 1
                 }

    experiment_cfg = {'formatter': 'snp',
                      'experiment_name': '60t_ema_test',
                      'dataset_config': 'ES_60t_regime_2015-01_to_2021-06_ema',
                      'vars_definition': 'ES_ema_r',
                      'architecture': 'TFTModel'
                      }

    config = ExperimentConfig(experiment_cfg['formatter'], experiment_cfg)
    formatter = config.make_data_formatter()
    formatter.update_model_params(model_cfg)
    formatter.update_fixed_params(fixed_cfg)
    model_folder = os.path.join(config.model_folder, experiment_cfg['experiment_name'])

    n_repeat = 2
    results = []
    for _ in range(n_repeat):
        results.append(train_test_model(use_gpu=True,
                                        architecture=experiment_cfg['architecture'],
                                        prefetch_data=False,
                                        model_folder=model_folder,
                                        data_config=config.data_config,
                                        data_formatter=formatter,
                                        use_testing_mode=True,
                                        predict_eval=True,
                                        tb_callback=False,
                                        use_best_params=False,
                                        indicators_use_time_subset=False
                                        ))

    # %%
    histories = [res['history'] for res in results]

    df = pd.DataFrame(histories)
    plotly_time_series(df,
                       title='Loss History',
                       save=False,
                       legend=True,
                       rows=[1, 1],
                       file_path=os.path.join(config.results_folder,
                                              'img',
                                              '{}_{}_loss_hist'.format(experiment_cfg['architecture'],
                                                                       experiment_cfg['vars_definition'])),
                       size=(1980, 1080),
                       color_col=None,
                       markers='lines+markers',
                       xaxis_title="epoch",
                       markersize=5,
                       plot_title=True,
                       label_scale=1,
                       plot_ytitles=False)

    # post_process_results(results, formatter, experiment_cfg)
    #
    # if general_cfg['save_forecast']:
    #     save_vars(results, os.path.join(config.results_folder,
    #                                     experiment_cfg['experiment_name'],
    #                                     '{}_{}_forecasts'.format(experiment_cfg['architecture'],
    #                                                              experiment_cfg['vars_definition'])))
    #
    # print(results['hit_rates']['global_hit_rate'][1])
