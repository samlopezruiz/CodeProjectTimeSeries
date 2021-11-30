import os
import time

from timeseries.experiments.market.expt_settings.configs import ExperimentConfig
from timeseries.experiments.market.utils.harness import train_test_model
from timeseries.experiments.market.utils.log import log_irace_run
from timeseries.experiments.utils.files import save_vars


def train_test_main(num_encoder_steps=30,
                    num_heads=4,
                    hidden_layer_size=160,
                    learning_rate=0.01,
                    dropout_rate=0.3,
                    minibatch_size=64,
                    pred_steps=5,
                    num_epochs=2):

    model_cfg = {'total_time_steps': num_encoder_steps + pred_steps,
                 'num_encoder_steps': num_encoder_steps,
                 'num_heads': num_heads,
                 'hidden_layer_size': hidden_layer_size,
                 'learning_rate': learning_rate,
                 'dropout_rate': dropout_rate,
                 'minibatch_size': minibatch_size,
                 }

    fixed_cfg = {'num_epochs': num_epochs,
                 }

    experiment_cfg = {'formatter': 'snp',
                      'experiment_name': '60t_ema_irace',
                      'dataset_config': 'ES_60t_regime_2015-01_to_2021-06_ema',
                      'vars_definition': 'ES_ema_r',
                      'architecture': 'TFTModel'
                      }

    config = ExperimentConfig(experiment_cfg['formatter'], experiment_cfg)
    formatter = config.make_data_formatter()
    formatter.update_model_params(model_cfg)
    formatter.update_fixed_params(fixed_cfg)
    model_folder = os.path.join(config.model_folder, experiment_cfg['experiment_name'])

    results = train_test_model(use_gpu=True,
                               architecture=experiment_cfg['architecture'],
                               prefetch_data=False,
                               model_folder=model_folder,
                               data_config=config.data_config,
                               data_formatter=formatter,
                               use_testing_mode=False,
                               predict_eval=False,
                               tb_callback=False,
                               use_best_params=False,
                               indicators_use_time_subset=False,
                               split_data=None,
                               n_train_samples=500
                               )
    log_irace_run(model_cfg, fixed_cfg, experiment_cfg, results)
    return results


if __name__ == "__main__":

    score = train_test_main(30, 2)
    print('score', score)
