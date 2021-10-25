import os

from timeseries.data.market.utils.names import get_inst_ohlc_names
from timeseries.experiments.market.expt_settings.configs import ExperimentConfig
from timeseries.experiments.market.utils.harness import train_test_tft


def train_test_main(num_encoder_steps, num_heads):
    model_cfg = {'total_time_steps': num_encoder_steps + 5,
                 'num_encoder_steps': num_encoder_steps,
                 'num_heads': num_heads,
                 'hidden_layer_size': 160,
                 }

    fixed_cfg = {'num_epochs': 3,
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
    formatter.update_fixed_params(fixed_cfg)

    results = train_test_tft(use_gpu=True,
                             prefetch_data=False,
                             model_folder=os.path.join(config.model_folder, experiment_cfg['experiment_name']),
                             data_config=config.data_config,
                             data_formatter=formatter,
                             use_testing_mode=False,
                             predict_eval=False)
    return results


if __name__ == "__main__":
    score = train_test_main(30, 2)
    print('score', score)
