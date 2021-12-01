import os

import pandas as pd

from algorithms.tft2.utils.data import get_col_mapping
from timeseries.experiments.market.expt_settings.configs import ExperimentConfig

# tensorboard --logdir src/timeseries/experiments/market/outputs/saved_models/snp/60t/logs/fit
from timeseries.experiments.market.utils.data import get_column_def_df
from timeseries.experiments.market.utils.filename import get_result_folder
from timeseries.plotly.plot import plotly_time_series
from timeseries.utils.utils import write_text_file, latex_table

if __name__ == "__main__":
    # %%
    general_cfg = {'save_col_def': True}

    results_cfg = {'formatter': 'snp',
                   'experiment_name': '60t_ema_q159',
                   }

    experiment_cfg = {'formatter': 'snp',
                      'dataset_config': 'ES_60t_regime_2015-01_to_2021-06_ema_group_w8',
                      'vars_definition': 'ES_ema_r',
                      'architecture': 'TFTModel'
                      }

    config = ExperimentConfig(experiment_cfg['formatter'], experiment_cfg)
    formatter = config.make_data_formatter()

    train, valid, test = formatter.split_data(config.data_config,
                                              scale=False,
                                              indicators_use_time_subset=True,
                                              )

    col_def_df = get_column_def_df(formatter._column_definition)
    col_def_df['data type'] = col_def_df['data type'].str.lower()
    col_def_df['input type'] = col_def_df['input type'].str.lower()
    if general_cfg['save_col_def']:
        results_folder = os.path.join(get_result_folder(results_cfg))

        write_text_file(os.path.join(results_folder,
                                     'txt',
                                     'col_def'),
                        latex_table('Columns definition', col_def_df.to_latex()))

    train.set_index('datetime', inplace=True)
    test.set_index('datetime', inplace=True)
    valid.set_index('datetime', inplace=True)

    # %%
    df_plot = test.head(1000)
    features = ['ESc', 'ESc_e3', 'ESc_e3_r', 'ESc_r']
    rows = [0, 0, 1, 1]
    plotly_time_series(df_plot, features=features,
                       markers='lines+markers',
                       rows=rows)
