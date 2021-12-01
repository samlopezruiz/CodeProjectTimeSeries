import os

import numpy as np
import pandas as pd

from algorithms.hmm.func import resample_dfs
from timeseries.data.market.files.utils import load_market, load_data
from timeseries.experiments.market.preprocess.func import append_timediff_subsets
from timeseries.experiments.market.split.func import time_subset, set_subsets_and_test
from timeseries.experiments.market.utils.filename import subsets_and_test_filename
from timeseries.experiments.market.utils.preprocessing import downsample_df, add_date_known_inputs
from timeseries.experiments.market.utils.save import save_subsets_and_test
from timeseries.plotly.plot import plotly_ts_regime

np.random.seed(42)

if __name__ == '__main__':
    # %%
    in_cfg = {'save_results': True,
              'save_plot': False,
              'verbose': 1,
              'plot_title': False,
              'image_folder': 'img',
              'results_folder': 'res'}

    data_cfg = {'inst': "ES",
                'sampling': 'minute',
                'suffix': "2012_1-2021_6",
                'market': 'cme',
                'src_folder': "data",
                'data_from': '2015-01',
                'data_to': '2021-06',
                'vol_filename': "Vol_5levels_ESc_2012_1-2021_6.z",
                'downsample': True,
                'downsample_p': '5T'}

    split_cfg = {'group': 'week',
                 'groups_of': 8,
                 'test_ratio': 0.15,
                 'valid_ratio': 0.15,
                 'random': True,
                 'time_thold': {'hours': 3, 'seconds': None, 'days': None},
                 'test_time_start': (8, 30),
                 'test_time_end': (15, 0),
                 'time_delta_split': True, }

    df = load_market(data_cfg)
    df = time_subset(df, data_cfg)

    if 'vol_filename' in data_cfg:
        vol_profile_levels = load_data(filename=data_cfg['vol_filename'],
                                       path=os.path.join('..', 'volume', 'res'))

        vol_profile_levels_resampled = resample_dfs(df, vol_profile_levels)
        df = pd.concat([df, vol_profile_levels_resampled], axis=1)
    # plot_mkt_candles(df, data_cfg['inst'], template='plotly_dark')

    # %%
    if data_cfg['downsample']:
        df = downsample_df(df, data_cfg['downsample_p'], ohlc_features=False, inst=data_cfg['inst'])
    # plot_mkt_candles(df.iloc[-30000:, :], data_cfg['inst'], resample=False, period='90T', template='plotly_dark')

    # %%
    df_subsets = set_subsets_and_test(df, split_cfg)
    append_timediff_subsets(df_subsets, split_cfg['time_thold'])

    # Add known inputs
    add_date_known_inputs(df_subsets)

    #%%
    df_plot = df_subsets.iloc[-40000:-10000, :]
    plotly_ts_regime(df_plot,
                     features=['ESc', 'test_train_subset', 'week_of_year'],
                     rows=[0, 1, 2],
                     resample=False,
                     regime_col='test',
                     period='90T',
                     markers='markers',
                     markersize=5,
                     plot_title=in_cfg['plot_title'],
                     template='plotly_white',
                     save=in_cfg['save_plot'],
                     file_path=[in_cfg['image_folder'], subsets_and_test_filename(data_cfg, split_cfg)],
                     save_png=True,
                     legend=True,
                     label_scale=2,
                     title='SPLIT CFG: {}'.format(str(split_cfg)),
                     legend_labels=['train', 'test', 'val'])

    df_subsets['symbol'] = data_cfg['inst']
    result = {
        'data': df_subsets,
        'split_cfg': split_cfg,
        'data_cfg': data_cfg
    }
    save_subsets_and_test(result, in_cfg, data_cfg, split_cfg)

    head = df_subsets.head(1000)
