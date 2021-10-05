from timeseries.data.market.files.utils import load_market
from timeseries.models.market.plot.plot import plot_train_test_groups, plot_mkt_candles
from timeseries.models.market.preprocess.func import append_timediff_subsets
from timeseries.models.market.split.func import time_subset, set_subsets_and_test
from timeseries.models.market.utils.filename import subset_filename, subsets_and_test_filename
from timeseries.models.market.utils.preprocessing import downsample_df
from timeseries.models.market.utils.save import save_subsets_and_test
import numpy as np

np.random.seed(42)

if __name__ == '__main__':
    # %%
    in_cfg = {'save_results': False, 'save_plot': True, 'verbose': 1, 'plot_title': True,
              'image_folder': 'img', 'results_folder': 'res'}
    data_cfg = {'inst': "ES", 'sampling': 'minute', 'suffix': "2012_1-2021_6", 'market': 'cme',
                'src_folder': "data", 'data_from': '2018-01', 'data_to': '2021-06',
                'downsample': True, 'downsample_p': '60T'}
    df, features = load_market(data_cfg)
    df = time_subset(df, data_cfg)
    # plot_mkt_candles(df, data_cfg['inst'], template='plotly_dark')

    # %%
    if data_cfg['downsample']:
        df = downsample_df(df, data_cfg['downsample_p'], ohlc_features=False, inst=data_cfg['inst'])
    # plot_mkt_candles(df.iloc[-30000:, :], data_cfg['inst'], resample=False, period='90T', template='plotly_dark')

    # %%
    split_cfg = {'group': 'week', 'groups_of': 12, 'test_ratio': 0.25, 'random': True,
                 'time_thold': {'hours': 3, 'seconds': None, 'days': None},
                 'test_time_start': (8, 30), 'test_time_end': (15, 0), 'time_delta_split': True, }
    df_subsets = set_subsets_and_test(df, split_cfg)
    append_timediff_subsets(df_subsets, split_cfg['time_thold'])
    plot_train_test_groups(df_subsets, split_cfg, plot_last=30000, features=['ESc', 'subset'],
                           resample=False, period='90T', template='plotly_dark', save=in_cfg['save_plot'],
                           file_path=[in_cfg['image_folder'], subsets_and_test_filename(data_cfg, split_cfg)])

    save_subsets_and_test([df_subsets, split_cfg, data_cfg], in_cfg, data_cfg, split_cfg)

