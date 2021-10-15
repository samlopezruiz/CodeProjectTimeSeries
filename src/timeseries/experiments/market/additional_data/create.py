from timeseries.data.market.files.utils import load_market
from timeseries.experiments.market.plot.plot import plot_train_test_groups, plot_mkt_candles
from timeseries.experiments.market.preprocess.func import append_timediff_subsets
from timeseries.experiments.market.split.func import time_subset, set_subsets_and_test
from timeseries.experiments.market.utils.preprocessing import downsample_df
from timeseries.experiments.market.utils.save import save_subsets_and_test, save_market_data
import numpy as np

np.random.seed(42)

if __name__ == '__main__':
    # %%
    in_cfg = {'save_results': True, 'verbose': 1, 'plot_title': True,
              'image_folder': 'img', 'results_folder': 'res'}
    data_cfg = {'inst': "NQ", 'sampling': 'minute', 'suffix': "2012_1-2021_6", 'market': 'cme',
                'src_folder': "data", 'data_from': '2012-01', 'data_to': '2021-07',
                'downsample': True, 'downsample_p': '60T'}
    df, features = load_market(data_cfg)
    df = time_subset(df, data_cfg)
    # plot_mkt_candles(df, data_cfg['inst'], template='plotly_dark')

    # %%
    if data_cfg['downsample']:
        df = downsample_df(df, data_cfg['downsample_p'], ohlc_features=False, inst=data_cfg['inst'])
        # plot_mkt_candles(df.iloc[-30000:, :], data_cfg['inst'], resample=False, period='90T', template='plotly_dark')

    result = {
        'data': df
    }
    save_market_data(result, in_cfg, data_cfg)

