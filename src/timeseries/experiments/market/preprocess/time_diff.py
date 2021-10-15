from timeseries.data.market.files.utils import load_market
from timeseries.experiments.market.plot.plot import plot_mkt_candles
from timeseries.experiments.market.preprocess.func import append_timediff_subsets, add_features
from timeseries.experiments.market.split.func import time_subset
from timeseries.preprocessing.func import macd, ln_returns

if __name__ == '__main__':
    in_cfg = {'save_results': False, 'verbose': 1, 'plot_title': True,
              'image_folder': 'img', 'results_folder': 'res'}
    data_cfg = {'inst': "ES", 'sampling': 'day', 'suffix': "2012_5-2021_6", 'market': 'cme',
                'src_folder': "data", 'data_from': '2011-12', 'data_to': '2021-12'}
    df, features = load_market(data_cfg)
    df = time_subset(df, data_cfg)

    # plot_mkt_candles(df, data_cfg['inst'], resample=False, period='90T', template='plotly_dark')

    # %%
    time_diff_cfg = {'hours': None, 'seconds': None, 'days': 4}
    append_timediff_subsets(df, time_diff_cfg)
    add_features(df, macds=['ESc'], returns=['ESc'], use_time_subset=True)
    #%%




