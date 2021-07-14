from timeseries.data.market.files.utils import load_market
from timeseries.models.market.plot.plot import plot_train_test_groups
from timeseries.models.market.split.func import subset, set_subsets_and_test

if __name__ == '__main__':
    # %%
    data_cfg = {'inst': "ES", 'sampling': 'minute', 'suffix': "2012_1-2021_6", 'market': 'cme',
                'src_folder': "data", 'data_from': '2018-12', 'data_to': '2021-12'}
    df, features = load_market(data_cfg)
    df = subset(df, data_cfg)
    # plot_mkt_candles(df, data_cfg['inst'], resample=True, period='90T', template='plotly_dark')

    # %%
    split_cfg = {'group': 'day', 'groups_of': 1, 'test_ratio': 0.1, 'random': True,
                 'test_time_start': (8, 30), 'test_time_end': (15, 0), 'time_delta_split': True}

    df_subsets = set_subsets_and_test(df, split_cfg)
    plot_train_test_groups(df_subsets, split_cfg, plot_last=10000,
                           resample=False, period='90T', template='plotly_dark')



