import numpy as np

from timeseries.data.market.files.utils import load_market
from timeseries.data.market.utils.names import get_inst_ohlc_names
from timeseries.models.lorenz.functions.dataprep import step_feature_multi_step_xy_from_mv
from timeseries.models.market.multivariate.architectures.cnnlstm import cnnlstm_func
from timeseries.models.market.plot.plot import plot_train_test_groups
from timeseries.models.market.split.func import subset, set_subsets_and_test, get_subsets, get_xy_from_subsets, get_xy

if __name__ == '__main__':
    # %%
    data_cfg = {'inst': "ES", 'sampling': 'day', 'suffix': "2012_5-2021_6", 'market': 'cme',
                'src_folder': "data", 'data_from': '2011-12', 'data_to': '2021-12'}
    df, features = load_market(data_cfg)
    df = subset(df, data_cfg)
    # plot_mkt_candles(df, data_cfg['inst'], resample=True, period='90T', template='plotly_dark')

    # %%
    split_cfg = {'group': 'week', 'groups_of': 12, 'test_ratio': 0.25, 'random': True,
                 'test_time_start': (8, 30), 'test_time_end': (15, 0), 'time_delta_split': False, 'time_thold': 1000}

    df_subsets = set_subsets_and_test(df, split_cfg)
    plot_train_test_groups(df_subsets, split_cfg, plot_last=30000, features=['ESc', 'subset'],
                           resample=False, period='90T', template='plotly_dark')

    #%% PREPROCESSING FIRST
    df_pp = df_subsets.copy()
    subsets = get_subsets(df_pp)

    #%% SPLIT
    training_cfg = {'inst': data_cfg['inst'], 'y_var': 'ESc', 'features': ['volume', 'atr'],
                    "append_train_to_test": True}
    model_cfg = {"n_steps_out": 6, "n_steps_in": 8, "n_seq": 2}
    model_func = cnnlstm_func()
    lookback = model_func['lookback'](model_cfg)

    train_X, test_X, features = get_xy(subsets, training_cfg, lookback, dim_f=1)

