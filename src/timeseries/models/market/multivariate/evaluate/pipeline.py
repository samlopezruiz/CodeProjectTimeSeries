from timeseries.data.market.files.utils import load_market
from timeseries.models.market.multivariate.architectures.cnnlstm import cnnlstm_func
from timeseries.models.market.plot.plot import plot_train_test_groups
from timeseries.models.market.split.func import time_subset, set_subsets_and_test, get_subsets, get_xy

if __name__ == '__main__':
    # %%
    in_cfg = {'save_results': False, 'verbose': 1, 'plot_title': True,
              'image_folder': 'img', 'results_folder': 'res'}
    data_cfg = {'inst': "ES", 'sampling': 'day', 'suffix': "2012_5-2021_6", 'market': 'cme',
                'src_folder': "data", 'data_from': '2011-12', 'data_to': '2021-12'}
    df, features = load_market(data_cfg)
    df = time_subset(df, data_cfg)

    # %% Training-Test Subsets
    split_cfg = {'group': 'week', 'groups_of': 12, 'test_ratio': 0.25, 'random': True, 'time_thold': 1000,
                 'test_time_start': (8, 30), 'test_time_end': (15, 0), 'time_delta_split': False}

    df_subsets = set_subsets_and_test(df, split_cfg)
    plot_train_test_groups(df_subsets, split_cfg, plot_last=30000, features=['ESc', 'subset'],
                           resample=False, period='90T', template='plotly_dark')

    #%% Preprocessing
    df_pp = df_subsets.copy()
    subsets = get_subsets(df_pp)

    #%% Training XY Subsets
    training_cfg = {'inst': data_cfg['inst'], 'y_var': 'ESc', 'features': ['volume', 'atr'],
                    "append_train_to_test": True}
    model_cfg = {"n_steps_out": 6, "n_steps_in": 8, "n_seq": 2}
    model_func = cnnlstm_func()
    lookback = model_func['lookback'](model_cfg)

    train_X, test_X, features = get_xy(subsets, training_cfg, lookback, dim_f=1)

    #%% Train Model

