import numpy as np
import pandas as pd

from timeseries.data.market.files.utils import describe
from timeseries.data.market.utils.names import get_inst_ohlc_names


def set_subsets_and_test(df, split_cfg):
    df_subsets = group_by(df, split_cfg)
    if split_cfg['random'] and split_cfg['groups_of'] == 1 and split_cfg['group'] == 'day':
        print('random test split')
        random_test_split(df_subsets, split_cfg)
    else:
        print('simple test split')
        simple_test_split(df_subsets, test_ratio=split_cfg['test_ratio'])
    if split_cfg['time_delta_split']:
        update_subset_time_delta(df_subsets, time_thold=split_cfg.get('time_thold', 1000))
    df_subsets.drop(['year', 'week', 'day', 'hour'], axis=1, inplace=True)
    return df_subsets


def shift_subsets(df):
    # mask_ini = (df_subsets['test'].shift(periods=-1, fill_value=0) - df_subsets['test']).eq(1)
    # ix_ini = df_subsets.loc[mask_ini, :].index
    mask_end = (df['test'].shift(periods=1, fill_value=0) - df['test']).eq(1)
    ix_end = df.loc[mask_end, :].index
    df['subset'] = 0
    for i in range(len(ix_end) - 1):
        df.loc[ix_end[i]:ix_end[i + 1], 'subset'] = i + 1
    df.loc[ix_end[i + 1]:, 'subset'] = i + 2


def random_test_split(df, split_cfg):
    test_time_ini, test_time_end = split_cfg['test_time_start'], split_cfg['test_time_end']
    test_ratio = split_cfg['test_ratio']
    df['test'] = 0
    grp = df.groupby('subset')
    for i, (group_cols, subset) in enumerate(grp):
        t = subset.index[0]
        t_ini = t.replace(hour=test_time_ini[0], minute=test_time_ini[1], second=0)
        t_end = t.replace(hour=test_time_end[0], minute=test_time_end[1], second=0)
        test_len = int(round(subset.shape[0] * test_ratio, 0))
        low = subset.index.searchsorted(t_ini)
        high = subset.index.searchsorted(t_end)
        if high - test_len > low:
            ix_start = np.random.randint(low, high - test_len)
            df.loc[subset.index[ix_start:ix_start + test_len], 'test'] = 1

    shift_subsets(df)


def subset(df, cfg, describe_=True):
    df_ss = df.loc[cfg.get('data_from', None):cfg.get('data_to', None)].copy()
    if describe_:
        describe(df_ss)
    return df_ss


def group_by(df_orig, cfg):
    group, groups_of = cfg['group'], cfg['groups_of']
    df = df_orig.copy()
    df['year'] = df.index.isocalendar().year
    df['week'] = df.index.isocalendar().week
    df['day'] = df.index.isocalendar().day
    df['hour'] = df.index.hour
    df['subset'] = 0

    group_periods = ['year', 'week', 'day', 'hour']
    if group == 'hour':
        grp = df.groupby(group_periods)
    elif group == 'day':
        grp = df.groupby(group_periods[:-1])
    elif group == 'week':
        grp = df.groupby(group_periods[:-2])
    elif group == 'year':
        grp = df.groupby(group_periods[:-3])
    else:
        print('Group has to be one of: [year, week, day, hour]')

    # dates, dfs, = [], []
    # for group_cols, data in grp:
    #     dates.append(group_cols)
    #     dfs.append(data)

    # dfs_groups = []
    # new_df = pd.DataFrame()
    # group_df = pd.DataFrame()
    g = 0
    for i, (group_cols, data) in enumerate(grp):
        if i % groups_of == 0:
            g += 1
        df.loc[data.index, 'subset'] = g
    #     group_df = pd.concat([group_df, df], axis=1)
    #
    #     if i % groups_of == 0:
    #         dfs_groups.append(new_df)
    #         new_df = pd.DataFrame()
    #     new_df = pd.concat([new_df, df], axis=0)
    # dfs_groups.append(new_df)
    # del dfs_groups[0]

    return df  # dfs_groups


def simple_test_split(df, test_ratio=0.2):
    df['test'] = 0
    grp = df.groupby('subset')
    for i, (group_cols, data) in enumerate(grp):
        test_len = int(round(data.shape[0] * test_ratio, 0))
        df.loc[data.index[-test_len:], 'test'] = 1
    #
    # dfs_train, dfs_test = [], []
    # for data in dfs_groups:
    #     test_len = int(round(data.shape[0] * test_ratio, 0))
    #     dfs_train.append(data.iloc[:-test_len, :])
    #     dfs_test.append(data.iloc[-test_len:, :])
    # return dfs_train, dfs_test


def merge_train_test_groups(dfs_train, dfs_test):
    df_merged = pd.DataFrame()
    for i, (train, test) in enumerate(zip(dfs_train, dfs_test)):
        train_df, test_df = train.copy(), test.copy()
        train_df['test'], test_df['test'] = 0, 1
        train_df['group'], test_df['group'] = i, i
        df_merged = pd.concat([df_merged, train_df, test_df], axis=0)
    return df_merged


def update_subset_time_delta(df, time_thold=1000):
    if 'subset' in df.columns:
        df['diff_s'] = (pd.Series(df.index).shift(periods=1, fill_value=np.nan) - pd.Series(
            df.index)).dt.total_seconds().values
        time_mask = df['diff_s'] < -time_thold
        subset_mask = (df['subset'].shift(periods=1, fill_value=0) - df['subset']).eq(-1)
        mask = pd.concat([time_mask, subset_mask], axis=1)
        mask['step'] = mask['diff_s'] | mask['subset']
        ix_end = df.loc[mask['step'], :].index

        df['subset'] = 0
        i = 0
        for i in range(len(ix_end) - 1):
            df.loc[ix_end[i]:ix_end[i + 1], 'subset'] = i + 1
        df.loc[ix_end[i + 1]:, 'subset'] = i + 2
        df.drop('diff_s', axis=1, inplace=True)
    else:
        print('subset column not found')


def get_subsets(df_pp):
    if 'subset' not in df_pp.columns or 'test' not in df_pp.columns:
        print('subset or test column not found')
        return None
    else:
        grp = df_pp.groupby(['subset', 'test'])
        subsets = []
        train_lens, test_lens = [], []
        for i, ((n_ss, test), subset) in enumerate(grp):
            if test == 0:
                train_lens.append(subset.shape[0])
            else:
                test_lens.append(subset.shape[0])
            subsets.append((n_ss, test, subset))

        print('Total subsets: train={}, test={:.0f}'.format(len(train_lens), len(test_lens)))
        print('Average lengths: train={:.0f}, test={:.0f}'.format(np.mean(train_lens),
                                                                  np.mean(test_lens)))
        return subsets


def get_xy_from_subsets(subsets, features, min_dim, look_back=0):
    train_X, test_X = [], []
    for i, (n_ss, test, df_ss) in enumerate(subsets):
        # print(n_ss)
        if df_ss.shape[0] > min_dim and test == 0:
            dataset = df_ss[features].to_numpy()
            # X, y = model_func['xy_from_train'](dataset, *model_func['xy_args'])
            # train_XY.append((X, y))
            train_X.append(dataset)
        elif test == 1:
            # append look_back from last test subset
            if look_back > 0 and i > 0:
                _, t, df_ss_test = subsets[i - 1]
                if t == 0:
                    df_ss = pd.concat([df_ss_test.iloc[-look_back:, :], df_ss], axis=0)
            if df_ss.shape[0] > min_dim:
                dataset = df_ss[features[:-1]].to_numpy()
                test_X.append(dataset)

    return train_X, test_X


def get_xy(subsets, training_cfg, lookback=0, dim_f=1):
    '''
    :param subsets:
    :param training_cfg: specify y_var and features
    :param lookback: if 0, no data is appended to test data
    :param dim_f: multiplier for lookback
    :return: list of train and test subsets
    '''

    inst, y_var = training_cfg.get('inst', None), training_cfg['y_var']
    features = ([] if inst is None else get_inst_ohlc_names(inst)) + training_cfg['features']
    append_train_to_test = training_cfg.get('append_train_to_test', False)
    # n_seq, n_steps_in, n_steps_out = model_cfg.get('n_seq', 1), model_cfg['n_steps_in'], model_cfg['n_steps_out']
    dim_limit = lookback * dim_f #model_func['lookback'](model_cfg) * dim_f
    features = features + [y_var]
    look_back = dim_limit if append_train_to_test else 0
    train_X, test_X = get_xy_from_subsets(subsets, features, dim_limit, look_back)
    return train_X, test_X, features
