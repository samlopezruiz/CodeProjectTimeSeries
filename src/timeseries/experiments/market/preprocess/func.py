import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

from timeseries.experiments.market.split.func import get_train_features, s_threshold
from timeseries.preprocessing.func import ln_returns, macd, rsi, ema


def append_timediff_subsets(df, time_diff_cfg, new_col='time_subset', plot_=False):
    diff = pd.Series(df.index).diff()
    diff_s = diff.dt.total_seconds().fillna(0)
    if plot_:
        sns.histplot(diff_s / 1000, bins=100)
        plt.show()

    s_thold = s_threshold(time_diff_cfg)
    keep = (diff_s <= s_thold).astype(int)

    print('Good intervals: {}/{}  {}% of data'.format(sum(keep), diff_s.shape[0],
                                                      round(sum(keep) * 100 / diff_s.shape[0], 2)))

    df[new_col] = 0
    if sum(keep) != diff_s.shape[0]:
        df_time = pd.DataFrame(keep.values, index=df.index, columns=['time_switch'])
        time_subsets_ix = df_time.loc[df_time['time_switch'] == 0, ['time_switch']].index
        i = 0
        for i in range(1, len(time_subsets_ix)):
            df.loc[time_subsets_ix[i - 1]:time_subsets_ix[i], new_col] = i
        df.loc[time_subsets_ix[i]:, new_col] = i + 1


def add_features(df,
                 macds=None,
                 rsis=None,
                 returns=None,
                 use_time_subset=True,
                 p0s=[12],
                 p1s=[26],
                 returns_from_ema=(3, False)):

    if 'time_subset' in df.columns and use_time_subset:
        df_grp = df.groupby('time_subset')
        for i, (group_cols, df_subset) in enumerate(df_grp):
            if returns is not None:
                for var in returns:
                    if returns_from_ema[1]:
                        p = int(returns_from_ema[0])
                        df.loc[df_subset.index, '{}_e{}'.format(var, p)] = ema(df_subset[var], period=p)
                        df.loc[df_subset.index, '{}_e{}_r'.format(var, p)] = ln_returns(df.loc[df_subset.index, '{}_e{}'.format(var, p)])

                    df.loc[df_subset.index, var + '_r'] = ln_returns(df_subset[var])

            if macds is not None:
                for var in macds:
                    for p0, p1 in zip(p0s, p1s):
                        df.loc[df_subset.index, '{}_macd_{}_{}'.format(var, p0, p1)] = macd(df_subset[var], p0=p0, p1=p1)

            if rsis is not None:
                for var in rsis:
                    df.loc[df_subset.index, '{}_rsi'.format(var)] = rsi(df_subset[var], periods=14, ema=True)
    else:
        if returns is not None:
            for var in returns:
                if returns_from_ema[1]:
                    p = int(returns_from_ema[0])
                    df['{}_e{}'.format(var, p)] = ema(df[var], period=p)
                    df['{}_e{}_r'.format(var, p)] = ln_returns(df['{}_e{}'.format(var, p)])

                df[var + '_r'] = ln_returns(df[var])

        if macds is not None:
            for var in macds:
                for p0, p1 in zip(p0s, p1s):
                    df['{}_macd_{}_{}'.format(var, p0, p1)] = macd(df[var], p0=p0, p1=p1)

        if rsis is not None:
            for var in rsis:
                df['{}_rsi'.format(var)] = rsi(df[var], periods=14, ema=True)

    if returns is not None:
        for var in returns:
            df.loc[:, var + '_r'].fillna(0, inplace=True)

            if returns_from_ema[1]:
                p = int(returns_from_ema[0])
                df.loc[:, '{}_e{}_r'.format(var, p)].fillna(0, inplace=True)


def scale_df(df, training_cfg):
    train_features = get_train_features(training_cfg)
    if training_cfg['scale']:
        if 'test' in df.columns:
            ss = StandardScaler()
            ss.fit(df.loc[df.loc[:, 'test'] == 0, train_features])
            df_scaled = pd.DataFrame(ss.transform(df.loc[:, train_features]),
                                     columns=train_features, index=df.index)
            return df_scaled, ss, train_features
        else:
            ss = StandardScaler()
            ss.fit(df.loc[:, train_features])
            df_scaled = pd.DataFrame(ss.transform(df.loc[:, train_features]),
                                     columns=train_features, index=df.index)
            return df_scaled, ss, train_features
    else:
        return df.loc[:, train_features], None, train_features
