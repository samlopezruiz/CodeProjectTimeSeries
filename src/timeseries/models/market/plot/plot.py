from timeseries.models.market.split.func import merge_train_test_groups
from timeseries.plotly.plot import plotly_ts_candles, plotly_ts_regime


def plot_mkt_candles(df, inst, features=None, resample=False, period='90T', ts_height=0.6, template='plotly_white'):
    df_plot = df.resample(period).last() if resample else df
    features = list(df.columns[4:]) if features is None else features
    plotly_ts_candles(df_plot, features=features, instrument=inst, adjust_height=(True, ts_height),
                      template=template, rows=[i for i in range(len(features))])


def plot_train_test_groups(df, split_cfg=None, plot_last=None, regime_col='test', features=['ESc', 'subset'],
                           resample=False, period='90T', template='plotly_dark', save=False, file_path=None):
    # df_merged = merge_train_test_groups(dfs_train, dfs_test)
    if plot_last is not None:
        df = df.iloc[-plot_last:, :]
    df_plot = df.resample(period).last() if resample else df
    title = 'SPLIT CFG: {}'.format(str(split_cfg)) if split_cfg is not None else "SPLIT GROUPS"
    plotly_ts_regime(df_plot, features=features, regime_col=regime_col, title=title, adjust_height=(True, 0.8),
                     template=template, rows=[i for i in range(len(features))], save=save, file_path=file_path)
