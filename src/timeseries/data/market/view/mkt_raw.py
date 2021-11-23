from timeseries.data.market.files.utils import load_market
from timeseries.experiments.market.preprocess.func import add_features
from timeseries.plotly.plot import plotly_time_series

if __name__ == '__main__':
    #%%
    data_cfg = {'inst': "ES", 'sampling': 'minute', 'suffix': "2012_1-2021_6",  'market': 'cme',
                'src_folder': "data"}

    df = load_market(data_cfg)

#%%
    resample = True
    data_from = '2011-12'
    data_to = '2021-12'
    df_ss = df.loc[data_from:data_to]
    df_ss = df_ss.resample('5T').last().dropna() if resample else df_ss

    add_features(df_ss,
                 macds=['ESc'],
                 rsis=['ESc'],
                 returns=['ESc'],
                 use_time_subset=True,
                 p0s=[12],
                 p1s=[26],
                 returns_from_ema=(3, True))

    # df_ss['macd_12_26'] = macd(df_ss['ESc'], p0=12, p1=26)
    # df_ss['macd_6_13'] = macd(df_ss['ESc'], p0=6, p1=13)
    # df_ss['ema_26'] = ema(df_ss['ESc'], period=26)
    # df_ss['ema_3'] = ema(df_ss['ESc'], period=3)
    # df_ss['rsi'] = rsi(df_ss['ESc'], periods=14, )
    # df_ss['ESc_r'] = ln_returns(df_ss['ESc'])
    # df_ss['ESc_e3_r'] = ln_returns(df_ss['ema_3'])


    #%%
    df_plot_ss = df_ss.head(10000)
    features = ['ESc', 'ESc_e3', 'ESc_r', 'ESc_e3_r']
    rows = [0, 0, 1, 1]

    plotly_time_series(df_plot_ss, features=features,
                       markers='lines+markers',
                       rows=rows)
    # plotly_ts_candles(df_plot, instrument=data_cfg['inst'], rows=[i for i in range(df_plot.shape[1] - 4)])
