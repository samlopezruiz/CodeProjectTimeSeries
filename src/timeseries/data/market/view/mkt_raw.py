from algorithms.dchange.func import get_regimes
from algorithms.hmm.func import fitHMM
from timeseries.data.market.files.utils import load_market
from timeseries.experiments.market.preprocess.func import add_features
from timeseries.plotly.plot import plotly_ts_candles, plotly_time_series
from timeseries.preprocessing.func import macd, ema, rsi
from timeseries.utils.dataframes import append_to_df




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
    df_ss = df_ss.resample('60T').last().dropna() if resample else df_ss
    df_ss['macd_12_26'] = macd(df_ss['ESc'], p0=12, p1=26)
    df_ss['macd_6_13'] = macd(df_ss['ESc'], p0=6, p1=13)
    df_ss['ema_26'] = ema(df_ss['ESc'], period=26)
    df_ss['ema_12'] = ema(df_ss['ESc'], period=12)
    df_ss['rsi'] = rsi(df_ss, periods=14, col='ESc')


#%% RSI
    # df_plot['rsi'] = rsi(df_plot, periods=14, col='ESc')

    #%%
    df_plot_ss = df_ss.head(10000)
    features = ['ESc', 'ema_26', 'ema_12', 'macd_12_26', 'macd_6_13', 'rsi']
    rows = [0, 0, 0, 1, 1, 2]
    plotly_time_series(df_plot_ss, features=features,
                       markers='lines+markers',
                       rows=rows)
    # plotly_ts_candles(df_plot, instrument=data_cfg['inst'], rows=[i for i in range(df_plot.shape[1] - 4)])
