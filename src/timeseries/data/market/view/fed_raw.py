from timeseries.data.market.files.utils import load_market
from timeseries.plotly.plot import plotly_time_series

if __name__ == '__main__':
    data_cfg = {'inst': "FED", 'sampling': 'day', 'suffix': "2012_1-2021_6",  'market': 'fed',
                'src_folder': "data"}

    df, features = load_market(data_cfg)

#%%
    resample = False
    data_from = '2011-12'
    data_to = '2021-12'
    df_ss = df.loc[data_from:data_to]
    df_plot = df_ss.resample('90T').last() if resample else df_ss
    plotly_time_series(df_plot, markers='lines+markers', rows=[i for i in range(df_plot.shape[1])])
