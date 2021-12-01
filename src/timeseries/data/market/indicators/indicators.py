import os

from timeseries.data.market.files.utils import load_market
from timeseries.preprocessing.func import atr, macd, ema

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from timeseries.data.market.utils.time import insert_weekend
from timeseries.plotly.plot import plotly_time_series
#%%





if __name__ == '__main__':
    #%%
    name = "DIRECTIONAL CHANGE REGIMES"
    in_cfg = {'steps': 3, 'save_results': False, 'verbose': 1, 'plot_title': True,
              'image_folder': 'regime_img', 'results_folder': 'results'}
    input_cfg = {'preprocess': True}

    data_cfg = {'inst': "ES", 'suffix': "2012-2020", 'sampling': 'day',
                'src_folder': "data", 'market': 'cme'}

    # %% LOAD DATA
    es, features_es = load_market(data_cfg)
    insert_weekend(es)

    #%%
    data_from = '2011-12-01 08:30:00'
    data_to = '2021-12-30 15:30:00'
    df = es.loc[data_from:data_to].copy()

    inst = 'ES'

    #%%
    df['ema14'] = ema(df['ESc'], 14)
    df['atr2'] = atr(df, 'ES')
    df['macd'] = macd(df['ESc'], p0=12, p1=9)

    #%%
    plotly_time_series(df, features=['ESc', 'ema14', 'atr2', 'macd'], rows=[0, 0, 1, 2],
                       title=name, markers='lines+markers', markersize=2)




