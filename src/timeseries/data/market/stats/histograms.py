import os

from timeseries.data.market.files.utils import load_market
from timeseries.models.lorenz.functions.preprocessing import preprocess_x

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
from timeseries.models.utils.config import unpack_in_cfg
from timeseries.plotly.plot import plotly_ts_candles
import seaborn as sns
import matplotlib.pyplot as plt

if __name__ == '__main__':
    in_cfg = {'steps': 3, 'save_results': False, 'verbose': 1, 'plot_title': True,
              'image_folder': 'explore_img', 'results_folder': 'results'}
    input_cfg = {'preprocess': True}

    data_cfg = {'inst': "ES", 'suffix': "2012-2020", 'sampling': 'minute',
                'src_folder': "data", 'market': 'cme'}

    # %% LOAD DATA
    es, features_es = load_market(data_cfg)
    data_from = '2020-01-12 08:30:00'
    data_to = '2020-03-28 15:30:00'
    df = es.loc[data_from:data_to]

    # %%
    returns, ss = preprocess_x(df['ESc'].values.reshape(-1, 1), detrend='ln_return')

    #%%
    fig, axes = plt.subplots(3, 1, figsize=(15, 9))
    axes[0].plot(df['ESc'].values)
    axes[1].plot(returns)
    axes[2].plot(df.index)
    plt.show()

