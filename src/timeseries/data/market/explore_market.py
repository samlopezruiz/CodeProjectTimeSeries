import os

import numpy as np

from timeseries.data.market.market_files import load_market

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from timeseries.models.utils.config import unpack_in_cfg
from timeseries.plotly.plot import plotly_ts_candles

if __name__ == '__main__':
    in_cfg = {'steps': 3, 'save_results': False, 'verbose': 1, 'plot_title': True,
              'image_folder': 'explore_img', 'results_folder': 'results'}

    data_cfg = {'inst': "ES", 'suffix': "2012-2020", 'sampling': 'minute',
                'src_folder': "data", 'market': 'cme'}

    # %% LOAD DATA
    es, features_es = load_market(data_cfg)

    # %%
    data_from = '2020-01-12 08:30:00'
    data_to = '2020-03-28 15:30:00'
    es_ss = es.loc[data_from:data_to]

    image_folder, plot_hist, plot_title, save_results, results_folder, verbose = unpack_in_cfg(in_cfg)
    rows = np.arange(4) + 1
    plotly_ts_candles(es_ss, instrument='ES', rows=rows, plot_title=plot_title, label_scale=1,
                      file_path=[image_folder, 'plot'], save=save_results, size=(1980, 1080))

