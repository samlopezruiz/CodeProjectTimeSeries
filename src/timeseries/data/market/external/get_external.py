import pandas as pd
import pandas_datareader.data as web
import sklearn.mixture as mix

import numpy as np
import scipy.stats as scs

import matplotlib as mpl
from matplotlib import cm
import matplotlib.pyplot as plt
from matplotlib.dates import YearLocator, MonthLocator

import seaborn as sns
from tqdm import tqdm

from timeseries.data.market.files.utils import save_df

if __name__ == '__main__':
    #%%
    mkt_insts = ['^VIX']
    fred_insts = ['TEDRATE', 'T10Y2Y', 'T10Y3M', 'DGS10']
    start = pd.to_datetime('2012-01-01')
    end = pd.datetime.today()

    mkt = web.DataReader(mkt_insts, 'yahoo', start, end)['Adj Close'].dropna()

    data = (web.DataReader(fred_insts, 'fred', start, end)
            .join(mkt, how='inner')
            .dropna()
            )
    data.index.names = ['datetime']
    print(data.head())

    #%%
    data_cfg = {'inst': "FED", 'sampling': 'day',
                'src_folder': "data", 'market': 'fed'}
    save_df(data, data_cfg)
