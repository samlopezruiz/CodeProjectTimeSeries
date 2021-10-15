import os

import pandas as pd

from timeseries.plotly.plot import plotly_time_series

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from algorithms.nnhmm.func import nnhmm_fit
from timeseries.data.lorenz.lorenz import regime_multivariate_lorenz
from timeseries.experiments.market.multivariate.architectures.cnnlstm import cnnlstm_func
from timeseries.experiments.market.utils.preprocessing import preprocess, reconstruct
import numpy as np

if __name__ == '__main__':
    # %% DATA
    input_cfg = {"variate": "multi", "granularity": 5, "noise": False, 'preprocess': True,
                 'trend': False, 'detrend': 'ln_return'}
    lorenz_df, train, test, t_train, t_test, hidden_states = regime_multivariate_lorenz(input_cfg)
    # plotly_time_series(lorenz_df, features=['x', 'y', 'z'], rows=list(range(3)), markers='lines')
    train_x, train_reg_prob = train
    test_x, test_reg_prob = test
    train_pp, test_pp, train_reg_prob, test_reg_prob, ss = preprocess(input_cfg, train_x, test_x,
                                                                      train_reg_prob, test_reg_prob)

    #%%
    test_y = test_x[:, -1]
    test_y_pp = test_pp[:, -1]
    y_reconst = reconstruct(test_y_pp, input_cfg, 6, test=test_y, ss=ss)

    df = pd.DataFrame(np.transpose([test_y, y_reconst]), columns=['true', 'reconstructed'])
    plotly_time_series(df)
