import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from timeseries.models.lorenz.functions.dataprep import one_step_xy_from_mv, multi_step_xy_from_mv
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from numpy import array
import numpy as np


def mlp_multi_step_mv_fit(train, cfg):
    # unpack architectures
    n_steps_in, n_steps_out = cfg['n_steps_in'], cfg['n_steps_out']
    n_nodes, n_epochs, n_batch = cfg['n_nodes'], cfg['n_epochs'], cfg['n_batch']
    # prepare data
    X, y = multi_step_xy_from_mv(train, n_steps_in, n_steps_out)
    n_input = X.shape[1]
    # define model
    model = Sequential()
    model.add(Dense(100, activation='relu', input_dim=n_input))
    model.add(Dense(n_steps_out))
    model.compile(optimizer='adam', loss='mse')
    # fit
    model.fit(X, y, epochs=n_epochs, batch_size=n_batch, verbose=0)
    return model


# forecast with a pre-fit model
def mlp_multi_step_mv_predict(model, history, cfg, steps=1):
    # unpack architectures
    n_steps_in = cfg['n_steps_in']
    n_input = n_steps_in * history.shape[1]
    # prepare data
    x_input = array(history[-n_steps_in:]).reshape(1, n_input)
    # forecast
    yhat = model.predict(x_input, verbose=0)
    return yhat[0]