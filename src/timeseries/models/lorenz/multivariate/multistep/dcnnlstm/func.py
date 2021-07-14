import math
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
from tensorflow.keras import initializers
from algorithms.wavenet.func import dcnn_1st_layer, dcnn_layer
from timeseries.plotly.plot import plot_history

from tensorflow import keras
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Reshape, ConvLSTM2D, LSTM, Dense, TimeDistributed, Conv1D, MaxPooling1D, Flatten
from timeseries.models.lorenz.functions.dataprep import step_feature_one_step_xy_from_uv, \
    step_feature_one_step_xy_from_mv, step_feature_multi_step_xy_from_mv, feature_multi_step_xy_from_mv
from numpy import array

import time


def dcnnlstm_multi_step_mv_fit(train, cfg, plot_hist=False, verbose=0):
    # unpack architectures
    n_steps_in, n_steps_out = cfg['n_steps_in'], cfg['n_steps_out']
    n_epochs, n_batch = cfg['n_epochs'], cfg['n_batch']
    # prepare data
    X, y = feature_multi_step_xy_from_mv(train, n_steps_in, n_steps_out)
    n_features = X.shape[2]
    # define model
    model = dcnnlstm_multi_step_mv_build(cfg, n_features)
    # fit
    start_time = time.time()
    history = model.fit(X, y, epochs=n_epochs, batch_size=n_batch, verbose=verbose)
    train_time = round((time.time() - start_time), 2)
    if plot_hist:
        plot_history(history, title='CNN-LSTM: ' + str(cfg), plot_title=True)
    return model, train_time, history.history['loss'][-1]


def dcnnlstm_multi_step_mv_build(cfg, n_features):
    n_steps_in, n_steps_out, n_filters = cfg['n_steps_in'], cfg['n_steps_out'], cfg['n_filters']
    n_kernel, n_epochs, n_batch = cfg['n_kernel'], cfg['n_epochs'], cfg['n_batch']
    n_layers, reg, n_nodes = cfg['n_layers'], cfg['reg'], cfg['n_nodes']
    input_shape = (n_steps_in, n_features)
    assert n_steps_in > 2 ** (n_layers - 1) * n_kernel

    stddev = math.sqrt(2 / (n_kernel * n_filters))

    # ARCHITECTURE
    sequence = keras.layers.Input(shape=input_shape, name='sequence')
    x = dcnn_1st_layer(n_filters, n_kernel, 1, '0', reg=reg)(sequence)
    for layer in range(1, n_layers):
        x = dcnn_layer(n_filters, n_kernel, 2 ** layer, str(layer), reg=reg)(x)
    # out_conv = keras.layers.Conv1D(n_filters, 1,
    #                                padding='same', use_bias=True,
    #                                activation='relu', name='conv1x1')(x)
    lstm = LSTM(n_nodes, activation='relu')(x)
    preoutput = keras.layers.Flatten()(lstm)
    preoutput = keras.layers.Dense(n_steps_out * 2, kernel_regularizer=reg, name='preoutput',
                                   kernel_initializer=initializers.RandomNormal(stddev=stddev))(preoutput)
    output = keras.layers.Dense(n_steps_out, kernel_regularizer=reg, name='output',
                                kernel_initializer=initializers.RandomNormal(stddev=stddev))(preoutput)
    model = keras.models.Model(inputs=[sequence], outputs=output)
    model.compile(loss='mse', optimizer='adam')
    return model


def dcnnlstm_multi_step_mv_predict(model, history, cfg, steps=1):
    # unpack architectures
    n_steps = cfg['n_steps_in']
    n_features = history.shape[1]
    # prepare data
    x_input = array(history[-n_steps:]).reshape(1, n_steps, n_features)
    # x_input = tf.convert_to_tensor(x_input, dtype=tf.float64)
    # forecast
    yhat = model.predict(x_input, verbose=0)
    return yhat[0]


def dcnnlstm_get_multi_step_mv_funcs():
    return [dcnnlstm_multi_step_mv_predict, dcnnlstm_multi_step_mv_fit,
            dcnnlstm_multi_step_mv_build]


def get_dcnnlstm_steps_cfgs(in_steps_range, n_seq_range, k_range):
    cfgs_steps_in = []
    for s in range(*in_steps_range):
        for q in range(*n_seq_range):
            for k in range(k_range[0], min(s - 3, k_range[1])):
                cfgs_steps_in.append((s, q, k))
    x = np.array(cfgs_steps_in)
    return {'n_steps_in': x[:, 0].tolist(), 'n_seq': x[:, 1].tolist(), "n_kernel": x[:, 2].tolist()}