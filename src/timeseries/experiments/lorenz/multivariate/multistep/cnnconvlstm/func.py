import os
import numpy as np
from timeseries.plotly.plot import plot_history
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import ConvLSTM2D, LSTM, Dense, TimeDistributed, Conv1D, MaxPooling1D, Flatten
from timeseries.experiments.lorenz.functions.dataprep import step_feature_one_step_xy_from_uv, \
    step_feature_one_step_xy_from_mv, step_feature_multi_step_xy_from_mv
from numpy import array
import time

def cnnconvlstm_multi_step_mv_fit(train, cfg, plot_hist=False, verbose=0):
    # unpack architectures
    n_seq, n_steps, n_steps_out, n_filters = cfg['n_seq'], cfg['n_steps_in'], cfg['n_steps_out'], cfg['n_filters']
    n_kernel, n_nodes, n_epochs, n_batch = cfg['n_kernel'], cfg['n_nodes'], cfg['n_epochs'], cfg['n_batch']
    n_input = n_seq * n_steps

    # prepare data
    X, y = step_feature_multi_step_xy_from_mv(train, n_input, n_steps_out, n_seq)
    n_features = X.shape[3]
    # define model
    model = cnnconvlstm_multi_step_mv_build(cfg, n_features)
    # fit
    start_time = time.time()
    history = model.fit(X, y, epochs=n_epochs, batch_size=n_batch, verbose=verbose)
    train_time = round((time.time() - start_time), 2)
    if plot_hist:
        plot_history(history, title='CNN-LSTM: ' + str(cfg), plot_title=True)
    return model, train_time, history.history['loss'][-1]

def cnnconvlstm_multi_step_mv_fit_tf(train, cfg, plot_hist=False, verbose=0):
    # unpack architectures
    n_seq, n_steps, n_steps_out, n_filters = cfg['n_seq'], cfg['n_steps_in'], cfg['n_steps_out'], cfg['n_filters']
    n_kernel, n_nodes, n_epochs, n_batch = cfg['n_kernel'], cfg['n_nodes'], cfg['n_epochs'], cfg['n_batch']
    n_input = n_seq * n_steps

    # prepare data
    X, y = step_feature_multi_step_xy_from_mv(train, n_input, n_steps_out, n_seq)
    n_features = X.shape[3]
    # define model
    model = cnnconvlstm_multi_step_mv_build(cfg, n_features)
    # fit
    start_time = time.time()
    X = tf.convert_to_tensor(X, dtype=tf.float64)
    y = tf.convert_to_tensor(y, dtype=tf.float64)
    history = model.fit(X, y, epochs=n_epochs, batch_size=n_batch, verbose=verbose)
    train_time = round((time.time() - start_time), 2)
    if plot_hist:
        plot_history(history, title='CNN-LSTM: ' + str(cfg), plot_title=True)
    return model, train_time, history.history['loss'][-1]


def cnnconvlstm_multi_step_mv_build(cfg, n_features):
    n_steps, n_steps_out, n_filters = cfg['n_steps_in'], cfg['n_steps_out'], cfg['n_filters']
    n_kernel, n_nodes = cfg['n_kernel'], cfg['n_nodes']
    model = Sequential()
    model.add(TimeDistributed(Conv1D(n_filters, n_kernel, activation='relu', input_shape=(None, n_steps, n_features))))
    model.add(TimeDistributed(Conv1D(n_filters, n_kernel, activation='relu')))
    model.add(TimeDistributed(MaxPooling1D()))
    model.add(TimeDistributed(Flatten()))
    model.add(LSTM(n_nodes, activation='relu'))
    # model.add(ConvLSTM2D(n_filters, (1, n_kernel), activation='relu'))
    model.add(Flatten())
    model.add(Dense(n_nodes, activation='relu'))
    model.add(Dense(n_steps_out))
    model.compile(loss='mse', optimizer='adam')
    return model


# forecast with a pre-fit model
def cnnconvlstm_multi_step_mv_predict_tf(model, history, cfg, steps=1):
    # unpack architectures
    n_seq, n_steps = cfg['n_seq'], cfg['n_steps_in']
    n_features = history.shape[1]
    n_input = n_seq * n_steps
    # prepare data
    x_input = array(history[-n_input:]).reshape((1, n_seq, n_steps, n_features))
    x_input = tf.convert_to_tensor(x_input, dtype=tf.float64)
    # forecast
    yhat = model.predict(x_input, verbose=0)
    return yhat[0]


def cnnconvlstm_multi_step_mv_predict(model, history, cfg, steps=1):
    # unpack architectures
    n_seq, n_steps = cfg['n_seq'], cfg['n_steps_in']
    n_features = history.shape[1]
    n_input = n_seq * n_steps
    # prepare data
    x_input = array(history[-n_input:]).reshape((1, n_seq, n_steps, n_features))
    # x_input = tf.convert_to_tensor(x_input, dtype=tf.float64)
    # forecast
    yhat = model.predict(x_input, verbose=0)
    return yhat[0]


def cnnconvlstm_get_multi_step_mv_funcs():
    return [cnnconvlstm_multi_step_mv_predict, cnnconvlstm_multi_step_mv_fit,
            cnnconvlstm_multi_step_mv_build]


def get_cnnconvlstm_steps_cfgs(in_steps_range, n_seq_range, k_range):
    cfgs_steps_in = []
    for s in range(*in_steps_range):
        for q in range(*n_seq_range):
            for k in range(k_range[0], min(s - 3, k_range[1])):
                cfgs_steps_in.append((s, q, k))
    x = np.array(cfgs_steps_in)
    return {'n_steps_in': x[:, 0].tolist(), 'n_seq': x[:, 1].tolist(), "n_kernel": x[:, 2].tolist()}