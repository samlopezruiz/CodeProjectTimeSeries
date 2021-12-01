import os
import time

from timeseries.plotly.plot import plot_history

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf


from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense, TimeDistributed, Conv1D, MaxPooling1D, Flatten
from timeseries.experiments.lorenz.functions.dataprep import step_feature_multi_step_xy_from_uv
from numpy import array


def cnnlstm_multi_step_uv_fit(train, cfg, plot_hist=False, verbose=0):
    # unpack architectures
    n_seq, n_steps, n_steps_out, n_filters = cfg['n_seq'], cfg['n_steps_in'], cfg['n_steps_out'],cfg['n_filters']
    n_kernel, n_nodes, n_epochs, n_batch = cfg['n_kernel'], cfg['n_nodes'], cfg['n_epochs'], cfg['n_batch']
    n_input = n_seq * n_steps

    # prepare data
    X, y = step_feature_multi_step_xy_from_uv(train, n_input, n_steps_out, 1, n_seq)
    # define model
    model = Sequential()
    model.add(TimeDistributed(Conv1D(n_filters, n_kernel, activation='relu', input_shape=(None, n_steps, 1))))
    model.add(TimeDistributed(Conv1D(n_filters, n_kernel, activation='relu')))
    model.add(TimeDistributed(MaxPooling1D()))
    model.add(TimeDistributed(Flatten()))
    model.add(LSTM(n_nodes, activation='relu'))
    model.add(Dense(n_nodes, activation='relu'))
    model.add(Dense(n_steps_out))
    model.compile(loss='mse', optimizer='adam')
    start_time = time.time()
    X = tf.convert_to_tensor(X, dtype=tf.float64)
    y = tf.convert_to_tensor(y, dtype=tf.float64)
    history = model.fit(X, y, epochs=n_epochs, batch_size=n_batch, verbose=verbose)
    train_time = round((time.time() - start_time), 2)
    if plot_hist:
        plot_history(history, title='CNN-LSTM: ' + str(cfg), plot_title=True)
    return model, train_time, history.history['loss'][-1]


# forecast with a pre-fit model
def cnnlstm_multi_step_uv_predict(model, history, cfg, steps=1):
    # unpack architectures
    n_seq, n_steps = cfg['n_seq'], cfg['n_steps_in']
    n_input = n_seq * n_steps
    # prepare data
    x_input = array(history[-n_input:]).reshape((1, n_seq, n_steps, 1))
    # forecast
    yhat = model.predict(x_input, verbose=0)
    return yhat[0]


def cnnlstm_get_multi_step_uv_funcs():
    return [cnnlstm_multi_step_uv_predict, cnnlstm_multi_step_uv_fit]
