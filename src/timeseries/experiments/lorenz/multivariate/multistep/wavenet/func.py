import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from algorithms.wavenet.func import dcnn_build, wavenet_build, wavenet_build2
from timeseries.plotly.plot import plot_history
from timeseries.experiments.lorenz.functions.dataprep import feature_multi_step_xy_from_mv
from numpy import array
import time


def wavenet_multi_step_mv_fit(train, cfg, plot_hist=False, verbose=0):
    # unpack architectures
    n_steps_in, n_steps_out = cfg['n_steps_in'], cfg['n_steps_out']
    n_epochs, n_batch = cfg['n_epochs'], cfg['n_batch']
    # prepare data
    X, y = feature_multi_step_xy_from_mv(train, n_steps_in, n_steps_out)
    n_features = X.shape[2]
    # define model
    model = wavenet_build(cfg, n_features)
    model.compile(loss='mse', optimizer='adam')
    if verbose > 0:
        print('No. of params: {}'.format(model.count_params()))
    # fit
    start_time = time.time()
    X = tf.convert_to_tensor(X, dtype=tf.float64)
    y = tf.convert_to_tensor(y, dtype=tf.float64)
    history = model.fit(X, y, epochs=n_epochs, batch_size=n_batch, verbose=verbose)
    train_time = round((time.time() - start_time), 2)
    # summarize history for accuracy
    if plot_hist:
        plot_history(history, title='D-CNN: '+str(cfg), plot_title=True)
    return model, train_time, history.history['loss'][-1]


def wavenet_multi_step_mv_predict(model, history, cfg, steps=1):
    # unpack architectures
    n_steps = cfg['n_steps_in']
    n_features = history.shape[1]
    # prepare data
    x_input = array(history[-n_steps:]).reshape(1, n_steps, n_features)
    x_input = tf.convert_to_tensor(x_input, dtype=tf.float64)
    # forecast
    yhat = model.predict(x_input, verbose=0)
    return yhat[0]


def wavenet_get_multi_step_mv_funcs():
    return [wavenet_multi_step_mv_predict, wavenet_multi_step_mv_fit, wavenet_build]

def wavenet_get_functions2():
    return [wavenet_multi_step_mv_predict, wavenet_multi_step_mv_fit2, wavenet_build]

def wavenet_multi_step_mv_fit2(train, cfg, plot_hist=False, verbose=0):
    # unpack architectures
    n_steps_in, n_steps_out = cfg['n_steps_in'], cfg['n_steps_out']
    n_epochs, n_batch = cfg['n_epochs'], cfg['n_batch']
    # prepare data
    X, y = feature_multi_step_xy_from_mv(train, n_steps_in, n_steps_out)
    n_features = X.shape[2]
    # define model
    model = wavenet_build2(cfg, n_features)
    model.compile(loss='mse', optimizer='adam')
    if verbose > 0:
        print('No. of params: {}'.format(model.count_params()))
    # fit
    start_time = time.time()
    X = tf.convert_to_tensor(X, dtype=tf.float64)
    y = tf.convert_to_tensor(y, dtype=tf.float64)
    history = model.fit(X, y, epochs=n_epochs, batch_size=n_batch, verbose=verbose)
    train_time = round((time.time() - start_time), 2)
    # summarize history for accuracy
    if plot_hist:
        plot_history(history, title='D-CNN: '+str(cfg), plot_title=True)
    return model, train_time, history.history['loss'][-1]