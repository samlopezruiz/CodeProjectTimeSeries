import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow.python.keras.layers import Conv1D
import tensorflow as tf
from timeseries.plotly.plot import plot_history
from timeseries.experiments.lorenz.functions.dataprep import row_col_multi_step_xy_from_mv
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import ConvLSTM2D, Dense, Flatten
from numpy import array
import time


def convlstm_multi_step_mv_fit(train, cfg, plot_hist=False, verbose=0):
    # unpack architectures
    n_seq, n_steps, n_steps_out, n_filters = cfg['n_seq'], cfg['n_steps_in'], cfg['n_steps_out'], cfg['n_filters']
    n_kernel, n_nodes, n_epochs, n_batch = cfg['n_kernel'], cfg['n_nodes'], cfg['n_epochs'], cfg['n_batch']
    n_input = n_seq * n_steps
    # prepare data
    X, y = row_col_multi_step_xy_from_mv(train, n_input, n_steps_out, n_seq)
    n_features = X.shape[4]
    # define model
    model = convlstm_multi_step_mv_build(cfg, n_features)
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
        plot_history(history, title='CONV LSTM: '+str(cfg), plot_title=True)
    return model, train_time, history.history['loss'][-1]


def convlstm_multi_step_mv_build(cfg, n_features):
    n_seq, n_steps, n_steps_out, n_filters = cfg['n_seq'], cfg['n_steps_in'], cfg['n_steps_out'], cfg['n_filters']
    n_kernel, n_nodes = cfg['n_kernel'], cfg['n_nodes']

    model = Sequential()
    model.add(ConvLSTM2D(n_filters, (1, n_kernel), activation='relu', input_shape=(n_seq, 1, n_steps, n_features)))
    # model.add(Conv1D(n_filters // 2, 1, activation='relu'))
    model.add(Flatten())
    model.add(Dense(n_nodes, activation='relu'))
    model.add(Dense(n_steps_out))
    model.compile(loss='mse', optimizer='adam')
    return model


# forecast with a pre-fit model
def convlstm_multi_step_mv_predict(model, history, cfg, steps=1):
    # unpack architectures
    n_seq, n_steps = cfg['n_seq'], cfg['n_steps_in']
    n_input = n_seq * n_steps
    n_features = history.shape[1]
    # prepare data
    x_input = array(history[-n_input:]).reshape((1, n_seq, 1, n_steps, n_features))
    x_input = tf.convert_to_tensor(x_input, dtype=tf.float64)
    # forecast
    yhat = model.predict(x_input, verbose=0)
    return yhat[0]


def convlstm_get_multi_step_mv_funcs():
    return [convlstm_multi_step_mv_predict, convlstm_multi_step_mv_fit, convlstm_multi_step_mv_build]