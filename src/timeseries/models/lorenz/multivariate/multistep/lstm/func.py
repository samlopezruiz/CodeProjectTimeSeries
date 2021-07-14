import os
import time

from timeseries.plotly.plot import plot_history

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from timeseries.models.lorenz.functions.dataprep import split_mv_seq_multi_step
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from numpy import array


def lstm_multi_step_mv_build(cfg, n_features):
    n_input, n_steps_out, n_nodes = cfg['n_steps_in'], cfg['n_steps_out'], cfg['n_nodes']
    model = Sequential()
    # model.add(LSTM(50, activation='relu', return_sequences=True, input_shape=(n_input, 1)))
    # model.add(LSTM(50, activation='relu'))
    model.add(LSTM(n_nodes, activation='relu', input_shape=(n_input, n_features), return_sequences=False))
    model.add(Dense((n_nodes + n_steps_out) // 2 + n_steps_out, activation='relu'))
    model.add(Dense(n_steps_out))
    model.compile(loss='mse', optimizer='adam')
    return model


def lstm_multi_step_mv_fit(train, cfg, plot_hist=False, verbose=0):
    n_input, n_steps_out = cfg['n_steps_in'],  cfg['n_steps_out']
    n_epochs, n_batch = cfg['n_epochs'], cfg['n_batch']
    # prepare data
    X, y = split_mv_seq_multi_step(train, n_input, n_steps_out)
    n_features = X.shape[2]
    # define model
    model = lstm_multi_step_mv_build(cfg, n_features)
    # fit
    start_time = time.time()
    history = model.fit(X, y, epochs=n_epochs, batch_size=n_batch, verbose=verbose)
    train_time = round((time.time() - start_time), 2)
    if plot_hist:
        plot_history(history, title='CNN-LSTM: ' + str(cfg), plot_title=True)
    return model, train_time, history.history['loss'][-1]


# forecast with a pre-fit model
def lstm_multi_step_mv_predict(model, history, cfg, steps=1):
    # unpack architectures
    n_input = cfg['n_steps_in']
    n_features = history.shape[1]
    # prepare data
    x_input = array(history[-n_input:]).reshape((1, n_input, n_features))
    # forecast
    yhat = model.predict(x_input, verbose=0)
    return yhat[0]


def lstm_get_multi_step_mv_funcs():
    return [lstm_multi_step_mv_predict, lstm_multi_step_mv_fit, lstm_multi_step_mv_build]
