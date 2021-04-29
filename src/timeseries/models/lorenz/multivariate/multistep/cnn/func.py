import os
from timeseries.plotly.plot import plot_history
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from timeseries.models.lorenz.functions.dataprep import feature_one_step_xy_from_mv, feature_multi_step_xy_from_mv
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv1D, MaxPooling1D
from numpy import array
import time


def cnn_multi_step_mv_fit(train, cfg, plot_hist=False, verbose=0):
    # unpack config
    n_steps_in, n_steps_out = cfg['n_steps_in'], cfg['n_steps_out']
    n_epochs, n_batch = cfg['n_epochs'], cfg['n_batch']
    # prepare data
    X, y = feature_multi_step_xy_from_mv(train, n_steps_in, n_steps_out)
    n_features = X.shape[2]
    # define model
    model = cnn_multi_step_mv_build(cfg, n_features)
    # fit
    start_time = time.time()
    history = model.fit(X, y, epochs=n_epochs, batch_size=n_batch, verbose=verbose)
    train_time = round((time.time() - start_time), 2)
    if plot_hist:
        plot_history(history, title='CNN: ' + str(cfg), plot_title=True)
    return model, train_time, history.history['loss'][-1]


def cnn_multi_step_mv_build(cfg, n_features):
    n_steps_in, n_steps_out, n_filters = cfg['n_steps_in'], cfg['n_steps_out'], cfg['n_filters']
    n_kernel = cfg['n_kernel']
    model = Sequential()
    model.add(Conv1D(n_filters, n_kernel, activation='relu', input_shape=(n_steps_in, n_features)))
    model.add(Conv1D(n_filters, n_kernel, activation='relu'))
    model.add(MaxPooling1D())
    model.add(Flatten())
    model.add(Dense(50, activation='relu'))
    model.add(Dense(n_steps_out))
    model.compile(loss='mse', optimizer='adam')
    return model


# forecast with a pre-fit model
def cnn_multi_step_mv_predict(model, history, cfg, steps=1):
    # unpack config
    n_steps = cfg['n_steps_in']
    n_features = history.shape[1]
    # prepare data
    x_input = array(history[-n_steps:]).reshape(1, n_steps, n_features)
    # forecast
    yhat = model.predict(x_input, verbose=0)
    return yhat[0]


def cnn_get_multi_step_mv_funcs():
    return [cnn_multi_step_mv_predict, cnn_multi_step_mv_fit, cnn_multi_step_mv_build]