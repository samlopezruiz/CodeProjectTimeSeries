import os
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from algorithms.wavenet.func import dcnn_build, wavenet_build
from timeseries.plotly.plot import plot_history
from timeseries.models.lorenz.functions.dataprep import feature_multi_step_xy_from_uv
from numpy import array


def wavenet_multi_step_uv_fit(train, cfg, plot_hist=False, verbose=0):
    # unpack architectures
    n_steps_in, n_steps_out = cfg['n_steps_in'], cfg['n_steps_out']
    n_epochs, n_batch = cfg['n_epochs'], cfg['n_batch']
    # prepare data
    X, y = feature_multi_step_xy_from_uv(train, n_steps_in, n_steps_out)
    # define model
    model = wavenet_build(cfg, n_features=1)
    model.compile(loss='mse', optimizer='adam')
    if verbose > 0:
        print('No. of params: {}'.format(model.count_params()))
    # fit
    start_time = time.time()
    history = model.fit(X, y, epochs=n_epochs, batch_size=n_batch, verbose=verbose)
    train_time = round((time.time() - start_time), 2)
    # summarize history for accuracy
    if plot_hist:
        plot_history(history, title='D-CNN: ' + str(cfg), plot_title=True)
    return model, train_time, history.history['loss'][-1]


# forecast with a pre-fit model
def wavenet_multi_step_uv_predict(model, history, cfg, steps=1):
    # unpack architectures
    n_input = cfg['n_steps_in']
    # prepare data
    x_input = array(history[-n_input:]).reshape((1, n_input, 1))
    # forecast
    yhat = model.predict(x_input, verbose=0)
    return yhat[0]


# forecast with a pre-fit model
def wavenet_multi_step_uv_predict_walk(model, history, cfg, steps=1):
    # unpack architectures
    n_steps_in = cfg['n_steps_in']
    n_steps_out = cfg['n_steps_out']
    # prepare data
    # forecast
    yhat = []
    history = list(history)
    for _ in range(n_steps_out):
        x_input = array(history[-n_steps_in:]).reshape(1, n_steps_in, 1)
        y = model.predict(x_input)[0]
        yhat.append(y[0])
        history.append(y[0])
    return array(yhat).ravel()

def wavenet_get_multi_step_uv_funcs():
    return [wavenet_multi_step_uv_predict, wavenet_multi_step_uv_fit, wavenet_build]
