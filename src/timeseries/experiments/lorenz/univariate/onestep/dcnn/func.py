import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from algorithms.wavenet.func import dcnn_build
from timeseries.plotly.plot import plot_history
from timeseries.experiments.lorenz.functions.dataprep import feature_multi_step_xy_from_uv, feature_one_step_xy_from_uv
from numpy import array
from copy import copy

def dcnn_one_step_uv_fit(train, cfg, plot_hist=False, verbose=0):
    # unpack architectures
    n_steps_in = cfg['n_steps_in']
    n_epochs, n_batch = cfg['n_epochs'], cfg['n_batch']
    # prepare data
    X, y = feature_one_step_xy_from_uv(train, n_steps_in)
    # define model
    cfg_ = copy(cfg)
    cfg_['n_steps_out'] = 1
    model = dcnn_build(cfg_, n_features=1)
    model.compile(loss='mse', optimizer='adam')
    print('No. of params: {}'.format(model.count_params()))
    # fit
    history = model.fit(X, y, epochs=n_epochs, batch_size=n_batch, verbose=verbose)
    # summarize history for accuracy
    if plot_hist:
        plot_history(history, title='D-CNN: ' + str(cfg), plot_title=True)
    return model


# forecast with a pre-fit model
def dcnn_one_step_uv_predict(model, history, cfg, steps=1):
    # unpack architectures
    n_input = cfg['n_steps_in']
    # prepare data
    x_input = array(history[-n_input:]).reshape((1, n_input, 1))
    # forecast
    yhat = model.predict(x_input, verbose=0)
    return yhat[0]

