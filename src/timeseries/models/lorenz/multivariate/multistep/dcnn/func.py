import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from algorithms.wavenet.func import dcnn_build
from timeseries.plotly.plot import plot_history
from timeseries.models.lorenz.functions.dataprep import feature_multi_step_xy_from_mv
from numpy import array
import time


def dcnn_multi_step_mv_fit(train, cfg, plot_hist=False, verbose=0):
    # unpack config
    n_steps_in, n_steps_out = cfg['n_steps_in'], cfg['n_steps_out']
    n_epochs, n_batch = cfg['n_epochs'], cfg['n_batch']
    # prepare data
    X, y = feature_multi_step_xy_from_mv(train, n_steps_in, n_steps_out)
    n_features = X.shape[2]
    # define model
    model = dcnn_build(cfg, n_features)
    model.compile(loss='mse', optimizer='adam')
    if verbose > 0:
        print('No. of params: {}'.format(model.count_params()))
    # fit
    start_time = time.time()
    history = model.fit(X, y, epochs=n_epochs, batch_size=n_batch, verbose=verbose)
    train_time = round((time.time() - start_time), 2)

    # summarize history for accuracy
    if plot_hist:
        plot_history(history, title='D-CNN: '+str(cfg), plot_title=True)
    return model, train_time


# forecast with a pre-fit model
def dcnn_multi_step_mv_predict(model, history, cfg, steps=1):
    # unpack config
    n_steps = cfg['n_steps_in']
    n_features = history.shape[1]
    # prepare data
    x_input = array(history[-n_steps:]).reshape(1, n_steps, n_features)
    # forecast
    yhat = model.predict(x_input, verbose=0)
    return yhat[0]
