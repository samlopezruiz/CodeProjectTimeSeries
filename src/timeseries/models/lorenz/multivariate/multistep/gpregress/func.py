import time
from algorithms.gpregress.gp_func import train_gpregress
from algorithms.gpregress.gp_func import selection_roullete, selection_tournament
from timeseries.models.lorenz.functions.dataprep import multi_step_xy_from_mv
from numpy import array
import numpy as np


def gpregress_multi_step_mv_fit(train, cfg, plot_hist=False, verbose=0):
    # unpack architectures
    n_steps_in, n_steps_out, n_gen, elitism_size = cfg['n_steps_in'], cfg['n_steps_out'], cfg['n_gen'], cfg['elitism_size']
    depth, n_pop, mxpb, cxpb, selec = cfg['depth'], cfg['n_pop'], cfg['mxpb'], cfg['cxpb'], cfg['selection']
    primitives = cfg['primitives']
    if selec == 'roullete':
        selection_function = selection_roullete
    else:
        selection_function = selection_tournament
    tour_size = cfg.get('tour_size', 3)

    # prepare data
    X, y = multi_step_xy_from_mv(train, n_steps_in, n_steps_out)
    n_input = X.shape[1]
    # define and fit model
    bests = []
    start_time = time.time()
    for step in range(n_steps_out):
        if verbose > 0:
            print('step: {}'.format(step))
        y_train = y[:, step].ravel()
        best, pop, log, stat, size_log = train_gpregress(n_gen, n_input, primitives, depth, X, y_train,
                                                          n_pop, selec=selection_function, cxpb=cxpb,
                                                          mxpb=mxpb, elitism_size=elitism_size, verbose=verbose+1,
                                                          tour_size=tour_size)
        bests.append(best)
    train_time = round((time.time() - start_time), 2)
    mses = [best.mses[0] for best in bests]
    return bests, train_time, np.mean(mses)


# forecast with a pre-fit model
def gpregress_multi_step_mv_predict(model, history, cfg, steps=1):
    # unpack architectures
    n_steps_in = cfg['n_steps_in']
    n_steps_out = cfg['n_steps_out']
    n_input = n_steps_in * history.shape[1]
    # prepare data
    x_input = array(history[-n_steps_in:]).reshape(1, n_input)

    yhat = []
    for step in range(n_steps_out):
        yhat.append(model[step].predict(x_input))
    return np.array(yhat).ravel()


# forecast with a pre-fit model
def gpregress_multi_step_uv_predict_walk(model, history, cfg, steps=1):
    # unpack architectures
    n_steps_in = cfg['n_steps_in']
    n_steps_out = cfg['n_steps_out']
    # prepare data
    # forecast
    yhat = []
    history = list(history)
    for _ in range(n_steps_out):
        x_input = array(history[-n_steps_in:]).reshape(1, -1)
        y = model.predict(x_input)[0]
        yhat.append(y[0])
        history.append(y[0])
    return array(yhat).ravel()


def gpregress_get_multi_step_mv_funcs():
    return [gpregress_multi_step_mv_predict, gpregress_multi_step_mv_fit]