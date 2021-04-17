import numpy as np
import matplotlib.pyplot as plt

from algorithms.stroganoff.classes import Individual
from algorithms.stroganoff.gp_func import train_stroganoff, selection_tournament
from timeseries.models.lorenz.functions.dataprep import split_uv_seq_multi_step


def func_ts(x):
    return np.sin(x) + np.sin(np.array(x) / 2) + 10 + np.cos(np.array(x) / 3)


if __name__ == '__main__':
    #%%
    cfg = {"n_steps_in": 5, "n_steps_out": 3, "n_gen": 10, "n_pop": 10,
           "cxpb": 0.6, "mxpb": 0.05, "depth": 5, 'elitism_size': 2}
    ts = func_ts([x / 5. for x in range(-100, 150)])
    ts_train, ts_test = ts[:200], ts[200:]
    X, y = split_uv_seq_multi_step(ts_train, cfg['n_steps_in'], cfg['n_steps_out'])

    n_steps_in, n_steps_out, n_gen, elitism_size = cfg['n_steps_in'], cfg['n_steps_out'], cfg['n_gen'], cfg[
        'elitism_size']
    depth, n_pop, mxpb, cxpb = cfg['depth'], cfg['n_pop'], cfg['mxpb'], cfg['cxpb']

    # %%
    bests = []
    for step in range(n_steps_out):
        print('step: {}'.format(step))
        y_train = y[:, step].ravel()
        best, pop, log, stat, size_log = train_stroganoff(n_gen, n_steps_in, depth, X, y_train,
                                                          n_pop, selec=selection_tournament, cxpb=cxpb,
                                                          mxpb=mxpb, elitism_size=elitism_size, verbose=0,
                                                          tour_size=3)
        bests.append(best)
        
    #%%
    yhat = []
    x_input = np.array(ts_train[-n_steps_in:]).reshape(1, -1)
    for step in range(n_steps_out):
        yhat.append(bests[step].predict(x_input))
    yhat = np.array(yhat).ravel()
