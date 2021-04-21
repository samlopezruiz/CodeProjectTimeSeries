from algorithms.gpregress.classes import Primitives, Individual
from algorithms.gpregress.gp_func import initialize_pop, evaluate_pop, train_gpregress, selection_roullete
from algorithms.gpregress.math import protected_div, protected_sqrt, score_error
from algorithms.gpregress.plot import plot_log, plot_pred
from algorithms.gpregress.standalone import print_tree
from algorithms.stroganoff.tests.multi import func_ts
from timeseries.models.lorenz.functions.dataprep import split_uv_seq_one_step
import operator
import numpy as np


if __name__ == '__main__':
    # %%
    cfg = {"n_steps_in": 5, "n_steps_out": 3, "n_gen": 10, "n_pop": 10,
           "cxpb": 0.6, "mxpb": 0.05, "depth": 5, 'elitism_size': 2}
    ts = func_ts([x / 5. for x in range(-100, 150)])
    ts_train, ts_test = ts[:200], ts[200:]
    X, y = split_uv_seq_one_step(ts_train, cfg['n_steps_in'])

    # %%
    primitives = Primitives()
    primitives.add(np.multiply, 2)
    primitives.add(protected_div, 2, 'div')
    primitives.add(protected_sqrt, 1, 'sqrt')
    primitives.add(np.cos, 1)

    #%%
    n_gen = 10
    n_pop = 10
    n_terminals = 5
    depth = 5

    best, pop, log, stat, size_log = train_gpregress(n_gen, n_terminals, primitives, depth, X, y, n_pop=n_pop,
                                                     selec=selection_roullete, cxpb=0.5, mxpb=0.1, elitism_size=2,
                                                     verbose=2, tour_size=3)

    plot_log(stat, ylabel='MDL', title='MDL vs GENERATION')
    plot_log(size_log, ylabel='DEPTH', title='DEPTH vs GENERATION')

    # %%
    n_steps_in = n_terminals
    y_pred = list(ts[:n_steps_in])
    history = list(ts[:n_steps_in])
    for i in range(n_steps_in, len(ts)):
        x = np.array(history[-n_steps_in:])
        y_pred.append(best.predict(x)[0])
        history.append(ts[i])

    y_pred = y_pred[:len(ts)]
    plot_pred(ts, y_pred)

    # %%
    best.print_tree()
    print('mse: {}'.format(score_error(ts, y_pred)))