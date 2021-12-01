from algorithms.gpregress.classes import Primitives, Individual
from algorithms.gpregress.math import protected_div, protected_sqrt
from algorithms.gpregress.standalone import print_tree
from algorithms.stroganoff.tests.multi import func_ts
from timeseries.experiments.lorenz.functions.dataprep import split_uv_seq_one_step
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
    ind = Individual(n_terminals=5, depth=5, primitives=primitives)
    ind.print_tree()
    ind.gmdh(X, y)
    y_pred = ind.eval(X)
    ind.update_n_nodes()