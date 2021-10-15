from algorithms.gpregress.math import covert_to_array, Xmat_from_inputX, protected_div, protected_sqrt
from algorithms.stroganoff.tests.multi import func_ts
from timeseries.experiments.lorenz.functions.dataprep import split_uv_seq_one_step
import numpy as np

if __name__ == '__main__':
    # %%
    cfg = {"n_steps_in": 5, "n_steps_out": 3, "n_gen": 10, "n_pop": 10,
           "cxpb": 0.6, "mxpb": 0.05, "depth": 5, 'elitism_size': 2}
    ts = func_ts([x / 5. for x in range(-100, 150)])
    ts_train, ts_test = ts[:200], ts[200:]
    X, y = split_uv_seq_one_step(ts_train, cfg['n_steps_in'])


    #%%
    x1 = X[:, 1]
    x2 = X[:, 2]

    x = Xmat_from_inputX(4, 5)

    #%%
    x1[2] = -1
    x2[2] = 0
    x0 = protected_div(x1, x2)
    x1 = protected_sqrt(x1)

