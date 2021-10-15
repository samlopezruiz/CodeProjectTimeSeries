from algorithms.stroganoff.classes import Individual
from algorithms.stroganoff.tests.multi import func_ts
from timeseries.experiments.lorenz.functions.dataprep import split_uv_seq_one_step

if __name__ == '__main__':
    #%%
    cfg = {"n_steps_in": 5, "n_steps_out": 3, "n_gen": 10, "n_pop": 10,
           "cxpb": 0.6, "mxpb": 0.05, "depth": 5, 'elitism_size': 2}
    ts = func_ts([x / 5. for x in range(-100, 150)])
    ts_train, ts_test = ts[:200], ts[200:]
    X, y = split_uv_seq_one_step(ts_train, cfg['n_steps_in'])

    #%%
    ind = Individual(5, 3)
    print(ind.is_valid())
    ind.gmdh(X, y)
    print(ind.is_valid())
