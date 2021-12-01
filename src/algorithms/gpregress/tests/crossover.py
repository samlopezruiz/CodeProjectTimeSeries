from algorithms.gpregress.classes import Primitives, Individual, SumNode, Node
from algorithms.gpregress.gp_func import crossover, mutate
from algorithms.gpregress.math import protected_div, protected_sqrt
from algorithms.gpregress.standalone import print_tree
from algorithms.gpregress.tree import find_subtree_by_id, get_subtrees
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
    parent1 = Individual(n_terminals=5, depth=5, primitives=primitives)
    parent2 = Individual(n_terminals=5, depth=5, primitives=primitives)
    parent1.gmdh(X, y)
    parent2.gmdh(X, y)
    ch1, ch2 = crossover(parent1, parent2, 4, print_=False)

    #%%
    mut, id_count = mutate(parent1, primitives, 5, 20, 5)
    print('---  P 1 ---')
    parent1.print_tree()
    print('---  MUT ---')
    mut.print_tree()