import numpy as np
from algorithms.stroganoff.standalone import print_tree
import pandas as pd


def print_crossover(parent1, parent2, old_subt_1, new_subt_2, old_subt_2, new_subt_1, child1, child2):
    print('-- CROSSOVER --')
    print('---PARENT 1---')
    parent1.print_tree()
    print('---PARENT 2---')
    parent2.print_tree()
    print('---w1---')
    print_tree(old_subt_1)
    print('---b2---')
    print_tree(new_subt_2)
    print('---w2---')
    print_tree(old_subt_2)
    print('---b1---')
    print_tree(new_subt_1)
    print('---CHILD 1---')
    child1.print_tree()
    print('---CHILD 2---')
    child2.print_tree()


def to_df(log, log_size):
    stat = pd.DataFrame(log, columns=['gen', 'best_ind', 'scores', 'best', 'worst', 'mean', 'mse'])
    stat.set_index('gen', inplace=True)
    log_size_df = pd.DataFrame(log_size, columns=['gen', 'min_depth', 'max_depth', 'mean_depth'])
    log_size_df.set_index('gen', inplace=True)
    return log_size_df, stat

def stats(scores):
    return round(min(scores), 2), round(max(scores), 2), round(np.mean(scores), 2)




