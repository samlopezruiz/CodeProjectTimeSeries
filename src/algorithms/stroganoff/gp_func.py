from copy import deepcopy
import numpy as np
from numpy.random import randint
from algorithms.stroganoff.classes import Individual, Tree, generate
from algorithms.stroganoff.func import stats, to_df, print_crossover
from algorithms.stroganoff.tree import get_subtrees, substitute_subtrees, rand_node, find_subtree, find_subtree_by_id


def initialize_pop(pop_size, n_terminals, depth=10, p=0):
    id_count = 0
    pop = []
    for _ in range(pop_size):
        ind = Individual(n_terminals=n_terminals, depth=depth, p=p, id=id_count)
        id_count += len(ind.nodes)
        pop.append(ind)
    return pop, id_count


def selection_tournament(pop, scores, k=3):
    # first random selection
    selection_ix = randint(len(pop))
    for ix in randint(0, len(pop), k - 1):
        # check if better (e.g. perform a tournament)
        if scores[ix] < scores[selection_ix]:
            selection_ix = ix
    return pop[selection_ix]


def selection_roullete(pop, scores):
    pos_scores = max(scores) - scores + 1
    p = np.random.uniform(0, sum(pos_scores))
    for i, f in enumerate(pos_scores):
        if p <= 0:
            break
        p -= f
    return pop[i]


def evaluate_pop(pop, X, y):
    F = []
    for ind in pop:
        ind.gmdh(X, y)
        F.append(ind.get_mdl())

    return np.array(F)


def train_stroganoff(n_gen, n_terminals, depth, X, y, n_pop=100, selec=selection_roullete,
             cxpb=0.5, mxpb=0.05, elitism_size=0, verbose=1, tour_size=3):
    global sorted_score
    pop, id_count = initialize_pop(n_pop, n_terminals, depth=depth)
    scores = evaluate_pop(pop, X, y)
    best = pop[np.argmin(scores)]
    log = []
    log_size = []
    for gen in range(n_gen):
        # print(scores)
        # SELECTION
        if selec == selection_roullete:
            selected = [selec(pop, scores) for _ in range(n_pop)]
        else:
            selected = [selec(pop, scores, tour_size) for _ in range(n_pop)]

        # ELITISM
        if elitism_size > 0:
            sorted_score = sorted(scores)

        for i in range(elitism_size):
            ix = np.argmax(scores == sorted_score[i])
            selected[i] = deepcopy(pop[ix])

        # CROSSOVER
        for i in range(elitism_size, n_pop, 2):
            if np.random.rand() < cxpb:
                selected[i], selected[i + 1] = crossover(selected[i], selected[i + 1], max_depth=depth)

                # MUTATION
                if np.random.rand() < mxpb:
                    selected[i], id_count = mutate(selected[i], n_terminals, id_count, depth)
                if np.random.rand() < mxpb:
                    selected[i+1], id_count = mutate(selected[i+1], n_terminals, id_count, depth)


        # NEW POPULATION
        pop = selected
        scores = evaluate_pop(pop, X, y)
        depths = [p.depth for p in pop]

        best_s, worst_s, mean_s = stats(scores)
        min_d, max_d, mean_d = stats(depths)

        if verbose > 1:
            print('{}, best: {}, worst:{}, avg: {}, avg depth:{}, best mse:{}'.format(gen, best_s, worst_s, mean_s,
                                                                        round(np.mean(depths),1), round(best.mses[0], 4)))

        x = pop[np.argmin(scores)]
        if best.get_mdl() > x.get_mdl() and x.is_valid():
            best = deepcopy(x)

        log.append((gen, deepcopy(best), scores, best_s, worst_s, mean_s, best.mses[0]))
        log_size.append((gen, min_d, max_d, mean_d))

        if best_s == worst_s and worst_s == mean_s:
            print('early break, constant pop')
            break

    log_size_df, stat = to_df(log, log_size)

    return best, pop, log, stat[['best', 'worst', 'mean']], log_size_df


def crossover(parent1, parent2, max_depth, print_=False):
    if np.isfinite(parent1.mdls[1:]).any() == True and np.isfinite(parent2.mdls[1:]).any() == True:
        child1, child2 = deepcopy(parent1), deepcopy(parent2)
        # subtrees with the largest and smallest mdl value, cannot select root
        new_subt_1, new_subt_2, old_subt_1, old_subt_2 = get_subtrees(child1, child2)

        substitute_subtrees(old_subt_1, new_subt_2)

        if parent1 != parent2:
            substitute_subtrees(old_subt_2, new_subt_1)

        if print_:
            print_crossover(parent1, parent2, old_subt_1, new_subt_2, old_subt_2, new_subt_1, child1, child2)

        child1.update_lists()
        child2.update_lists()
        if child1.depth > max_depth:
            child1 = deepcopy(parent1)
        if child2.depth > max_depth or parent1 == parent2:
            # return parent 2 in order to conserve the same individual
            child2 = deepcopy(parent2)

        return child1, child2
    else:
        return parent1, parent2


def mutate(individual, n_terminals, id_count, depth):
    ind = deepcopy(individual)
    # node to replace
    node_id, ix = rand_node(ind)
    if node_id is not None:
        tree = ind.tree
        subtree_to_be_replaced, depth_node = find_subtree_by_id(tree, node_id)
        # allowed_depth = ind.depth - depth_node + 1
        allowed_depth = depth - ind.levels[ix]

        if allowed_depth < 1:
            print('ERROR: allowed_depth: ', allowed_depth)
            return ind, id_count

        new_ind = Individual(n_terminals, allowed_depth, id=id_count)
        new_subtree = new_ind.tree
        id_count += len(new_ind.nodes)
        substitute_subtrees(subtree_to_be_replaced, new_subtree)
        ind.update_lists()
        # ind.gmdh(X, y)
    return ind, id_count
