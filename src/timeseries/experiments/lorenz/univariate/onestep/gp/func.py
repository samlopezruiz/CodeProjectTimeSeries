import operator
import random
import numpy as np
from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp
from math import sqrt
from collections import deque
from timeseries.experiments.lorenz.functions.dataprep import split_uv_seq_multi_step
from timeseries.experiments.lorenz.univariate.onestep.gp.plotgp import plot_gp_log, plot_ind


def original(x):
    return np.sin(x) + np.sin(np.array(x) / 2) + 10 + np.cos(np.array(x) / 3)


def protectedDiv(left, right):
    try:
        return left / right
    except:
        return 1


def protectedSqrt(x):
    try:
        return sqrt(x)
    except:
        return 1


def create_toolbox(ts_train, in_steps=10, out_steps=1):
    pset = gp.PrimitiveSet("MAIN", in_steps * 2)
    pset.addPrimitive(operator.add, 2)
    pset.addPrimitive(operator.sub, 2)
    # pset.addPrimitive(operator.mul, 2)
    # pset.addPrimitive(protectedDiv, 2, 'div')
    # pset.addPrimitive(operator.neg, 1)
    # pset.addPrimitive(protectedSqrt, 1, 'sqrt')
    # pset.addEphemeralConstant("rand101", lambda: random.randint(0,10))

    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=2)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=pset)

    def evalSymbReg(individual, Xy):
        # Transform the tree expression in a callable function
        func = toolbox.compile(expr=individual)
        # Evaluate the mean squared error between the expression
        X, Y = Xy
        diffs = []
        n = X.shape[1]
        errs = deque(maxlen=n)
        for _ in range(n):
            errs.append(0)
        # print(individual)

        for x, Y_obs in zip(X, Y):
            history = list(x)
            for i, y in enumerate(Y_obs):
                inputs = list(errs) + history[-n:]
                y_pred = func(*inputs)
                # print(inputs, y_pred)
                if i == 0:
                    errs.append(y - y_pred)
                diffs.append((y_pred - y) ** 2)
                history.append(y_pred)

        # print(np.sqrt(np.mean(diffs)))
        return np.sqrt(np.mean(diffs)),

    toolbox.register("evaluate", evalSymbReg, Xy=split_uv_seq_multi_step(ts_train, in_steps, out_steps))
    toolbox.register("select", tools.selTournament, tournsize=4)
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

    toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=10))
    toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=10))

    return toolbox, pset


def train_gp(ts_train, in_steps=10, out_steps=1, ngen=40, cxpb=0.8, mutpb=0.2):
    # random.seed(318)

    toolbox, pset = create_toolbox(ts_train, in_steps=in_steps, out_steps=out_steps)

    pop = toolbox.population(n=300)
    hof = tools.HallOfFame(2)

    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    mstats.register("avg", np.mean)
    mstats.register("std", np.std)
    mstats.register("min", np.min)
    mstats.register("max", np.max)

    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=cxpb, mutpb=mutpb, ngen=ngen, stats=mstats,
                                   halloffame=hof, verbose=True)
    # print log
    return toolbox, pset, pop, log, hof


def gp_one_step_uv_fit(train, cfg, plot_hist=False, verbose=0):
    toolbox, pset, pop, log, hof = train_gp(train, in_steps=cfg['n_steps_in'],
                                            out_steps=cfg.get('n_steps_out',1), ngen=cfg['ngen'], cxpb=cfg['cxpb'], mutpb=cfg['mutpb'])
    if plot_hist:
        plot_gp_log(log)

    pop_wo_nans = [p for p in pop if p.fitness.values[0] > 0]
    pop_wo_nans.sort(key=lambda x: x.fitness, reverse=True)
    for i in range(3):
        print("{}: {} = {}".format(i, str(pop_wo_nans[i]), round(toolbox.evaluate(pop_wo_nans[i])[0], 6)))

    if plot_hist:
        best = pop_wo_nans[0]
        plot_ind(best, root=0)

    return gp.compile(pop[0], pset)


# forecast with a pre-fit model
def gp_one_step_uv_predict(model, history, errors, cfg, steps=1):
    # unpack architectures
    n_input = cfg['n_steps_in']
    # prepare data
    x_input = list(errors) + list(history[-n_input:])
    # forecast
    yhat = model(*x_input)
    return yhat
