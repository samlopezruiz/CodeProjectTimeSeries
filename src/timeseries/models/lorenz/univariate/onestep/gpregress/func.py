import time

from algorithms.gpregress.gp_func import train_gpregress
from algorithms.stroganoff.gp_func import train_stroganoff, selection_roullete, selection_tournament
from algorithms.stroganoff.plot import plot_log
from timeseries.models.lorenz.functions.dataprep import split_uv_seq_one_step
from numpy import array


def gpregress_one_step_uv_fit(train, cfg, plot_hist=False, verbose=0):
    # unpack config
    n_steps_in, n_gen, elitism_size = cfg['n_steps_in'], cfg['n_gen'], cfg['elitism_size']
    depth, n_pop, mxpb, cxpb, selec = cfg['depth'], cfg['n_pop'], cfg['mxpb'], cfg['cxpb'], cfg['selection']
    primitives = cfg['primitives']
    if selec == 'roullete':
        selection_function = selection_roullete
    else:
        selection_function = selection_tournament
    tour_size = cfg.get('tour_size', 3)

    # prepare data
    X, y = split_uv_seq_one_step(train, n_steps_in)
    # define and fit model
    start_time = time.time()
    best, pop, log, stat, size_log = train_gpregress(n_gen, n_steps_in, primitives, depth, X, y,
                                              n_pop, selec=selection_function, cxpb=cxpb,
                                              mxpb=mxpb, elitism_size=elitism_size, verbose=verbose, tour_size=tour_size)
    train_time = round((time.time() - start_time), 2)
    # summarize history for accuracy
    if plot_hist:
        plot_log(stat, ylabel='MDL', title='MDL vs GENERATION')
        plot_log(size_log, ylabel='DEPTH', title='DEPTH vs GENERATION')
        best.print_tree()

    return best, train_time, best.mses[0]


# forecast with a pre-fit model
def gpregress_one_step_uv_predict(model, history, cfg, steps=1):
    # unpack config
    n_steps_in = cfg['n_steps_in']
    # prepare data
    x_input = array(history[-n_steps_in:]).reshape(1, -1)
    # forecast
    yhat = model.predict(x_input)
    return yhat[0]


def gpregress_get_one_step_uv_funcs():
    return [gpregress_one_step_uv_predict, gpregress_one_step_uv_fit]