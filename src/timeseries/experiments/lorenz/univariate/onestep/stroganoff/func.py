from algorithms.stroganoff.gp_func import train_stroganoff, selection_roullete, selection_tournament
from algorithms.stroganoff.plot import plot_log
from timeseries.experiments.lorenz.functions.dataprep import split_uv_seq_one_step
from numpy import array


def stroganoff_one_step_uv_fit(train, cfg, plot_hist=False, verbose=0):
    # unpack architectures
    n_steps_in, n_gen, elitism_size = cfg['n_steps_in'], cfg['n_gen'], cfg['elitism_size']
    depth, n_pop, mxpb, cxpb, selec = cfg['depth'], cfg['n_pop'], cfg['mxpb'], cfg['cxpb'], cfg['selection']

    if selec == 'roullete':
        selection_function = selection_roullete
    else:
        selection_function = selection_tournament
    tour_size = cfg.get('tour_size', 3)

    # prepare data
    X, y = split_uv_seq_one_step(train, n_steps_in)
    # define and fit model
    best, pop, log, stat, size_log = train_stroganoff(n_gen, n_steps_in, depth, X, y,
                                              n_pop, selec=selection_function, cxpb=cxpb,
                                              mxpb=mxpb, elitism_size=elitism_size, verbose=verbose, tour_size=tour_size)
    # summarize history for accuracy
    if plot_hist:
        plot_log(stat, ylabel='MDL', title='MDL vs GENERATION')
        plot_log(size_log, ylabel='DEPTH', title='DEPTH vs GENERATION')
        best.print_tree()

    return best


# forecast with a pre-fit model
def stroganoff_one_step_uv_predict(model, history, cfg, steps=1):
    # unpack architectures
    n_steps_in = cfg['n_steps_in']
    # prepare data
    x_input = array(history[-n_steps_in:]).reshape(1, -1)
    # forecast
    yhat = model.predict(x_input)
    return yhat[0][0]
