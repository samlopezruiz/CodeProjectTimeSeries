import numpy as np
import pandas as pd


def get_moo_args(params):
    return (params['name'],
            params['algo_cfg'],
            params['prob_cfg'],
            params.get('problem', None),
            params.get('plot', False),
            params.get('file_path', ['img', 'res']),
            params.get('save_plots', False),
            params.get('verbose', 1),
            params.get('seed', None))


def get_hv_hist_vs_n_evals(algos_runs, algos_hv_hist_runs):
    algos_n_evals_runs = [[gen.evaluator.n_eval for gen in algo_runs[0]['result']['res'].history]
                          for algo_runs in algos_runs]
    mean_hv_hist = [np.mean(hv_hist_runs, axis=0) for hv_hist_runs in algos_hv_hist_runs]
    series = [pd.DataFrame(hv, index=n_eval) for hv, n_eval in zip(mean_hv_hist, algos_n_evals_runs)]
    df2 = pd.concat(series, ignore_index=False, join='outer', axis=1)
    return df2.values.T


import numpy as np


def get_deap_pops_obj(logbook):
    pops = logbook.select('pop')
    pops_obj = [np.array([ind.fitness.values for ind in pop]) for pop in pops]
    ref = np.max([np.max(wobjs, axis=0) for wobjs in pops_obj], axis=0) #+ 1
    return pops_obj, ref

def get_pymoo_pops_obj(res):
    pops = [pop.pop for pop in res.history]
    pops_obj = [np.array([ind.F for ind in pop]) for pop in pops]
    ref = np.max([np.max(wobjs, axis=0) for wobjs in pops_obj], axis=0) #+ 1
    return pops_obj, ref

def get_deap_pop_hist(logbook):
    pop_hist = []
    for gen in logbook:
        pop = gen['pop']
        pop_hist.append(np.array([ind.fitness.values for ind in gen['pop']]))
    return pop_hist