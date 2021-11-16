import os

import joblib
import numpy as np
from matplotlib import pyplot as plt

from algorithms.moo.utils.plot import plot_hist_hv
from timeseries.experiments.market.expt_settings.configs import ExperimentConfig
from timeseries.experiments.market.moo.harness.moo import run_moo, get_algorithm
from timeseries.experiments.market.moo.problem_def import WeightsNN_Moo
from timeseries.experiments.market.moo.utils.utils import get_loss_to_obj_function
from timeseries.experiments.market.plot.plot import plot_2D_pareto_front, plot_2D_moo_results
from timeseries.experiments.market.utils.filename import get_output_folder, quantiles_name, get_result_folder
from timeseries.experiments.market.utils.harness import get_model_data_config
from timeseries.experiments.utils.files import save_vars
from timeseries.utils.utils import get_type_str
import seaborn as sns

sns.set_theme('poster')

if __name__ == '__main__':
    #%%
    general_cfg = {'save_results': True,
                   'save_plot': True,
                   'save_history': False}

    prob_cfg = {}
    results_cfg = {'formatter': 'snp',
                   'experiment_name': '60t_ema_q159',
                   'results': 'TFTModel_ES_ema_r_q159_lr01_pred'
                   }

    algo_cfg = {'termination': ('n_gen', 150),
                'pop_size': 150,
                }

    model_results = joblib.load(os.path.join(get_result_folder(results_cfg), results_cfg['results']+'.z'))

    config, formatter, model_folder = get_model_data_config(model_results['experiment_cfg'],
                                                            model_results['model_cfg'],
                                                            model_results['fixed_cfg'])

    experiment_cfg = model_results['experiment_cfg']

    type_func = 'mean_across_quantiles' #'ind_loss_woP50' #'mean_across_quantiles'

    problem = WeightsNN_Moo(architecture=experiment_cfg['architecture'],
                            model_folder=model_folder,
                            data_formatter=formatter,
                            data_config=config.data_config,
                            loss_to_obj=get_loss_to_obj_function(type_func),
                            use_gpu=False,
                            parallelize_pop=True)

    name = 'NSGA2'

    algorithm = get_algorithm(name, algo_cfg, n_obj=problem.n_obj) #, sampling=problem.ini_ind)
    prob_cfg['n_var'], prob_cfg['n_obj'] = problem.n_var, problem.n_obj
    prob_cfg['hv_ref'] = [5] * problem.n_obj
    algo_cfg['name'] = get_type_str(algorithm)
    moo_result = run_moo(problem,
                         algorithm,
                         algo_cfg,
                         verbose=2,
                         save_history=general_cfg['save_history'])

    _, eq_F = problem.compute_eq_F(moo_result['res'].pop.get('X'))

    # %%
    X, F = moo_result['res'].X, moo_result['res'].F
    ix = np.argsort(F, axis=0)
    X_sorted = X[ix[:, 0], :]
    F_sorted = F[ix[:, 0], :]
    eq_F_sorted = eq_F[ix[:, 0], :]


   #%%
    original_ix = np.argmin(np.sum(np.abs(F_sorted - problem.original_losses), axis=1))

    plot_2D_moo_results(F_sorted, eq_F_sorted,
                        original_ixs=original_ix,
                        figsize=(20, 15),
                        title='Multi objective optimization for quantiles: {}'.format(problem.quantiles))

    #%%
    if general_cfg['save_history']:
        plot_hist_hv(moo_result['res'], save=False)

    result = {'X': X_sorted,
              'F': F_sorted,
              'eq_F': eq_F_sorted,
              'original_losses': problem.original_losses,
              'p50_w': problem.p50_w,
              'p50_b': problem.p50_b,
              'loss_to_obj_type': type_func,
              'quantiles': problem.quantiles,
              'experiment_cfg': model_results['experiment_cfg'],
              'model_cfg': model_results['model_cfg'],
              'fixed_cfg': model_results['fixed_cfg'],
              }

    if general_cfg['save_results']:
        save_vars(result,
                  os.path.join(config.results_folder,
                               experiment_cfg['experiment_name'],
                               'moo',
                               '{}_q{}_moo_weights'.format(experiment_cfg['vars_definition'],
                                                           quantiles_name(result['quantiles']))))