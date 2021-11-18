import os
import time

import joblib
import numpy as np
import telegram_send
from matplotlib import pyplot as plt

from algorithms.moo.utils.plot import plot_hist_hv
from timeseries.experiments.market.expt_settings.configs import ExperimentConfig
from timeseries.experiments.market.moo.harness.moo import run_moo, get_algorithm
from timeseries.experiments.market.moo.problem_def import WeightsNN_Moo
from timeseries.experiments.market.moo.utils.utils import get_loss_to_obj_function, sort_1st_col
from timeseries.experiments.market.plot.plot import plot_2D_pareto_front, plot_2D_moo_results
from timeseries.experiments.market.utils.filename import get_output_folder, quantiles_name, get_result_folder, \
    termination_name
from timeseries.experiments.market.utils.harness import get_model_data_config
from timeseries.experiments.utils.files import save_vars
from timeseries.utils.utils import get_type_str
import seaborn as sns

sns.set_theme('poster')

if __name__ == '__main__':
    # %%
    general_cfg = {'save_results': True,
                   'save_history': True,
                   'send_notifications': True}

    prob_cfg = {}
    results_cfg = {'formatter': 'snp',
                   'experiment_name': '60t_ema_q159',
                   'results': 'TFTModel_ES_ema_r_q159_lr01_pred'
                   }

    algo_cfg = {'termination': ('n_gen', 250),
                'pop_size': 250,
                'use_sampling': True,
                }

    agg_obj_type_func = 'ind_loss_woP50'  # 'ind_loss_woP50' #'mean_across_quantiles'

    t0 = time.time()
    model_results = joblib.load(os.path.join(get_result_folder(results_cfg), results_cfg['results'] + '.z'))

    config, formatter, model_folder = get_model_data_config(model_results['experiment_cfg'],
                                                            model_results['model_cfg'],
                                                            model_results['fixed_cfg'])
    experiment_cfg = model_results['experiment_cfg']

    problem = WeightsNN_Moo(architecture=experiment_cfg['architecture'],
                            model_folder=model_folder,
                            data_formatter=formatter,
                            data_config=config.data_config,
                            loss_to_obj=get_loss_to_obj_function(agg_obj_type_func),
                            use_gpu=False,
                            parallelize_pop=True)

    moo_method = 'NSGA2'

    algorithm = get_algorithm(moo_method,
                              algo_cfg,
                              n_obj=problem.n_obj,
                              sampling=np.tile(problem.ini_ind, (algo_cfg['pop_size'], 1)) if algo_cfg['use_sampling'] else None)
                              # sampling=problem.ini_ind if algo_cfg['use_sampling'] else None)

    prob_cfg['n_var'], prob_cfg['n_obj'] = problem.n_var, problem.n_obj
    prob_cfg['hv_ref'] = [5] * problem.n_obj
    algo_cfg['name'] = get_type_str(algorithm)
    moo_result = run_moo(problem,
                         algorithm,
                         algo_cfg,
                         verbose=2,
                         save_history=general_cfg['save_history'])

    _, eq_F = problem.compute_eq_F(moo_result['res'].pop.get('X'))
    X_sorted, F_sorted, eq_F_sorted = sort_1st_col(moo_result['res'].pop.get('X'),
                                                   moo_result['res'].pop.get('F'),
                                                   eq_F)

    # %%
    filename = '{}_{}_q{}_{}_{}_p{}_s{}_k{}_wmoo'.format(experiment_cfg['architecture'],
                                                         experiment_cfg['vars_definition'],
                                                         quantiles_name(problem.quantiles),
                                                         moo_method,
                                                         termination_name(algo_cfg['termination']),
                                                         algo_cfg['pop_size'],
                                                         int(algo_cfg['use_sampling']),
                                                         F_sorted.shape[1])

    if general_cfg['send_notifications']:
        telegram_send.send(messages=["moo for {} completed in {} min".format(filename,
                                                                             round((time.time() - t0) / 60, 0))])

    if general_cfg['save_history']:
        plot_hist_hv(moo_result['res'], save=False)

    result = {'X': X_sorted,
              'F': F_sorted,
              'eq_F': eq_F_sorted,
              'original_losses': problem.original_losses,
              'p50_w': problem.p50_w,
              'p50_b': problem.p50_b,
              'loss_to_obj_type': agg_obj_type_func,
              'quantiles': problem.quantiles,
              'experiment_cfg': model_results['experiment_cfg'],
              'model_cfg': model_results['model_cfg'],
              'fixed_cfg': model_results['fixed_cfg'],
              'moo_method': moo_method,
              'algo_cfg': algo_cfg,
              'pop_hist': moo_result['pop_hist']
              }

    if general_cfg['save_results']:
        save_vars(result,
                  os.path.join(config.results_folder,
                               experiment_cfg['experiment_name'],
                               'moo',
                               filename))
