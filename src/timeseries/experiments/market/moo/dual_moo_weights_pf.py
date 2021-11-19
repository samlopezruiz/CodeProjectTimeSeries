import gc
import os
import time

import joblib
import numpy as np
import telegram_send
from matplotlib import pyplot as plt

from algorithms.moo.utils.plot import plot_hist_hv
from timeseries.experiments.market.expt_settings.configs import ExperimentConfig
from timeseries.experiments.market.moo.dual_problem import DualQuantileWeights
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
                   'save_history': False,
                   'send_notifications': True}

    prob_cfg = {}
    results_cfg = {'formatter': 'snp',
                   'experiment_name': '60t_ema_q159',
                   'results': 'TFTModel_ES_ema_r_q159_lr01_pred'
                   }

    algo_cfg = {'termination': ('n_gen', 100),
                'pop_size': 100,
                'use_sampling': False,
                }

    moo_method = 'NSGA2'

    t0 = time.time()
    model_results = joblib.load(os.path.join(get_result_folder(results_cfg), results_cfg['results'] + '.z'))

    config, formatter, model_folder = get_model_data_config(model_results['experiment_cfg'],
                                                            model_results['model_cfg'],
                                                            model_results['fixed_cfg'])
    experiment_cfg = model_results['experiment_cfg']

    dual_q_problem = DualQuantileWeights(architecture=experiment_cfg['architecture'],
                                         model_folder=model_folder,
                                         data_formatter=formatter,
                                         data_config=config.data_config,
                                         use_gpu=False,
                                         parallelize_pop=False if moo_method == 'MOEAD' else True)

    lower_q_problem, upper_q_problem = dual_q_problem.get_problems()

    filename = '{}_{}_q{}_{}_{}_p{}_s{}_dual_wmoo'.format(experiment_cfg['architecture'],
                                                          experiment_cfg['vars_definition'],
                                                          quantiles_name(dual_q_problem.quantiles),
                                                          moo_method,
                                                          termination_name(algo_cfg['termination']),
                                                          algo_cfg['pop_size'],
                                                          int(algo_cfg['use_sampling']))

    results, times = {}, []
    for bound, problem in zip(['lq'], [lower_q_problem]):
    # for bound, problem in zip(['lq', 'uq'], [lower_q_problem, upper_q_problem]):

        sampling = np.tile(problem.ini_ind, (algo_cfg['pop_size'], 1)) if algo_cfg['use_sampling'] else None
        algorithm = get_algorithm(moo_method,
                                  algo_cfg,
                                  n_obj=problem.n_obj,
                                  sampling=sampling)

        prob_cfg['n_var'], prob_cfg['n_obj'] = problem.n_var, problem.n_obj
        prob_cfg['hv_ref'] = [5] * problem.n_obj
        algo_cfg['name'] = get_type_str(algorithm)
        moo_result = run_moo(problem,
                             algorithm,
                             algo_cfg,
                             verbose=2,
                             save_history=general_cfg['save_history'])

        X_sorted, F_sorted = sort_1st_col(moo_result['res'].pop.get('X'),
                                          moo_result['res'].pop.get('F'))

        times.append(round((time.time() - t0) / 60, 0))

        results[bound] = {'X': X_sorted,
                          'F': F_sorted,
                          'eq_F': None,
                          'original_losses': problem.original_losses,
                          'original_weights': dual_q_problem.original_weights,
                          'loss_to_obj_type': None,
                          'quantiles': problem.quantiles,
                          'experiment_cfg': model_results['experiment_cfg'],
                          'model_cfg': model_results['model_cfg'],
                          'fixed_cfg': model_results['fixed_cfg'],
                          'moo_method': moo_method,
                          'algo_cfg': algo_cfg,
                          'pop_hist': moo_result['pop_hist']}

        gc.collect()

    if general_cfg['send_notifications']:
        try:
            telegram_send.send(messages=["moo for {} completed in {} mins, tot: {}".format(filename,
                                                                                           times,
                                                                                           sum(times))])
        except Exception as e:
            print(e)

    if general_cfg['save_results']:
        save_vars(results,
                  os.path.join(config.results_folder,
                               experiment_cfg['experiment_name'],
                               'moo',
                               filename))
