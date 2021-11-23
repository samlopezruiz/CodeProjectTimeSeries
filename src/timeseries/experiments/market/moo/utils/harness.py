import gc
import time

import numpy as np

from timeseries.experiments.market.moo.harness.moo import get_algorithm, run_moo
from timeseries.experiments.market.moo.utils.utils import sort_1st_col
from timeseries.utils.utils import get_type_str


def run_dual_moo_weights(moo_method,
                         algo_cfg,
                         general_cfg,
                         prob_cfg,
                         lower_q_problem,
                         upper_q_problem,
                         dual_q_problem,
                         model_results):

    results, times = {}, []
    for bound, problem in zip(['lq', 'uq'], [lower_q_problem, upper_q_problem]):
        t0 = time.time()
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

        _, eq_F = problem.compute_eq_F(moo_result['res'].pop.get('X'))
        X_sorted, F_sorted, eq_F_sorted = sort_1st_col(moo_result['res'].pop.get('X'),
                                                       moo_result['res'].pop.get('F'),
                                                       eq_F)

        times.append(round((time.time() - t0) / 60, 0))

        results[bound] = {'X': X_sorted,
                          'F': F_sorted,
                          'eq_F': eq_F_sorted,
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

    return {'results': results,
            'times': times}
