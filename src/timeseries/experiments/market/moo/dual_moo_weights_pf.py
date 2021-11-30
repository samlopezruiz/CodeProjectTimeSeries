import gc
import os
import time

import joblib
import numpy as np
import seaborn as sns
import telegram_send

from timeseries.experiments.market.moo.dual_problem import DualQuantileWeights
from timeseries.experiments.market.moo.harness.moo import run_moo, get_algorithm
from timeseries.experiments.market.moo.utils.harness import run_dual_moo_weights
from timeseries.experiments.market.moo.utils.utils import sort_1st_col
from timeseries.experiments.market.utils.filename import quantiles_name, get_result_folder, \
    termination_name
from timeseries.experiments.market.utils.harness import get_model_data_config
from timeseries.experiments.utils.files import save_vars
from timeseries.utils.utils import get_type_str

sns.set_theme('poster')

if __name__ == '__main__':
    # %%
    general_cfg = {'save_results': True,
                   'save_history': False,
                   'send_notifications': True}

    prob_cfg = {}
    results_cfg = {'formatter': 'snp',
                   'experiment_name': '60t_ema_q357',
                   'results': 'TFTModel_ES_ema_r_q357_lr01_pred'
                   }

    algo_cfg = {'termination': ('n_gen', 100),
                'pop_size': 100,
                'use_sampling': False,
                'optimize_eq_weights': False,
                'use_constraints': True,
                'constraints': [1., 1.],
                }


    moo_method = 'NSGA2'

    model_results = joblib.load(os.path.join(get_result_folder(results_cfg), results_cfg['results'] + '.z'))

    config, formatter, model_folder = get_model_data_config(model_results['experiment_cfg'],
                                                            model_results['model_cfg'],
                                                            model_results['fixed_cfg'])
    experiment_cfg = model_results['experiment_cfg']

    dual_q_problem = DualQuantileWeights(architecture=experiment_cfg['architecture'],
                                         model_folder=model_folder,
                                         data_formatter=formatter,
                                         data_config=config.data_config,
                                         use_gpu=True,
                                         parallelize_pop=False if moo_method == 'MOEAD' else True,
                                         constraints_limits=algo_cfg['constraints'] if algo_cfg[
                                             'use_constraints'] else None,
                                         optimize_eq_weights=algo_cfg['optimize_eq_weights'])

    lower_q_problem, upper_q_problem = dual_q_problem.get_problems()

    filename = '{}_{}_q{}_{}_{}_p{}_s{}_c{}_eq{}_dual_wmoo'.format(experiment_cfg['architecture'],
                                                              experiment_cfg['vars_definition'],
                                                              quantiles_name(dual_q_problem.quantiles),
                                                              moo_method,
                                                              termination_name(algo_cfg['termination']),
                                                              algo_cfg['pop_size'],
                                                              int(algo_cfg['use_sampling']),
                                                              int(algo_cfg['use_constraints']),
                                                              int(algo_cfg['optimize_eq_weights']),
                                                              )

    res = run_dual_moo_weights(moo_method,
                               algo_cfg,
                               general_cfg,
                               prob_cfg,
                               lower_q_problem,
                               upper_q_problem,
                               dual_q_problem,
                               model_results,
                               verbose=2)

    if general_cfg['send_notifications']:
        try:
            telegram_send.send(messages=["moo for {} completed in {} mins, tot: {}".format(filename,
                                                                                           res['times'],
                                                                                           sum(res['times']))])
        except Exception as e:
            print(e)

    if general_cfg['save_results']:
        save_vars(res['results'],
                  os.path.join(config.results_folder,
                               experiment_cfg['experiment_name'],
                               'moo',
                               filename))
