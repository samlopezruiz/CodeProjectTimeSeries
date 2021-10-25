import os

import numpy as np

from algorithms.moo.utils.plot import plot_hist_hv
from timeseries.data.market.utils.names import get_inst_ohlc_names
from timeseries.experiments.market.expt_settings.configs import ExperimentConfig
from timeseries.experiments.market.moo.harness.moo import run_moo, get_algorithm
from timeseries.experiments.market.moo.problem_def import TFT_Moo
from timeseries.experiments.utils.files import save_vars
from timeseries.utils.utils import get_type_str

if __name__ == '__main__':
    prob_cfg = {}
    algo_cfg = {'termination': ('n_gen', 80),
                'pop_size': 100,
                'hv_ref': [10]*2  # used only for SMS-EMOA
                }

    name = 'snp'
    experiment_name = 'fixed'
    config = ExperimentConfig(name,
                              market_file="split_ES_minute_60T_dwn_smpl_2018-01_to_2021-06_g12week_r25_4",
                              additional_file='subset_NQ_minute_60T_dwn_smpl_2012-01_to_2021-07',
                              regime_file="regime_ESc_r_ESc_macd_T10Y2Y_VIX",
                              macd_vars=['ESc'],
                              returns_vars=get_inst_ohlc_names('ES'),
                              add_prefix_col="NQ",
                              add_macd_vars=['NQc'],
                              add_returns_vars=get_inst_ohlc_names('NQ'),
                              true_target='ESc',
                              )

    formatter = config.make_data_formatter()

    # loss_to_obj = np.ndarray.flatten
    loss_to_obj = lambda x: np.mean(x, axis=0)

    problem = TFT_Moo(model_folder=os.path.join(config.model_folder, experiment_name),
                      data_formatter=formatter,
                      data_config=config.data_config,
                      loss_to_obj=loss_to_obj,
                      use_gpu=False,
                      parallelize_pop=True)

    name = 'NSGA2'

    algorithm = get_algorithm(name, algo_cfg, n_obj=problem.n_obj, sampling=problem.ini_ind)
    prob_cfg['n_var'], prob_cfg['n_obj'] = problem.n_var, problem.n_obj
    prob_cfg['hv_ref'] = [5] * problem.n_obj
    algo_cfg['name'] = get_type_str(algorithm)
    result = run_moo(problem, algorithm, algo_cfg, verbose=2)


    #%%
    import seaborn as sns
    import matplotlib.pyplot as plt

    sns.set_theme('poster')
    F = result['res'].pop.get('F')
    fig, ax = plt.subplots(figsize=(15, 9))
    ax.plot(F[:, 0], F[:, 1], '*')
    ax.plot(problem.original_losses[0], problem.original_losses[1], 'o')
    plt.show()

    plot_hist_hv(result['res'], save=False)

    res = result['res'].X, result['res'].F
    save_vars([res],
              os.path.join(config.results_folder, experiment_name, 'moo_weights'))

    # plot_results_moo(result['res'],
    #                  file_path=['img', 'opt_deap_res'],
    #                  title=create_title(prob_cfg, algo_cfg, algorithm),
    #                  save_plots=False)
    #
    # hv_pop = get_hypervolume(result['pop_hist'][-1], prob_cfg['hv_ref'])
    # hv_opt = get_hypervolume(result['res'].opt.get('F'), prob_cfg['hv_ref'])
    # print('Hypervolume from all population for reference point {}: {}'.format(str(prob_cfg['hv_ref']),
    #                                                                           round(hv_pop, 4)))
    # print('Hypervolume from opt population for reference point {}: {}'.format(str(prob_cfg['hv_ref']),
    #                                                                           round(hv_opt, 4)))
    # print('Optimum population {}/{}'.format(len(result['res'].opt),
    #                                         len(result['pop_hist'][-1])))
