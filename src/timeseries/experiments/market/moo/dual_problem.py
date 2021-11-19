import copy
import gc
import os
import time

import joblib
import numpy as np
import tensorflow as tf
from pymoo.core.problem import Problem

from algorithms.tft2.harness.train_test import moo_q_loss_model, compute_moo_q_loss
from algorithms.tft2.libs.hyperparam_opt import HyperparamOptManager
from algorithms.tft2.libs.tft_model import TemporalFusionTransformer
from algorithms.tft2.utils.nn import dense_layer_output
from timeseries.data.market.utils.names import get_inst_ohlc_names
from timeseries.experiments.market.expt_settings.configs import ExperimentConfig
from timeseries.experiments.market.moo.utils.model import get_last_layer_weights, params_conversion_weights, \
    run_moo_nn, create_output_map, reconstruct_weights, run_single_w_nn, get_ix_ind_from_weights
from timeseries.experiments.market.moo.utils.utils import get_loss_to_obj_function
from timeseries.experiments.market.utils.filename import get_result_folder
from timeseries.experiments.market.utils.harness import get_model, get_model_data_config
from timeseries.utils.parallel import repeat_different_args


class DualQuantileWeights:

    def __init__(self,
                 architecture,
                 model_folder,
                 data_formatter,
                 data_config,
                 use_gpu=True,
                 parallelize_pop=True,
                 exclude_p50=True):
        Model = get_model(architecture)
        train, valid, test = data_formatter.split_data(data_config)

        # Sets up default params
        fixed_params = data_formatter.get_experiment_params()
        params = data_formatter.get_default_model_params()
        params["model_folder"] = model_folder

        # Sets up hyperparam manager
        print("\n*** Loading hyperparm manager ***")
        opt_manager = HyperparamOptManager({k: [params[k]] for k in params}, fixed_params, model_folder)
        model_params = opt_manager.get_next_parameters()

        with tf.device('/device:GPU:0' if use_gpu else "/cpu:0"):
            model = Model(model_params)
            model.load(opt_manager.hyperparam_folder, use_keras_loadings=True)
            weights, last_layer = get_last_layer_weights(model)

            print("\n*** Test Q-Loss with original weights ***")
            losses, unscaled_output_map = moo_q_loss_model(data_formatter, model, valid,
                                                           return_output_map=True)

            outputs, output_map, data = model.predict_all(valid, batch_size=128)
            transformer_output = outputs['transformer_output']

        self.transformer_output = transformer_output
        self.data_map = data
        self.data_formatter = data_formatter
        self.valid = valid
        self.original_weights = weights
        self.last_layer = last_layer
        self.exclude_p50 = exclude_p50
        self.original_losses = losses
        self.parallelize_pop = parallelize_pop
        self.quantiles = copy.copy(model.quantiles)
        self.output_size = copy.copy(model.output_size)
        self.time_steps = copy.copy(model.time_steps)
        self.num_encoder_steps = copy.copy(model.num_encoder_steps)

        ind_lq = get_ix_ind_from_weights(self.original_weights, 0)
        ind_uq = get_ix_ind_from_weights(self.original_weights, 2)

        self.lower_quantile_problem = SingleQuantileWeights(ind_lq,
                                                            self.quantiles,
                                                            self.output_size,
                                                            self.data_map,
                                                            self.time_steps,
                                                            self.num_encoder_steps,
                                                            self.transformer_output,
                                                            0,  # index of lower quantile
                                                            self.original_weights,
                                                            self.parallelize_pop,
                                                            self.original_losses)

        self.upper_quantile_problem = SingleQuantileWeights(ind_uq,
                                                            self.quantiles,
                                                            self.output_size,
                                                            self.data_map,
                                                            self.time_steps,
                                                            self.num_encoder_steps,
                                                            self.transformer_output,
                                                            2,  # index of lower quantile
                                                            self.original_weights,
                                                            self.parallelize_pop,
                                                            self.original_losses)

        # self.n_var = ind.shape[1]
        # self.n_obj = len(self.original_losses)
        # super().__init__(self.n_var, self.n_obj, n_constr=0, xl=-1.0, xu=1.0, **kwargs)

        gc.collect()

    def get_problems(self):
        return self.lower_quantile_problem, self.upper_quantile_problem


class SingleQuantileWeights(Problem):

    def __init__(self,
                 original_ind,
                 quantiles,
                 output_size,
                 data_map,
                 time_steps,
                 num_encoder_steps,
                 transformer_output,
                 ix_weight,
                 original_weights,
                 parallelize_pop,
                 original_losses,
                 **kwargs):
        self.ini_ind = original_ind
        self.quantiles = quantiles
        self.output_size = output_size
        self.data_map = data_map
        self.time_steps = time_steps
        self.num_encoder_steps = num_encoder_steps
        self.transformer_output = transformer_output
        self.ix_weight = ix_weight
        self.original_weights = original_weights
        self.parallelize_pop = parallelize_pop
        self.original_losses = original_losses[ix_weight, :]

        n_var = original_ind.shape[1]
        n_obj = 2
        super().__init__(n_var, n_obj, n_constr=0, xl=-1.0, xu=1.0, **kwargs)

    def _evaluate(self, X, out, *args, **kwargs):
        args = [[x,
                 self.quantiles,
                 self.output_size,
                 self.data_map,
                 self.time_steps,
                 self.num_encoder_steps,
                 self.transformer_output,
                 self.ix_weight,
                 self.original_weights
                 ]
                for x in X]

        F = repeat_different_args(run_single_w_nn,
                                  args,
                                  parallel=self.parallelize_pop,
                                  n_jobs=None,
                                  use_tqdm=False)

        # gc.collect()
        # calculate the function values in a parallelized manner and wait until done
        out["F"] = np.array(F)

    def compute_eq_F(self, X):
        args = [[x,
                 self.quantiles,
                 self.output_size,
                 self.data_map,
                 self.time_steps,
                 self.num_encoder_steps,
                 self.transformer_output,
                 self.ix_weight,
                 self.original_weights,
                 True]  # eq_f_loss
                for x in X]

        F_eq_F = repeat_different_args(run_single_w_nn,
                                       args,
                                       parallel=self.parallelize_pop,
                                       n_jobs=None,
                                       use_tqdm=False)

        F = [f[0] for f in F_eq_F]
        eq_F = [f[1] for f in F_eq_F]

        return np.array(F), np.array(eq_F)


if __name__ == "__main__":
    results_cfg = {'formatter': 'snp',
                   'experiment_name': '60t_ema_q159',
                   'results': 'TFTModel_ES_ema_r_q159_lr01_pred'
                   }

    model_results = joblib.load(os.path.join(get_result_folder(results_cfg), results_cfg['results'] + '.z'))

    config, formatter, model_folder = get_model_data_config(model_results['experiment_cfg'],
                                                            model_results['model_cfg'],
                                                            model_results['fixed_cfg'])

    experiment_cfg = model_results['experiment_cfg']

    type_func = 'mean_across_quantiles'  # 'ind_loss_woP50' #'mean_across_quantiles'

    dual_q_problem = DualQuantileWeights(architecture=experiment_cfg['architecture'],
                                         model_folder=model_folder,
                                         data_formatter=formatter,
                                         data_config=config.data_config,
                                         use_gpu=True,
                                         parallelize_pop=False)

    lower_q_problem, upper_q_problem = dual_q_problem.get_problems()

    # %%
    X = np.random.rand(100, lower_q_problem.n_var)

    print('Evaluating')
    res = {}
    t0 = time.time()
    lower_q_problem._evaluate(X, res)
    print('Eval time: {} s'.format(round(time.time() - t0, 4)))
    F_l = res['F']

    F_2, eq_F = lower_q_problem.compute_eq_F(X)

    print('Evaluating')
    res = {}
    t0 = time.time()
    upper_q_problem._evaluate(X, res)
    print('Eval time: {} s'.format(round(time.time() - t0, 4)))
    F_u = res['F']

    F_2, eq_F = upper_q_problem.compute_eq_F(X)
