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
    run_moo_nn, create_output_map, reconstruct_weights
from timeseries.experiments.market.moo.utils.utils import get_loss_to_obj_function
from timeseries.experiments.market.utils.filename import get_result_folder
from timeseries.experiments.market.utils.harness import get_model, get_model_data_config
from timeseries.utils.parallel import repeat_different_args


class WeightsNN_Moo(Problem):

    def __init__(self,
                 architecture,
                 model_folder,
                 data_formatter,
                 data_config,
                 loss_to_obj,
                 use_gpu=True,
                 parallelize_pop=True,
                 exclude_p50=True,

                 **kwargs):

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
            if exclude_p50:
                self.p50_w = weights[0][:, 1]
                self.p50_b = weights[1][1]
                mod_weights = [weights[0][:, [0, 2]], weights[1][[0, 2]]]
                ind, w_params = params_conversion_weights(mod_weights)
            else:
                self.p50_w = None
                self.p50_b = None
                ind, w_params = params_conversion_weights(weights)

            print("\n*** Test Q-Loss with original weights ***")
            losses, unscaled_output_map = moo_q_loss_model(data_formatter, model, valid,
                                                           return_output_map=True)

            outputs, output_map, data = model.predict_all(valid, batch_size=128)
            transformer_output = outputs['transformer_output']

            # if exclude_p50:
            #     orig_weights_p50 = ind[:, ind.shape[1] // 3: ind.shape[1] // 3 * 2]
            #     ind = np.concatenate([ind[:, ind.shape[1] // 3], ind[:, ind.shape[1] // 3 * 2:]])
            # else:
            #     orig_weights_p50 = None

        self.transformer_output = transformer_output
        self.data_map = data
        self.data_formatter = data_formatter
        self.loss_to_obj = loss_to_obj
        self.valid = valid
        self.ini_weights = weights
        self.ini_ind = ind
        self.last_layer = last_layer
        self.weights_params = w_params
        self.exclude_p50 = exclude_p50
        # self.orig_weights_p50 = orig_weights_p50
        self.original_losses = self.loss_to_obj(losses)
        self.parallelize_pop = parallelize_pop
        self.quantiles = copy.copy(model.quantiles)
        self.output_size = copy.copy(model.output_size)
        self.time_steps = copy.copy(model.time_steps)
        self.num_encoder_steps = copy.copy(model.num_encoder_steps)

        n_var = ind.shape[1]
        n_obj = len(self.original_losses)
        super().__init__(n_var, n_obj, n_constr=0, xl=-1.0, xu=1.0, **kwargs)

        gc.collect()

    def _evaluate(self, X, out, *args, **kwargs):
        args = [[x,
                 self.quantiles,
                 self.output_size,
                 self.data_map,
                 self.time_steps,
                 self.num_encoder_steps,
                 self.transformer_output,
                 self.weights_params,
                 self.loss_to_obj,
                 self.p50_w,
                 self.p50_b]
                for x in X]

        F = repeat_different_args(run_moo_nn,
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
                 self.weights_params,
                 self.loss_to_obj,
                 self.p50_w,
                 self.p50_b,
                 True] # output_eq_loss
                for x in X]

        F_eq_F = repeat_different_args(run_moo_nn,
                                       args,
                                       parallel=self.parallelize_pop,
                                       n_jobs=None,
                                       use_tqdm=False)

        F = [f[0] for f in F_eq_F]
        eq_F = [f[1] for f in F_eq_F]

        return np.array(F), np.array(eq_F)


if __name__ == "__main__":
    results_cfg = {'formatter': 'snp',
                   'experiment_name': '60t_ema',
                   'results': 'TFTModel_ES_ema_r_q159_pred_1'
                   }

    model_results = joblib.load(os.path.join(get_result_folder(results_cfg), results_cfg['results'] + '.z'))

    config, formatter, model_folder = get_model_data_config(model_results['experiment_cfg'],
                                                            model_results['model_cfg'],
                                                            model_results['fixed_cfg'])

    experiment_cfg = model_results['experiment_cfg']

    type_func = 'mean_across_quantiles'  # 'ind_loss_woP50' #'mean_across_quantiles'

    problem = WeightsNN_Moo(architecture=experiment_cfg['architecture'],
                            model_folder=model_folder,
                            data_formatter=formatter,
                            data_config=config.data_config,
                            loss_to_obj=get_loss_to_obj_function(type_func),
                            use_gpu=False,
                            parallelize_pop=True)

    # %%
    X = np.random.rand(100, problem.n_var)

    print('Evaluating')
    res = {}
    t0 = time.time()
    problem._evaluate(X, res)
    print('Eval time: {} s'.format(round(time.time() - t0, 4)))
    F = res['F']

    F_2, eq_F = problem.compute_eq_F(X)
