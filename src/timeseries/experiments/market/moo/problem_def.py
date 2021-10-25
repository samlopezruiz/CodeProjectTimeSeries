import copy
import gc
import os
import time

import numpy as np
import tensorflow as tf
from pymoo.core.problem import Problem

from algorithms.tft2.harness.train_test import moo_q_loss_model
from algorithms.tft2.libs.hyperparam_opt import HyperparamOptManager
from algorithms.tft2.libs.tft_model import TemporalFusionTransformer
from timeseries.data.market.utils.names import get_inst_ohlc_names
from timeseries.experiments.market.expt_settings.configs import ExperimentConfig
from timeseries.experiments.market.moo.utils.model import get_last_layer_weights, params_conversion_weights, \
    run_moo_nn
from timeseries.utils.parallel import repeat_different_args


class TFT_Moo(Problem):

    def __init__(self,
                 model_folder,
                 data_formatter,
                 data_config,
                 loss_to_obj,
                 use_gpu=True,
                 parallelize_pop=True,
                 **kwargs):
        train, valid, test = data_formatter.split_data(data_config)

        # Sets up default params
        fixed_params = data_formatter.get_experiment_params()
        params = data_formatter.get_default_model_params()
        params["model_folder"] = model_folder

        # Sets up hyperparam manager
        print("*** Loading hyperparm manager ***")
        opt_manager = HyperparamOptManager({k: [params[k]] for k in params}, fixed_params, model_folder)
        model_params = opt_manager.get_next_parameters()

        with tf.device('/device:GPU:0' if use_gpu else "/cpu:0"):
            model = TemporalFusionTransformer(model_params)
            model.load(opt_manager.hyperparam_folder, use_keras_loadings=True)
            weights, last_layer = get_last_layer_weights(model)
            ind, w_params = params_conversion_weights(weights)

            print("*** Test Q-Loss with original weights ***")
            losses, unscaled_output_map = moo_q_loss_model(data_formatter, model, valid,
                                                           return_output_map=True)

            outputs, output_map, data = model.predict_all(valid, batch_size=128)
            transformer_output = outputs['transformer_output']

        self.transformer_output = transformer_output
        self.data_map = data
        self.data_formatter = data_formatter
        self.loss_to_obj = loss_to_obj
        self.valid = valid
        self.ini_weights = weights
        self.ini_ind = ind.reshape(1, -1)
        self.last_layer = last_layer
        self.weights_params = w_params
        self.original_losses = self.loss_to_obj(losses)
        self.parallelize_pop = parallelize_pop
        self.quantiles = copy.copy(model.quantiles)
        self.output_size = copy.copy(model.output_size)
        self.time_steps = copy.copy(model.time_steps)
        self.num_encoder_steps = copy.copy(model.num_encoder_steps)

        n_var = len(ind)
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
                 self.loss_to_obj]
                for x in X]

        F = repeat_different_args(run_moo_nn,
                                  args,
                                  parallel=self.parallelize_pop,
                                  n_jobs=None,
                                  use_tqdm=False)

        # gc.collect()
        # calculate the function values in a parallelized manner and wait until done
        out["F"] = np.array(F)


if __name__ == "__main__":
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

    # %%
    X = np.random.rand(100, problem.n_var)

    print('Evaluating')
    res = {}
    t0 = time.time()
    problem._evaluate(X, res)
    print('Eval time: {} s'.format(round(time.time() - t0, 4)))
    F = res['F']
