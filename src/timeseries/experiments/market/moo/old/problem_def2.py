import copy
import os
import time

import numpy as np
import tensorflow as tf
from joblib import wrap_non_picklable_objects, delayed, Parallel
from pymoo.core.problem import Problem
from tqdm import tqdm

from algorithms.tft2.harness.train_test import moo_q_loss_model
from algorithms.tft2.libs.attn_model import AttnModel
from algorithms.tft2.libs.hyperparam_opt import HyperparamOptManager
from algorithms.tft2.libs.tft_model import TemporalFusionTransformer
from timeseries.data.market.utils.names import get_inst_ohlc_names
from timeseries.experiments.market.expt_settings.configs import ExperimentConfig
from timeseries.experiments.market.moo.utils.model import get_last_layer_weights, params_conversion_weights, \
    reconstruct_weights
from timeseries.utils.parallel import repeat_different_args


class TFT_Moo(Problem):

    def __init__(self,
                 model_folder,
                 data_formatter,
                 data_config,
                 loss_to_obj,
                 pop_size,
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

        self.models = []
        with tf.device('/device:GPU:0' if use_gpu else "/cpu:0"):
            for _ in range(pop_size):
                model = TemporalFusionTransformer(model_params)
                model.load(opt_manager.hyperparam_folder, use_keras_loadings=True)
                weights, last_layer = get_last_layer_weights(model)
                self.models.append((model, last_layer))

            self.orig_model = TemporalFusionTransformer(model_params)
            self.orig_model.load(opt_manager.hyperparam_folder, use_keras_loadings=True)
            weights, last_layer = get_last_layer_weights(self.orig_model)
            ind, w_params = params_conversion_weights(weights)

            print("*** Test Q-Loss with original weights ***")
            losses, unscaled_output_map = moo_q_loss_model(data_formatter, self.orig_model, valid,
                                                           return_output_map=True)

        self.use_gpu = use_gpu
        self.pop_size = pop_size
        self.data_formatter = data_formatter
        self.loss_to_obj = loss_to_obj
        self.valid = valid
        self.ini_weights = weights
        self.last_layer = last_layer
        self.weights_params = w_params
        self.original_losses = self.loss_to_obj(losses)
        self.eval = self.eval_func(loss_to_obj, not parallelize_pop, use_gpu)
        self.parallelize_pop = parallelize_pop

        n_var = len(ind)
        n_obj = len(self.original_losses)
        super().__init__(n_var, n_obj, n_constr=0, xl=0.0, xu=1.0, **kwargs)

    def eval_func(self, loss_to_obj, multi_processing=False, use_gpu=True):
        def run_moo_nn(x, model, last_layer, data_formatter, valid, w_params):
            with tf.device('/device:GPU:0' if use_gpu else "/cpu:0"):
                new_weights = reconstruct_weights(x, w_params)
                last_layer.set_weights(new_weights)
                return loss_to_obj(moo_q_loss_model(data_formatter, model, valid, multi_processing))

        return run_moo_nn

    def _evaluate(self, X, out, *args, **kwargs):
        if X.shape[0] > self.pop_size:
            raise Exception('evaluating {} individuals, but problem has only {} models'.format(X.shape[0],
                                                                                               self.pop_size))

        arg_placeholder = ['x', 'model', 'last_layer', self.data_formatter, self.valid, self.weights_params]
        args = []
        for i, x in enumerate(X):
            arg = copy.copy(arg_placeholder)
            arg[0] = x
            arg[1], arg[2] = self.models[i]
            args.append(arg)
        F = repeat_different_args(self.eval, args, parallel=self.parallelize_pop, n_jobs=2, backend='multiprocessing')
        # F = np.array(F)

        # calculate the function values in a parallelized manner and wait until done
        out["F"] = np.array(F)

        # F = []
        # for x in X:
        #     _, last_layer = get_last_layer_weights(self.model)
        #     new_weights = reconstruct_weights(x, self.weights_params)
        #     self.last_layer.set_weights(new_weights)
        #     f = self.loss_to_obj(moo_q_loss_model(self.data_formatter, self.model, self.valid))
        #     F.append(f)


import tempfile
from tensorflow.keras.models import load_model, save_model, Model
from tensorflow.python.keras.layers import deserialize, serialize
from tensorflow.python.keras.saving import saving_utils

# def unpack(model, training_config, weights):
#     restored_model = deserialize(model)
#     if training_config is not None:
#         restored_model.compile(
#             **saving_utils.compile_args_from_training_config(
#                 training_config
#             )
#         )
#     restored_model.set_weights(weights)
#     return restored_model
#
# # Hotfix function
# def make_keras_picklable():
#
#     def __reduce__(self):
#         model_metadata = saving_utils.model_metadata(self)
#         training_config = model_metadata.get("training_config", None)
#         model = serialize(self)
#         weights = self.get_weights()
#         return (unpack, (model, training_config, weights))
#
#     # cls = AttnModel
#     cls = Model
#     cls.__reduce__ = __reduce__

# Hotfix function
def make_keras_picklable(dir):
    def __getstate__(self):
        model_str = ""
        with tempfile.NamedTemporaryFile(suffix='.model', delete=True, dir=dir) as fd: #hdf5
            save_model(self, fd.name, overwrite=True)
            model_str = fd.read()
        d = {'model_str': model_str}
        return d

    def __setstate__(self, state):
        with tempfile.NamedTemporaryFile(suffix='.model', delete=True, dir=dir) as fd:
            fd.write(state['model_str'])
            fd.flush()
            model = load_model(fd.name)
        self.__dict__ = model.__dict__


    cls = Model
    cls.__getstate__ = __getstate__
    cls.__setstate__ = __setstate__

# Run the function


if __name__ == "__main__":
    name = 'snp'
    experiment_name = 'fixed'
    with tf.device("/cpu:0"):
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
        temp_dir = os.path.normpath(os.path.join(config.model_folder, '../..', '..', 'temp'))
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
        make_keras_picklable(temp_dir)

        formatter = config.make_data_formatter()

        # loss_to_obj = np.ndarray.flatten
        loss_to_obj = lambda x: np.mean(x, axis=0)
        pop_size = 2


        # problem = TFT_Moo(model_folder=os.path.join(config.model_folder, experiment_name),
        #                   data_formatter=formatter,
        #                   data_config=config.data_config,
        #                   loss_to_obj=loss_to_obj,
        #                   use_gpu=False,
        #                   pop_size=pop_size,
        #                   parallelize_pop=True)

        # X = np.random.rand(pop_size, problem.n_var)

        @delayed
        @wrap_non_picklable_objects
        def run_moo_nn(x, model, last_layer, data_formatter, valid, w_params):
            with tf.device("/cpu:0"):
                new_weights = reconstruct_weights(x, w_params)
                last_layer.set_weights(new_weights)
                return loss_to_obj(moo_q_loss_model(data_formatter, model, valid, multi_processing=False))


        # res = {}
        # t0 = time.time()
        # problem._evaluate(X, res)
        # print('Eval time: {} s'.format(round(time.time() - t0, 4)))
        # F = res['F']

        model_folder = os.path.join(config.model_folder, experiment_name)
        data_formatter = formatter
        train, valid, test = data_formatter.split_data(config.data_config)

        # Sets up default params
        fixed_params = data_formatter.get_experiment_params()
        params = data_formatter.get_default_model_params()
        params["model_folder"] = model_folder

        # Sets up hyperparam manager
        print("*** Loading hyperparm manager ***")
        opt_manager = HyperparamOptManager({k: [params[k]] for k in params}, fixed_params, model_folder)
        model_params = opt_manager.get_next_parameters()

        # arg_placeholder = ['x', 'model', 'last_layer', data_formatter, valid, problem.weights_params]
        # args = []

#%%
        from keras_pickle_wrapper import KerasPickleWrapper
        models = []
        with tf.device("/cpu:0"):
            for _ in range(pop_size):
                model = TemporalFusionTransformer(model_params)
                model.load(opt_manager.hyperparam_folder, use_keras_loadings=True)
                weights, last_layer = get_last_layer_weights(model)
                # model = KerasPickleWrapper(model.model)
                model = model.model
                # models.append(model)
                models.append((model, last_layer))

        base_model = TemporalFusionTransformer(model_params)
        data = base_model._batch_data(valid)
        inputs = data['inputs']

        args = []
        for model, layer in models:
            args.append((model, inputs))

        # %%
        minibatch = base_model.minibatch_size

        # @delayed
        # @wrap_non_picklable_objects
        def predict(model, inputs):
            with tf.device("/cpu:0"):
                # model = TemporalFusionTransformer(model_params)
                # model.load(opt_manager.hyperparam_folder, use_keras_loadings=True)

                # in_model = model.model
                # mw = KerasPickleWrapper(in_model)
                combined = model.predict(
                    inputs,
                    workers=1,
                    use_multiprocessing=False,
                    batch_size=minibatch)
                return combined

        # res = repeat(predict, (model, inputs), n_repeat=2, parallel=False, n_jobs=2)
        t0 = time.time()
        F = repeat_different_args(predict, args, parallel=True, n_jobs=2)
        print('exec time: {}s'.format(round(time.time() - t0, 4)))

#%%

        # executor = Parallel(n_jobs=2, backend='multiprocessing')
        # tasks = (predict(inputs) for _ in tqdm(range(2)))
        # # tasks = (predict(*arg) for arg in tqdm(args))
        # result = executor(tasks)

    #%%
        # result= []
        # for arg in tqdm(args):
        #     result.append(predict(*arg))

# %%

# f = run_moo_nn(x, model, layer, data_formatter, valid, problem.weights_params)
# F = np.array(F)

#
# times = []
# models = []
# for _ in range(10):
#     t0 = time.time()
#     model = TemporalFusionTransformer(model_params)
#     model.load(opt_manager.hyperparam_folder, use_keras_loadings=True)
#     weights, last_layer = get_last_layer_weights(model)
#     models.append((model, last_layer))
#     t1 = time.time() - t0
#     times.append(t1)
#     print('time: {} s'.format(round(t1, 4)))
# print('avg t: {}'.format(np.mean(times)))
#
# ind, w_params = params_conversion_weights(weights)
#
#
#
#
# F = []
# for model, last_layer in models:
#     print('predict')
#     new_rand_weights = [np.random.rand(*w.shape) for w in weights]
#     last_layer.set_weights(new_rand_weights)
#     f = loss_to_obj(moo_q_loss_model(data_formatter, model, valid))
#     F.append(f)
# F = np.array(F)

# %%
# algo_cfg = {'name': 'NSGA2',
#             'termination': ('n_gen', 100),
#             'max_gen': 150,
#             'pop_size': 100,
#             'hv_ref': [10] * 4  # used only for SMS-EMOA
#             }
#
# algorithm = get_algorithm(algo_cfg['name'],
#                           algo_cfg,
#                           n_obj=problem.n_obj,
#                           sampling=problem.original_losses)
#
# result = run_moo(problem, algorithm, algo_cfg, verbose=2)
