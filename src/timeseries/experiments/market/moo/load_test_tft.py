# Lint as: python3
"""Trains TFT based on a defined set of parameters.

Uses default parameters supplied from the configs file to train a TFT model from
scratch.

Usage:
python3 script_train_fixed_params {expt_name} {output_folder}

Command line args:
  expt_name: Name of dataset/experiment to train.
  output_folder: Root folder in which experiment is saved


"""

import os

import numpy as np
import tensorflow as tf

from algorithms.tft2.expt_settings.save import save_forecasts
from algorithms.tft2.harness.train_test import valid_test_model, moo_q_loss_model
from algorithms.tft2.libs.hyperparam_opt import HyperparamOptManager
from algorithms.tft2.libs.tft_model import TemporalFusionTransformer
from timeseries.data.market.utils.names import get_inst_ohlc_names
from timeseries.experiments.market.expt_settings.configs import ExperimentConfig
from timeseries.experiments.market.moo.utils.model import get_last_layer_weights, params_conversion_weights, \
    reconstruct_weights
from timeseries.experiments.market.utils.preprocessing import reconstruct_forecasts

ModelClass = TemporalFusionTransformer


def load_test_tft(expt_name,
                  use_gpu,
                  model_folder,
                  data_config,
                  data_formatter):
    """Trains tft based on defined model params.

  Args:
    expt_name: Name of experiment
    use_gpu: Whether to run tensorflow with GPU operations
    model_folder: Folder path where models are serialized
    data_config: Data input file configurations
    data_formatter: Dataset-specific data fromatter (see
      expt_settings.dataformatter.GenericDataFormatter)
    use_testing_mode: Uses a smaller models and data sizes for testing purposes
      only -- switch to False to use original default settings
  """

    num_repeats = 1

    print("*** Training from defined parameters for {} ***".format(expt_name))

    print("Loading & splitting data...")
    train, valid, test = data_formatter.split_data(data_config)

    # Sets up default params
    fixed_params = data_formatter.get_experiment_params()
    params = data_formatter.get_default_model_params()
    params["model_folder"] = model_folder

    # Sets up hyperparam manager
    print("*** Loading hyperparm manager ***")
    opt_manager = HyperparamOptManager({k: [params[k]] for k in params}, fixed_params, model_folder)

    print("*** Running tests ***")
    model_params = opt_manager.get_next_parameters()

    models = []
    for _ in range(3):
        model = ModelClass(model_params, use_cudnn=use_gpu)
        model.load(opt_manager.hyperparam_folder, use_keras_loadings=True)
        weights, last_layer = get_last_layer_weights(model)
        models.append((model, weights, last_layer))

    for model, weights, last_layer in models:
        new_rand_weights = [np.random.rand(*w.shape) for w in weights]
        last_layer.set_weights(new_rand_weights)
        losses, unscaled_output_map = moo_q_loss_model(data_formatter, model, test, return_output_map=True)
        print(losses)

    # weights, last_layer = get_last_layer_weights(model)
    # ind, params = params_conversion_weights(weights)
    # new_weights = reconstruct_weights(ind, params)
    # new_rand_weights = [np.random.rand(*w.shape) for w in new_weights]
    # last_layer.set_weights(new_rand_weights)
    #
    # losses_2 = moo_q_loss_model(data_formatter, model, test, return_output_map=False)
    # print(losses_2)

    # results = {'quantiles': fixed_params['quantiles'],
    #            'forecasts': unscaled_output_map,
    #            'losses': losses,
    #            'target': formatter.test_true_y.columns[0] if formatter.test_true_y is not None else None
    #            }
    # return results


# tensorboard --logdir src/timeseries/models/market/outputs/saved_models/snp/fixed/logs/fit
if __name__ == "__main__":
    tf.autograph.set_verbosity(0)
    print("Is Cuda Gpu Available: ", tf.config.list_physical_devices('GPU'))
    print("Device Name: ", tf.test.gpu_device_name())
    print('TF eager execution: {}'.format(tf.executing_eagerly()))

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
    save_forecast = False

    results = load_test_tft(expt_name=name,
                            use_gpu=True,
                            model_folder=os.path.join(config.model_folder, experiment_name),
                            data_config=config.data_config,
                            data_formatter=formatter)

    results['reconstructed_forecasts'] = reconstruct_forecasts(formatter, results['forecasts'])
    if save_forecast:
        save_forecasts(config, experiment_name, results)
