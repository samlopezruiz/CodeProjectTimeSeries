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

import datetime as dte
import os

# import data_formatter.base
# import expt_settings.configs
import algorithms.tft2.libs.utils as utils
import numpy as np
import pandas as pd
import tensorflow as tf

from algorithms.tft2.utils.data import extract_numerical_data
from timeseries.data.market.utils.names import get_inst_ohlc_names
from timeseries.experiments.market.data_formatter.base import InputTypes
from timeseries.experiments.market.expt_settings.configs import ExperimentConfig
from algorithms.tft2.expt_settings.save import save_forecasts
from algorithms.tft2.libs.tft_model import TemporalFusionTransformer
from algorithms.tft2.libs.hyperparam_opt import HyperparamOptManager
from timeseries.experiments.market.utils.preprocessing import reconstruct_from_ln_r, series_from_ln_r, reconstruct_forecasts
from timeseries.experiments.utils.files import get_new_file_path, save_vars

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
    train_samples, valid_samples = data_formatter.get_num_samples_for_calibration()

    # Sets up default params
    fixed_params = data_formatter.get_experiment_params()
    params = data_formatter.get_default_model_params()
    params["model_folder"] = model_folder

    # Sets up hyperparam manager
    print("*** Loading hyperparm manager ***")
    opt_manager = HyperparamOptManager({k: [params[k]] for k in params}, fixed_params, model_folder)

    print("*** Running tests ***")
    model_params = opt_manager.get_next_parameters()
    model = ModelClass(model_params, use_cudnn=use_gpu)

    model.load(opt_manager.hyperparam_folder, use_keras_loadings=True)

    print("Computing best validation loss")
    val_loss = model.evaluate(valid)

    print("Computing test loss")
    output_map = model.predict(test, return_targets=True)

    unscaled_output_map = {}
    for k, df in output_map.items():
        unscaled_output_map[k] = data_formatter.format_predictions(df)

    losses = {}
    targets = unscaled_output_map['targets']
    for q in model.quantiles:
        key = 'p{}'.format(int(q * 100))
        losses[key + '_loss'] = utils.numpy_normalised_quantile_loss(
            extract_numerical_data(targets), extract_numerical_data(unscaled_output_map[key]), q)

    print("Training completed @ {}".format(dte.datetime.now()))
    print("Best validation loss = {}".format(val_loss))
    print("\nNormalised Quantile Losses for Test Data: {}".format(
        [p_loss.mean() for k, p_loss in losses.items()]))

    results = {'quantiles': model.quantiles,
               'forecasts': unscaled_output_map,
               'losses': losses,
               'target': formatter.test_true_y.columns[0] if formatter.test_true_y is not None else None
               }
    return results


# tensorboard --logdir src/timeseries/models/market/outputs/saved_models/snp/fixed/logs/fit
if __name__ == "__main__":
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    print("Is Cuda Gpu Available: ", tf.test.is_gpu_available(cuda_only=True))
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
