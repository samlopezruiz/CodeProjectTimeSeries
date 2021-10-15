import os

import tensorflow as tf

from algorithms.tft2.libs.hyperparam_opt import HyperparamOptManager
from algorithms.tft2.libs.tft_model import TemporalFusionTransformer
from timeseries.data.market.utils.names import get_inst_ohlc_names
from timeseries.experiments.market.expt_settings.configs import ExperimentConfig
from timeseries.experiments.market.moo.utils.model import get_last_layer_weights, params_conversion_weights, \
    reconstruct_weights

ModelClass = TemporalFusionTransformer

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
    data_formatter = config.make_data_formatter()
    model_folder = os.path.join(config.model_folder, experiment_name)
    print("*** Training from defined parameters for {} ***".format(experiment_name))

    print("Loading & splitting data...")
    train, valid, test = data_formatter.split_data(config.data_config)
    train_samples, valid_samples = data_formatter.get_num_samples_for_calibration()

    # Sets up default params
    fixed_params = data_formatter.get_experiment_params()
    params = data_formatter.get_default_model_params()
    params["model_folder"] = model_folder

    # Parameter overrides for testing only! Small sizes used to speed up script.

    fixed_params["num_epochs"] = 1
    params["hidden_layer_size"] = 7
    train_samples, valid_samples = 100, 10

    # Sets up hyperparam manager
    print("*** Loading hyperparm manager ***")
    opt_manager = HyperparamOptManager({k: [params[k]] for k in params}, fixed_params, model_folder)

    params = opt_manager.get_next_parameters()
    model = ModelClass(params, use_cudnn=True)

    if not model.training_data_cached():
        model.cache_batched_data(train, "train", num_samples=train_samples)
        model.cache_batched_data(valid, "valid", num_samples=valid_samples)

    model.fit()

    #%%
    weights, last_layer = get_last_layer_weights(model)
    ind, params = params_conversion_weights(weights)
    new_weights = reconstruct_weights(ind, params)

    last_layer.set_weights(new_weights)

    import numpy as np

    new_rand_weights = [np.random.rand(*w.shape) for w in new_weights]

