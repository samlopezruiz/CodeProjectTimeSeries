import os

import joblib as joblib
import tensorflow as tf

import algorithms.tft2.libs.hyperparam_opt as hyperparam_opt
import algorithms.tft2.libs.tft_model
from timeseries.data.market.utils.names import get_inst_ohlc_names
from timeseries.experiments.market.expt_settings.configs import ExperimentConfig

ModelClass = algorithms.tft2.libs.tft_model.TemporalFusionTransformer


def get_attention_tft_model(use_gpu,
                            model_folder,
                            data_config,
                            data_formatter,
                            get_attentions=False,
                            samples=None):
    """Trains tft based on defined model params.

  Args:
    expt_name: Name of experiment
    use_gpu: Whether to run tensorflow with GPU operations
    model_folder: Folder path where models are serialized
    data_csv_path: Path to csv file containing data
    data_formatter: Dataset-specific data fromatter (see
      expt_settings.dataformatter.GenericDataFormatter)
    use_testing_mode: Uses a smaller models and data sizes for testing purposes
      only -- switch to False to use original default settings
  """

    print("Loading & splitting data...")
    train, valid, test = data_formatter.split_data(data_config)
    train_samples, valid_samples = data_formatter.get_num_samples_for_calibration()

    # Sets up default params
    fixed_params = data_formatter.get_experiment_params()
    params = data_formatter.get_default_model_params()
    params["model_folder"] = model_folder

    if samples is not None:
        valid_samples = samples

    # Sets up hyperparam manager
    print("*** Loading hyperparm manager ***")
    opt_manager = hyperparam_opt.HyperparamOptManager({k: [params[k]] for k in params}, fixed_params, model_folder)

    opt_manager.load_results()
    print("Params Selected:")
    for k in params:
        print("{}: {}".format(k, params[k]))

    best_params = opt_manager.get_best_params()

    with tf.device('/device:GPU:0' if use_gpu else "/cpu:0"):
        model = ModelClass(best_params)
        model.load(opt_manager.hyperparam_folder, use_keras_loadings=True)

        if not model.training_data_cached():
            model.cache_batched_data(valid, "valid", num_samples=valid_samples)

        attentions = model.get_attention() if get_attentions else None

    results = {'attentions': attentions,
               'params': best_params}
    return results


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
                              )
    formatter = config.make_data_formatter()

    model_folder = os.path.join(config.model_folder, experiment_name)
    results = get_attention_tft_model(use_gpu=True,
                                      model_folder=model_folder,
                                      data_config=config.data_config,
                                      data_formatter=formatter,
                                      get_attentions=True,
                                      samples=None)
    print('Saving File')
    joblib.dump(results, os.path.join(config.results_folder, experiment_name, 'attention_valid.z'))
    print('Done')
