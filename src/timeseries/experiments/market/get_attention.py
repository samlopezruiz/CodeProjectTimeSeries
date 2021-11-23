import os

import joblib as joblib
import tensorflow as tf

import algorithms.tft2.libs.hyperparam_opt as hyperparam_opt
from timeseries.experiments.market.utils.filename import get_result_folder
from timeseries.experiments.market.utils.harness import get_model_data_config, get_model
from timeseries.experiments.utils.files import save_vars


def get_attention_model(use_gpu,
                        architecture,
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
    Model = get_model(architecture)

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

    params = opt_manager.get_next_parameters()
    print("Params Selected:")
    for k in params:
        print("{}: {}".format(k, params[k]))

    with tf.device('/device:GPU:0' if use_gpu else "/cpu:0"):
        model = Model(params)
        model.load(opt_manager.hyperparam_folder, use_keras_loadings=True)

        if not model.training_data_cached():
            model.cache_batched_data(valid, "valid", num_samples=valid_samples)

        attentions = model.get_attention() if get_attentions else None

    results = {'attentions': attentions,
               'params': params}
    return results


if __name__ == "__main__":
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    print("Is Cuda Gpu Available: ", tf.test.is_gpu_available(cuda_only=True))
    print("Device Name: ", tf.test.gpu_device_name())
    print('TF eager execution: {}'.format(tf.executing_eagerly()))
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    print("Device Name: ", tf.test.gpu_device_name())
    print('TF eager execution: {}'.format(tf.executing_eagerly()))

    general_cfg = {'save_forecast': True,
                   'save_plot': False,
                   'use_all_data': True,
                   }

    results_cfg = {'formatter': 'snp',
                   'experiment_name': '5t_ema_q258',
                   'results': 'TFTModel_ES_ema_r_q258_lr005_pred'
                   }

    moo_results = joblib.load(os.path.join(get_result_folder(results_cfg), results_cfg['results'] + '.z'))

    config, formatter, model_folder = get_model_data_config(moo_results['experiment_cfg'],
                                                            moo_results['model_cfg'],
                                                            moo_results['fixed_cfg'])

    experiment_cfg = moo_results['experiment_cfg']

    results = get_attention_model(use_gpu=True,
                                  architecture=experiment_cfg['architecture'],
                                  model_folder=model_folder,
                                  data_config=config.data_config,
                                  data_formatter=formatter,
                                  get_attentions=True,
                                  samples=None)
    print('Saving File')
    save_vars(results,
              file_path=os.path.join(get_result_folder(results_cfg), 'attention_valid'))
    print('Done')
