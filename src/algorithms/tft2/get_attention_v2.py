import os

import joblib as joblib
import pandas as pd

import libs.hyperparam_opt as hyperparam_opt
import libs.tft_model
from algorithms.tft2.expt_settings.configs import ExperimentConfig

ModelClass = libs.tft_model.TemporalFusionTransformer


def get_attention_tft_model(use_gpu,
                            model_folder,
                            data_csv_path,
                            data_formatter,
                            get_attentions=False):
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
    raw_data = pd.read_csv(data_csv_path, index_col=0)
    train, valid, test = data_formatter.split_data(raw_data)
    train_samples, valid_samples = data_formatter.get_num_samples_for_calibration()

    # Sets up default params
    fixed_params = data_formatter.get_experiment_params()
    params = data_formatter.get_default_model_params()
    params["model_folder"] = model_folder

    # Sets up hyperparam manager
    print("*** Loading hyperparm manager ***")
    opt_manager = hyperparam_opt.HyperparamOptManager({k: [params[k]] for k in params}, fixed_params, model_folder)

    opt_manager.load_results()
    print("Params Selected:")
    for k in params:
        print("{}: {}".format(k, params[k]))

    best_params = opt_manager.get_best_params()
    model = ModelClass(best_params)
    model.load(opt_manager.hyperparam_folder, use_keras_loadings=True)

    attentions = model.get_attention(valid) if get_attentions else None

    results = {'model': model,
               'attentions': attentions,
               'params': best_params}
    return results


if __name__ == "__main__":
    name = 'volatility'
    experiment_name = 'fixed_complete'
    config = ExperimentConfig(name, None)
    formatter = config.make_data_formatter()

    model_folder = os.path.join(config.model_folder, experiment_name)
    results = get_attention_tft_model(use_gpu='yes',
                                      model_folder=model_folder,
                                      data_csv_path=config.data_csv_path,
                                      data_formatter=formatter,
                                      get_attentions=True)
    del results['model']
    joblib.dump(results, os.path.join(config.results_folder, experiment_name, 'attention_valid.z'))
