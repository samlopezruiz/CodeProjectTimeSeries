import datetime
import os

import pandas as pd

from timeseries.experiments.utils.files import get_new_file_path, save_vars


def save_forecasts(config, experiment_name, results, use_date_suffix=False):
    save_folder = os.path.join(config.results_folder, experiment_name)

    # config.results_folder comes as absolute path and a '\' has to be added
    # after the disk letter
    path_list = save_folder.split('\\')
    path_list[0] = path_list[0] + '\\'

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    print('Saving forecasts to: {}'.format(os.path.join(*path_list)))
    save_vars(results, path_list + ['test_forecasts'], use_date_suffix)
    for key in results:
        if isinstance(results[key], pd.DataFrame):
            results[key].to_csv(os.path.join(*get_new_file_path(path_list + [key], '.csv', use_date_suffix)))
        # else:
        #     print('key={} is not a pandas DataFrame'.format(key))
