import os
from os import listdir

import joblib
import pandas as pd

from timeseries.utils.dataframes import renamer

DATA_ROOT = 'D:\\MEGA\\Proyectos\\Trading\\Algo_Trading\\Historical_Data'
PROJECT_ROOT = 'D:\\MEGA\\CienciasDeLaComputacion\\Tesis\\CodeProjectTimeSeries\\src\\timeseries'


def get_data_root():
    return DATA_ROOT


def get_project_root():
    return PROJECT_ROOT


def load_files(data_cfg, subfolder, last_folder='src_folder', end=".z"):
    path = get_model_market_path(data_cfg, subfolder=subfolder, last_folder=last_folder)
    filename = data_cfg.get('filename', None)
    if filename is not None:
        filename = filename + end
        print("\nLoading", filename)
        if filename.endswith(".csv") or filename.endswith(".txt"):
            data = read_csv(os.path.join(path, filename))
        else:
            data = joblib.load(os.path.join(path, filename))
        return data
    else:
        raise Exception('filename not found in data_cfg')


def get_model_market_path(data_cfg, subfolder='split', last_folder='src_folder'):
    src_folder = data_cfg.get(last_folder, 'res')
    return os.path.join(PROJECT_ROOT, 'experiments', 'market', subfolder, src_folder)


def get_market_path(data_cfg, last_folder='src_folder'):
    sampling, src_folder = data_cfg['sampling'], data_cfg[last_folder]
    inst, market = data_cfg['inst'], data_cfg['market']
    return os.path.join(DATA_ROOT, market, sampling, inst, src_folder)


def list_files(data_cfg, suffix=".txt", last_folder='src_folder', include_substring=''):
    path = get_market_path(data_cfg, last_folder=last_folder)
    files = find_filenames(path, suffix=suffix, include_substring=include_substring)
    return files


def find_filenames(path_to_dir, suffix=".txt", include_substring=''):
    filenames = listdir(path_to_dir)
    return [filename for filename in filenames if filename.endswith(suffix) and include_substring in filename]


def load_market(data_cfg, end=".csv", last_folder='src_folder'):
    path = get_market_path(data_cfg, last_folder=last_folder)
    filename = data_cfg.get('filename', None)
    if filename is None:
        inst, suffix = data_cfg['inst'], data_cfg['suffix']
        filename = inst + "_" + suffix + end
    dataset = load_data(filename, path)
    return dataset


def load_multiple_markets(data_cfgs, resampling='D', ffill=True):
    data = [load_market(cfg) for cfg in data_cfgs]
    data_resampled = [df.resample(resampling).last() for df in data]
    df = pd.concat(data_resampled, axis=1)
    df.rename(columns=renamer(), inplace=True)
    if ffill:
        df.ffill(axis=0, inplace=True)
        df.dropna(inplace=True)
    return df


def load_data(filename, path):
    dataset = None
    print("\nLoading", filename)
    try:
        if filename.endswith(".csv") or filename.endswith(".txt"):
            dataset = read_csv(os.path.join(path, filename))
        else:
            dataset = joblib.load(os.path.join(path, filename))
    except Exception as ex:
        print(ex)

    return dataset


def save_df(df, data_cfg, timestamp=True, last_folder='src_folder', end='.csv', suffix=''):
    filename = data_cfg.get('filename', None)
    if filename is None:
        inst = data_cfg['inst']
        if timestamp:
            ini_date = str(df.index[0].year) + '_' + str(df.index[0].month)
            end_date = str(df.index[-1].year) + '_' + str(df.index[-1].month)
        sufx = ('_' + suffix) if len(suffix) > 1 else ''
        filename = inst + "_" + ini_date + "-" + end_date + sufx + end
    path = get_market_path(data_cfg, last_folder=last_folder)
    df.to_csv(os.path.join(path, filename), index=True)
    print("File {} saved".format(filename))


def describe(df):
    print('Initial Date : ' + str(df.index[0]))
    print('Final Date   : ' + str(df.index[-1]))
    print('Dataset Shape: ' + str(df.shape))


def read_csv(file_path, datetime_col='datetime'):
    return pd.read_csv(file_path, header=0, infer_datetime_format=True,
                       parse_dates=[datetime_col], index_col=[datetime_col],
                       converters={'vol': eval, 'volp': eval})


def load_file(file_path, end=".z"):
    file_path[-1] = file_path[-1] + end
    path = os.path.join(*file_path)
    if os.path.exists(path):
        print("\nLoading", file_path[-1])
        if file_path[-1].endswith(".csv") or file_path[-1].endswith(".txt"):
            data = read_csv(path)
        else:
            data = joblib.load(path)
        return data
    else:
        raise Exception("file {} doesn't exist".format(path))
