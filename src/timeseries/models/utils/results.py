import os
import joblib
import numpy as np
import pandas as pd


def rename_ensemble(df):
    as_list = df.index.tolist()
    ixs = np.flatnonzero(np.core.defchararray.find(as_list, '&') != -1)
    for ix in ixs:
        as_list[ix] = 'ENSEMBLE'
    df.index = as_list
    return df


def load_results(cfg, back_folders=4, result_folder='results'):
    suffix, series, steps = cfg['suffix'], cfg['series'], cfg['steps']
    date, model, stage = cfg['date'], cfg['model'], cfg['stage']
    file_name = suffix + '_' + series + str(steps) + '_' + date + '.z'
    back = ['..'] * back_folders

    path = os.path.join(*back, result_folder, model, stage, file_name)
    return joblib.load(path)


def get_col_and_rename(d, e, var, res_cfg):
    d = d.loc[:, [var[0]]]
    if var[1] is None:
        e['none'] = 0
        e = e.loc[:, ['none']]
    else:
        e = e.loc[:, [var[1]]]
    d.columns = [col + ' STEPS OUT=' + str(res_cfg['steps']) for col in d.columns]
    e.columns = [col + ' STEPS OUT=' + str(res_cfg['steps']) for col in e.columns]
    return d, e


def concat_sort_results(dat, err):
    data = pd.concat(dat, axis=1)
    errors = pd.concat(err, axis=1)
    data.sort_values([data.columns[0]], ascending=False, inplace=True)
    errors = errors.loc[data.index, :]
    return data, errors