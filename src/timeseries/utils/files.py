import datetime
import os

import joblib
import pandas as pd

from timeseries.experiments.utils.metrics import get_data_error
from timeseries.experiments.utils.results import load_results, rename_ensemble, get_col_and_rename, concat_sort_results


def new_dir(file_path):
    if not os.path.exists(file_path):
        os.makedirs(file_path)

def save_csv(df, res_cfg, suffix=None):
    if res_cfg['save_results']:
        if suffix is None:
            csv_file_name = res_cfg['preffix'] + '_' + res_cfg['date'] + '_' + datetime.datetime.now().strftime(
                "%Y_%m_%d_%H-%M") + ".csv"
        else:
            csv_file_name = res_cfg['preffix'] + '_' + res_cfg['date'] + '_' + suffix + '_' + datetime.datetime.now().strftime(
                "%Y_%m_%d_%H-%M") + ".csv"
        new_dir('csv')
        df.to_csv(os.path.join('csv', csv_file_name))


def load_data_err(res_cfg, compare, var=None):
    dat, err = [], []
    for op in compare[1]:
        if compare[0] in res_cfg:
            res_cfg[compare[0]] = op
            suffix = None
        else:
            suffix = op
        (in_cfg, input_cfg, names, model_cfgs, summary), model_name = load_results(res_cfg, suffix=suffix)
        summary = rename_ensemble(summary)
        if var is None:
            var = (in_cfg['score_type'], 'score_std')

        d, e = get_data_error(summary, in_cfg['score_type'])
        d, e = get_col_and_rename(d, e, var, compare[0], op)
        dat.append(d)
        err.append(e)
    data, errors = concat_sort_results(dat, err)
    return data, errors, input_cfg

