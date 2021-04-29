import copy
from datetime import date
import os
import joblib
import datetime
from timeseries.models.utils.config import unpack_input_cfg


def models_strings(names, model_cfgs, suffix):
    models_info, models_name = '', ''
    for i, name in enumerate(names):
        models_info += '<br>' + name + ': ' + str(model_cfgs[i])
        models_name += name + '_'

    return models_info, models_name+suffix


def get_models_params(model_cfgs, functions, names, train):
    n_params = []
    for i, model_cfg in enumerate(model_cfgs):
        funcs = functions[i]
        params = calc_param(funcs, model_cfg, train)
        n_params.append((names[i], params))
    return n_params


def calc_param(functions, model_cfg, train):
    depth = model_cfg.get('depth', None)
    if depth is None:
        model_cfg_ = copy.copy(model_cfg)
        model_cfg_['n_epochs'] = 1
        model, _ = functions[1](train, model_cfg_)
        params = model.count_params()
    else:
        params = 2 ** (model_cfg.get('depth', 1) - 2) * 6
    return params


def save_vars(vars, file_path=None, save_results=True):
    if save_results:
        if file_path is None:
            file_path = ['results', 'result']
        print("saving .z")
        if not os.path.exists(file_path[0]):
            os.makedirs(file_path[0])
        path = file_path[:-1].copy() + [file_path[-1] + '_' + datetime.datetime.now().strftime("%Y_%m_%d_%H-%M") + ".z"]
        # path = file_path[:-1].copy() + [file_path[-1] + '_' + today.strftime("%Y_%m_%d") + ".z"]
        joblib.dump(vars, os.path.join(*path))


def save_gs_results(input_cfg, gs_cfg, model_cfg, in_cfg, vars):
    file_name =  '_'.join(gs_cfg.keys()) + '_' + get_suffix(input_cfg, model_cfg)
    save_vars(vars, [in_cfg['results_folder'], file_name], in_cfg['save_results'])


def get_suffix(input_cfg, model_cfg):
    variate, granularity, noise, trend, detrend, preprocess = unpack_input_cfg(input_cfg)
    steps = model_cfg['n_steps_out']
    suffix = ''
    if trend:
        suffix += 'trend_'
    if noise:
        suffix += 'noise_'
    suffix += 's' + str(steps)
    return suffix


def get_params(model, model_cfg):
    depth = model_cfg.get('depth', None)
    if depth is None:
        params = model.count_params()
    else:
        params = 2 ** (model_cfg.get('depth', 1) - 2) * 6
    return params