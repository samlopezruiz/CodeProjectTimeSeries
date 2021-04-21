import copy
from datetime import date
import os
import joblib


def models_strings(names, model_cfgs, suffix):
    models_info, models_name = '', ''
    for i, name in enumerate(names):
        models_info += '<br>' + name + ': ' + str(model_cfgs[i])
        models_name += name + '_'

    return models_info, models_name+suffix


def get_params(model_cfgs, functions, names, train):
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


def get_params(model, model_cfg):
    depth = model_cfg.get('depth', None)
    if depth is None:
        params = model.count_params()
    else:
        params = 2 ** (model_cfg.get('depth', 1) - 2) * 6
    return params


def save_vars(vars, file_path):
    today = date.today()
    print("saving .z")
    if not os.path.exists(file_path[0]):
        os.makedirs(file_path[0])
    path = file_path[:-1].copy() + [file_path[-1] + '_' + today.strftime("%Y_%m_%d") + ".z"]
    joblib.dump(vars, os.path.join(*path))