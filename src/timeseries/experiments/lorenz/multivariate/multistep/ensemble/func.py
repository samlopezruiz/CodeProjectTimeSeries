import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
from itertools import chain, combinations
from itertools import combinations
import time
#%%


def ensemble_multi_step_mv_fit(train, cfg, plot_hist=False, verbose=0):
    losses = []
    models = []
    start_time = time.time()
    for name, input_cfg, model_cfg, func_cfg in cfg:
        model, _, loss = func_cfg[1](train, model_cfg, plot_hist=plot_hist, verbose=verbose)
        models.append(model)
        losses.append(loss)
    train_time = round((time.time() - start_time), 2)

    return models, train_time, np.mean(losses)


def ensemble_multi_step_mv_predict(models, history, cfg, steps=1):
    preds = []
    for model, cfg in zip(models, cfg):
        name, input_cfg, model_cfg, func_cfg = cfg
        pred = func_cfg[0](model, history, model_cfg, steps=steps)
        preds.append(pred)
    preds = np.array(preds)

    return preds.mean(axis=0)


def ensemble_get_multi_step_mv_funcs():
    return [ensemble_multi_step_mv_predict, ensemble_multi_step_mv_fit]


def get_ensemble_steps_cfgs(model_cfg):
    def powerset(iterable):
        s = list(iterable)
        return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))

    cfgs = list(powerset(model_cfg))
    cfg2 = []
    for cfg in cfgs:
        cfg2.append(list(cfg))
    return cfg2[1:]