import math
from collections import deque

import numpy as np
import time
from math import sqrt
from multiprocessing import cpu_count
from warnings import catch_warnings, filterwarnings
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAXResultsWrapper
from joblib import Parallel, delayed
from sklearn.metrics import mean_squared_error


def walk_forward_step_forecast_werrs(train, test, cfg, model_forecast, model_fit, steps=1, verbose=0):
    is_mv = len(train.shape) > 1
    predictions = list()
    history, test_bundles, y_test = get_bundles(is_mv, steps, test, train)
    errors = deque(maxlen=cfg['in_steps'])
    for _ in range(cfg['in_steps']):
        errors.append(0)
    model = model_fit(train, cfg)
    start_time = time.time()

    for bundle, y in zip(test_bundles, y_test):
        bundle = reshape_bundle(bundle, is_mv)
        # print_progress(i, test_bundles, verbose)
        yhat = model_forecast(model, steps=steps, history=history, errors=errors, cfg=cfg)
        errors.append(bundle[-1] - yhat)
        [predictions.append(y) for y in yhat] if steps > 1 else predictions.append(yhat)
        history = np.vstack([history, bundle]) if is_mv else np.hstack([history, bundle])
    try:
        error = measure_rmse(y_test, predictions[:len(y_test)])
    except:
        print("error in rmse")
        error = None
    print_pred_time(start_time, test_bundles, verbose)
    return error, predictions[:len(y_test)]


def walk_forward_step_forecast(train, test, cfg, model_forecast, model_fit, steps=1, verbose=0, plot_hist=False):
    is_mv = len(train.shape) > 1
    predictions = list()
    history, test_bundles, y_test = get_bundles(is_mv, steps, test, train)

    model = model_fit(train, cfg, plot_hist=plot_hist, verbose=verbose)
    start_time = time.time()
    for i, bundle in enumerate(test_bundles):
        bundle = reshape_bundle(bundle, is_mv)
        print_progress(i, test_bundles, verbose+1)
        yhat = model_forecast(model, steps=steps, history=history, cfg=cfg)
        [predictions.append(y) for y in yhat] if steps > 1 else predictions.append(yhat)
        history = np.vstack([history, bundle]) if is_mv else np.hstack([history, bundle])

    print_pred_time(start_time, test_bundles, verbose)
    predictions = prep_forecast(predictions)
    return predictions[:len(y_test)]


def prep_forecast(forecast):
    forecast = np.array(forecast)
    if len(forecast.shape) == 2:
        if forecast.shape[1] == 1:
            # case of an array of arrays
            forecast = forecast.ravel()
    return forecast


def reshape_bundle(bundle, is_mv):
    if is_mv and len(bundle.shape) == 1:
        return bundle.reshape(1, bundle.shape[0])
    if not is_mv and len(bundle.shape) > 1:
        return bundle.ravel()
    return bundle


def get_bundles(is_mv, steps, test, train):
    train = np.array(train) if is_mv else train.reshape(train.shape[0], 1)
    test = np.array(test) if is_mv else test.reshape(test.shape[0], 1)
    # remove last column
    history = np.array(train[:, :-1]) if is_mv else train.ravel()
    X_test, y_test = (test[:, :-1], test[:, -1]) if is_mv else (test, test.ravel())
    # step over each time-step in the test set
    test_bundles = [X_test[i:i + steps] for i in range(0, X_test.shape[0], steps)] if steps > 1 else X_test
    return history, test_bundles, y_test


def print_progress(i, test_bundles, verbose):
    if verbose >= 2 and i % 50 == 0:
        print("{}/{} - {}% predictions done".format(i, len(test_bundles), round(i * 100 / len(test_bundles))),
              end='\r')


def print_pred_time(start_time, test_bundles, verbose):
    end_time = time.time()
    if verbose >= 1:
        print("{} predictions in {}s: avg: {}s".format(len(test_bundles), round(end_time - start_time, 2),

                                                       round((end_time - start_time) / len(test_bundles), 4)))
def append_data_to_model(bundle, model, steps):
    if steps == 1:
        model = model.append([bundle])
    else:
        model = model.append(bundle)
    return model


def measure_rmse(actual, predicted):
    return sqrt(mean_squared_error(actual, predicted))


# split a univariate dataset into train/test sets
def train_test_split(data, n_test):
    return data[:-n_test], data[-n_test:]


# score a models, return None on failure
def score_model(train, test, cfg, model_forecast, model_creation, steps=1, debug=False):
    key = str(cfg)
    if debug:
        result, _ = walk_forward_step_forecast(train, test, cfg, model_forecast, model_creation, steps=steps)
    else:
        # one failure during models validation suggests an unstable config
        try:
            # never show warnings when grid searching, too noisy
            with catch_warnings():
                filterwarnings("ignore")
                result, _ = walk_forward_step_forecast(train, test, cfg, model_forecast, model_creation, steps=steps)
        except:
            result = None
    # check for an interesting result
    if result is not None:
        print(' > Model[%s] %.3f' % (key, result))
    return key, result


# grid search configs
def grid_search(train, test, cfg_list, model_forecast, model_fit, parallel=True):
    if parallel:
        # execute configs in parallel
        executor = Parallel(n_jobs=cpu_count(), backend='multiprocessing')
        tasks = (delayed(score_model)(train, test, cfg, model_forecast, model_fit) for cfg in cfg_list)
        scores = executor(tasks)
    else:
        scores = [score_model(train, test, cfg, model_forecast, model_fit) for cfg in cfg_list]
    # remove empty results
    scores = [r for r in scores if r[1] is not None]
    # sort configs by error, asc
    scores.sort(key=lambda tup: tup[1])
    return scores
