# root mean squared error or rmse
import math
import time
from math import sqrt
from multiprocessing import cpu_count
from warnings import catch_warnings, filterwarnings
import statsmodels
from joblib import Parallel, delayed
from sklearn.metrics import mean_squared_error


# walk-forward validation for univariate data
def walk_forward_validation(data, n_test, cfg, model_forecast, verbose=False):
    predictions = list()
    # split dataset
    train, test = train_test_split(data, n_test)
    # seed history with training dataset
    history = [x for x in train]
    # step over each time-step in the test set
    for i in range(len(test)):
        if verbose:
            print(i, "/", len(test))
        # fit model_forecast and make forecast for history
        yhat = model_forecast(history, cfg)
        # store forecast in list of predictions
        predictions.append(yhat)
        # add actual observation to history for the next loop
        history.append(test[i])
    # estimate prediction error
    try:
        error = measure_rmse(test, predictions)
    except:
        error = None
    return error, predictions


def walk_forward_step_forecast(train, test, cfg, model_forecast, model_creation, steps=1, verbose=False):
    predictions = list()
    history = list(train)
    # step over each time-step in the test set
    if steps > 1:
        test_bundles = [test[i:i + steps] for i in range(0, len(test), steps)]
    else:
        test_bundles = list(test)

    model = model_creation(train, cfg)
    append_as_list = isinstance(model, statsmodels.tsa.statespace.sarimax.SARIMAXResultsWrapper)

    start_time = time.time()
    for i, bundle in enumerate(test_bundles):
        if verbose and i % 50 == 0:
            print(i, "/", len(test_bundles))
        # fit model_forecast and make forecast for history
        yhat = model_forecast(model, steps=steps)
        model = model.append([bundle]) if append_as_list else model.append(bundle)

        # store forecast in list of predictions
        [predictions.append(y) for y in yhat] if steps > 1 else predictions.append(yhat)
        [history.append(t) for t in bundle] if steps > 1 else history.append(bundle)
    try:
        error = measure_rmse(test, predictions[:len(test)])
    except:
        print("error in mse")
        error = None
    print("{} predictions in {}s: avg: {}".format(len(test_bundles), round(time.time() - start_time, 2), round(len(test_bundles)/(time.time() - start_time),2)))
    return error, predictions[:len(test)]


def measure_rmse(actual, predicted):
    return sqrt(mean_squared_error(actual, predicted))


# split a univariate dataset into train/test sets
def train_test_split(data, n_test):
    return data[:-n_test], data[-n_test:]


# score a model_forecast, return None on failure
def score_model(data, test_size, cfg, model_forecast, model_creation, steps=1, debug=False):
    train, test = train_test_split(data, test_size)
    key = str(cfg)
    if debug:
        result, _ = walk_forward_step_forecast(train, test, cfg, model_forecast, model_creation, steps=steps)
    else:
        # one failure during model_forecast validation suggests an unstable config
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
def grid_search(data, cfg_list, test_size, model_forecast, model_creation, parallel=True):
    if parallel:
        # execute configs in parallel
        executor = Parallel(n_jobs=cpu_count(), backend='multiprocessing')
        tasks = (delayed(score_model)(data, test_size, cfg, model_forecast, model_creation) for cfg in cfg_list)
        scores = executor(tasks)
    else:
        scores = [score_model(data, test_size, cfg, model_forecast, model_creation) for cfg in cfg_list]
    # remove empty results
    scores = [r for r in scores if r[1] is not None]
    # sort configs by error, asc
    scores.sort(key=lambda tup: tup[1])
    return scores
