# root mean squared error or rmse
from math import sqrt
from multiprocessing import cpu_count
from warnings import catch_warnings, filterwarnings

from joblib import Parallel, delayed
from sklearn.metrics import mean_squared_error


# walk-forward validation for univariate data
def walk_forward_validation(data, n_test, cfg, model_forecast):
    predictions = list()
    # split dataset
    train, test = train_test_split(data, n_test)
    # seed history with training dataset
    history = [x for x in train]
    # step over each time-step in the test set
    for i in range(len(test)):
        # fit model and make forecast for history
        yhat = model_forecast(history, cfg)
        # store forecast in list of predictions
        predictions.append(yhat)
        # add actual observation to history for the next loop
        history.append(test[i])
    # estimate prediction error
    error = measure_rmse(test, predictions)
    return error


def measure_rmse(actual, predicted):
    return sqrt(mean_squared_error(actual, predicted))


# split a univariate dataset into train/test sets
def train_test_split(data, n_test):
    return data[:-n_test], data[-n_test:]


# score a model, return None on failure
def score_model(data, n_test, cfg, model_forecast, debug=False):
    result = None
    # convert config to a key
    key = str(cfg)
    # show all warnings and fail on exception if debugging
    if debug:
        result = walk_forward_validation(data, n_test, cfg, model_forecast)
    else:
        # one failure during model validation suggests an unstable config
        try:
            # never show warnings when grid searching, too noisy
            with catch_warnings():
                filterwarnings("ignore")
                result = walk_forward_validation(data, n_test, cfg, model_forecast)
        except:
            error = None
    # check for an interesting result
    if result is not None:
        print(' > Model[%s] %.3f' % (key, result))
    return key, result


# grid search configs
def grid_search(data, cfg_list, n_test, model_forecast, parallel=True):

    if parallel:
        # execute configs in parallel
        executor = Parallel(n_jobs=cpu_count(), backend='multiprocessing')
        tasks = (delayed(score_model)(data, n_test, cfg, model_forecast) for cfg in cfg_list)
        scores = executor(tasks)
    else:
        scores = [score_model(data, n_test, cfg, model_forecast) for cfg in cfg_list]
    # remove empty results
    scores = [r for r in scores if r[1] is not None]
    # sort configs by error, asc
    scores.sort(key=lambda tup: tup[1])
    return scores
