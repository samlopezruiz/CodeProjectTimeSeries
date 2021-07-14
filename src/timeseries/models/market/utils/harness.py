import time

import numpy as np
import tensorflow as tf
from algorithms.nnhmm.func import nnhmm_fit, nnhmm_predict
from timeseries.models.market.utils.console import print_progress, print_pred_time
from timeseries.models.market.utils.models import get_params
from timeseries.models.market.utils.preprocessing import reconstruct_pred
from timeseries.models.utils.config import unpack_in_cfg
from timeseries.models.utils.forecast import multi_step_forecast_df, merge_forecast_df
from timeseries.models.utils.metrics import forecast_accuracy
from timeseries.models.utils.models import get_suffix
from timeseries.plotly.plot import plotly_time_series
from timeseries.preprocessing.func import ismv


def reshape_bundle(bundle, is_mv=True):
    if is_mv and len(bundle.shape) == 1:
        return bundle.reshape(1, bundle.shape[0])
    if not is_mv and len(bundle.shape) > 1:
        return bundle.ravel()
    return bundle


def prep_forecast(forecast):
    forecast = np.array(forecast)
    if len(forecast.shape) == 2:
        if forecast.shape[1] == 1:
            # case of an array of arrays
            forecast = forecast.ravel()
    return forecast


def walk_forward_forecast(train, test, reg_prob_train, reg_prob_test, cfg, model_funcs, n_states, steps=1, verbose=0,
                          plot_hist=False, ):
    is_mv = ismv(train)
    use_regimes = reg_prob_train is not None and cfg.get('regime', False)
    predictions = list()
    history, test_bundles, y_test, reg_prob_bundles = get_bundles(is_mv, steps, test, train, reg_prob_test)

    model, train_t, train_loss = nnhmm_fit(train, reg_prob_train, cfg, n_states, model_funcs, plot_hist=plot_hist,
                                           verbose=verbose, use_regimes=use_regimes)
    n_params = get_params(model, cfg)
    start_time = time.time()

    for i, (bundle, reg_prob) in enumerate(zip(test_bundles, reg_prob_bundles)):
        bundle = reshape_bundle(bundle, is_mv)
        print_progress(i, test_bundles, verbose + 1)
        yhat = nnhmm_predict(model, history, model_funcs, cfg, reg_prob, use_regimes)
        if steps > 1:
            [predictions.append(y) for y in yhat]
        elif hasattr(yhat, 'shape'):  # len(yhat) > 1
            predictions.append(yhat[0])
        else:
            predictions.append(yhat)
        history = np.vstack([history, bundle]) if is_mv else np.hstack([history, bundle])

    end_time = time.time()
    print_pred_time(start_time, test_bundles, verbose)
    pred_t = round((end_time - start_time) / len(test_bundles), 4)
    predictions = prep_forecast(predictions)
    return predictions[:len(y_test)], train_t, pred_t, n_params, train_loss


def train_model(model_cfg, model_func, data_in, verbose=1, summary=False, use_regimes=False, plot_hist=False):
    t_train, train_x, train_pp, train_reg_prob, train_reg_prob_pp = unpack_data_in(data_in)

    n_states = get_n_states(train_reg_prob)

    model, train_time, train_loss = nnhmm_fit(train_pp, train_reg_prob_pp, model_cfg, n_states, model_func,
                                              verbose=verbose, use_regimes=use_regimes, plot_hist=plot_hist)
    n_params = get_params(model, model_cfg)
    if summary:
        model.summary()
        tf.keras.utils.plot_model(
            model, to_file='model.png', show_shapes=True, show_dtype=False,
            show_layer_names=True, rankdir='TB', expand_nested=False, dpi=96
        )

    return model, train_time, train_loss, n_params


def test_model(model, input_cfg, model_cfg, model_func, in_cfg, data_in, ss, label_scale=1,
               size=(1980, 1080), plot=True, use_regimes=False):
    t_test, test_x, test_pp, test_reg_prob, test_reg_prob_pp = unpack_data_in(data_in)
    image_folder, plot_hist, plot_title, save_results, results_folder, verbose = unpack_in_cfg(in_cfg)

    model_n_steps_out = model_cfg['n_steps_out']  # same as xy_args

    predictions, pred_t = walkforward_test_model(model, model_cfg, model_func, model_n_steps_out,
                                                 test_pp, test_reg_prob_pp, use_regimes, verbose)

    forecast_reconst, test_y = reconstruct_forecast(input_cfg, model_n_steps_out,
                                                    predictions, ss, test_pp, test_x)
    metrics = forecast_accuracy(forecast_reconst, test_y)
    print(metrics)

    df = merge_forecast_df(test_y, forecast_reconst, t_test, test_reg_prob)

    if plot:
        plot_forecast(df, input_cfg, label_scale, metrics, model_cfg,
                      plot_title, size, test_reg_prob, use_regimes)
    return metrics, df, pred_t


def plot_forecast(df, input_cfg, label_scale, metrics, model_cfg, plot_title, size, test_reg_prob, use_regimes):
    name = model_cfg.get('name', 'model')
    model_title = {'n_steps_out': model_cfg[0][2]['n_steps_out']} if isinstance(model_cfg, list) else model_cfg
    rows = [0, 0] + [1 for _ in range(test_reg_prob.shape[1])] if use_regimes else None
    plotly_time_series(df, rows=rows, size=size, label_scale=label_scale,
                       title="SERIES: " + str(input_cfg) + '<br>' + name + ': ' + str(model_title) +
                             '<br>RES: ' + str(metrics), markers='lines', plot_title=plot_title)


def reconstruct_forecast(input_cfg, model_n_steps_out, predictions, ss, test_pp, test_x):
    forecast = prep_forecast(predictions)
    # forecast can be larger than test subset
    forecast = forecast[:test_pp.shape[0]]
    test_y = test_x[:, -1]
    forecast_reconst = reconstruct_pred(forecast, input_cfg, model_n_steps_out, test=test_y, ss=ss)
    return forecast_reconst, test_y


def walkforward_test_model(model, model_cfg, model_func, model_n_steps_out, test_pp, test_reg_prob_pp, use_regimes,
                           verbose):
    lookback = model_func['lookback'](model_cfg)
    x_test_pp_bundles = step_out_bundles(test_pp, model_n_steps_out, lookback)
    reg_prob_test_bundles = step_out_bundles(test_reg_prob_pp, model_n_steps_out, lookback, all=True)
    # history starts with lookback data from test
    x_history = test_pp[:lookback, :-1]
    # predictions start with preprocessed lookback test data
    predictions = list(test_pp[:lookback, -1])
    # last hmm state available
    reg_prob = test_reg_prob_pp[lookback, :]
    start_time = time.time()
    for i, (x_bundle, reg_prob_bundle) in enumerate(zip(x_test_pp_bundles, reg_prob_test_bundles)):
        print_progress(i, x_test_pp_bundles, verbose + 1)
        yhat = model_func['predict'](model, x_history, model_cfg, use_regimes, reg_prob)[:model_n_steps_out]
        [predictions.append(y) for y in yhat]
        # update next information available
        reg_prob = reg_prob_bundle[-1, :]
        x_history = np.vstack([x_history, reshape_bundle(x_bundle)])
    end_time = time.time()
    print_pred_time(start_time, x_test_pp_bundles, verbose)
    pred_t = round((end_time - start_time) / len(x_test_pp_bundles), 4)
    return predictions, pred_t


def step_out_bundles(data, n_steps_out, lookback=0, y_col=-1, all=False):
    if all:
        bundles = [data[i:i + n_steps_out, :] for i in
                   range(lookback, data.shape[0], n_steps_out)]
    else:
        bundles = [data[i:i + n_steps_out, :y_col] for i in
                   range(lookback, data.shape[0], n_steps_out)]
    return bundles


def get_n_states(reg_prob):
    if reg_prob is not None:
        n_states = reg_prob.shape[1]
    else:
        n_states = None
    return n_states


def extract_y_var(test, train, train_pp):
    if len(train.shape) > 1:
        if train_pp.shape[1] > 1:
            train_y, test_y = train[:, -1], test[:, -1]
        else:
            train_y, test_y = train, test
    else:
        train_y, test_y = train, test
    return test_y, train_y


def unpack_data_in(data_in):
    if len(data_in) == 5:
        t_train, train_x, train_pp, train_reg_prob, train_reg_prob_pp = data_in
    elif len(data_in) == 3:
        t_train, train_x, train_pp = data_in
        train_reg_prob, train_reg_prob_pp = None, None
    else:
        raise Exception('data_in must have len = 3 or 5')
    return t_train, train_x, train_pp, train_reg_prob, train_reg_prob_pp


def get_bundles(is_mv, steps, test, train, reg_prob_test):
    train = np.array(train) if is_mv else train.reshape(train.shape[0], 1)
    test = np.array(test) if is_mv else test.reshape(test.shape[0], 1)
    # remove last column
    history = np.array(train[:, :-1]) if is_mv else train.ravel()
    X_test, y_test = (test[:, :-1], test[:, -1]) if is_mv else (test, test.ravel())
    # step over each time-step in the test set
    test_bundles = [X_test[i:i + steps] for i in range(0, X_test.shape[0], steps)] if steps > 1 else X_test
    if reg_prob_test is None:
        reg_prob_bundle = np.full(len(test_bundles), 0)
    else:
        reg_prob_bundle = reg_prob_test[::steps]
    return history, test_bundles, y_test, reg_prob_bundle
