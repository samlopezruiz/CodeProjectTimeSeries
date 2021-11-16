import datetime as dte
import multiprocessing
import os
import time
from contextlib import contextmanager
from multiprocessing import cpu_count

import numpy as np
import tensorflow as tf
from joblib import Parallel, delayed
from tensorflow import keras
from tqdm import tqdm

from algorithms.nnhmm.func import nnhmm_fit, nnhmm_predict
from algorithms.tft2.libs import utils as utils
from algorithms.tft2.libs.hyperparam_opt import HyperparamOptManager
from algorithms.tft2.libs.tft_model import TemporalFusionTransformer
from algorithms.tft2.utils.data import extract_numerical_data
# from timeseries.experiments.market.train_tft_fixed_params import formatter
from algorithms.ts.models import TFTModel, LSTMModel, DCNNModel
from timeseries.experiments.market.expt_settings.configs import ExperimentConfig
from timeseries.experiments.market.moo.utils.model import get_last_layer_weights, params_conversion_weights, \
    reconstruct_weights
from timeseries.experiments.market.utils.console import print_progress, print_pred_time, print_progress_loop
from timeseries.experiments.market.utils.dataprep import to_np
from timeseries.experiments.market.utils.models import get_params
from timeseries.experiments.market.utils.preprocessing import reconstruct_pred, prep_forecast
from timeseries.experiments.utils.forecast import merge_forecast_df
from timeseries.experiments.utils.metrics import forecast_accuracy
from timeseries.plotly.plot import plotly_time_series
from timeseries.preprocessing.func import ismv


@contextmanager
def poolcontext(*args, **kwargs):
    pool = multiprocessing.pool.ThreadPool(*args, **kwargs)
    yield pool
    pool.terminate()


'''
EXAMPLE CODE TO USE MULTIPROCESSING
partial_model_pred = partial(model_pred, model, model_cfg, model_func, model_n_steps_out, ss,
                             test_x_pp, training_cfg, unscaled_test_y, use_regimes, verbose)
with poolcontext(processes=cpu_count()) as pool:
    result = pool.map(partial_model_pred, range(len(test_x_pp)))
'''


def reshape_bundle(bundle, is_mv=True):
    if is_mv and len(bundle.shape) == 1:
        return bundle.reshape(1, bundle.shape[0])
    if not is_mv and len(bundle.shape) > 1:
        return bundle.ravel()
    return bundle


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


def train_model(model_cfg, model_func, train_data, test_data=None, summary=False,
                plot_hist=False, model=None, callbacks=False):
    # t_train, train_x, train_pp, train_reg_prob, train_reg_prob_pp = unpack_data_in(data_in)
    # train_x, train_prob = train_data
    verbose, use_regimes = model_cfg['verbose'], model_cfg['use_regimes']
    n_states = get_n_states(train_data[2])
    t0 = time.time()
    model, train_time, train_loss = nnhmm_fit(train_data, model_cfg, n_states, model_func, model=model,
                                              test_data=test_data, verbose=verbose, use_regimes=use_regimes,
                                              plot_hist=plot_hist, callbacks=callbacks)
    print('Train Time: {}s'.format(round(time.time() - t0, 4)))
    # n_params = get_params(model, model_cfg)
    if summary:
        model.summary()
        tf.keras.utils.plot_model(
            model, to_file='model.png', show_shapes=True, show_dtype=False,
            show_layer_names=True, rankdir='TB', expand_nested=False, dpi=96
        )

    return model, train_time, train_loss


def test_model(model, model_cfg, training_cfg, model_func, test_data, ss, parallel=True):
    """
    test data consists on a tuple of two lists
    the first element is the preprocessed X test bundles
    the second element is the unscaled test y variable
    """
    test_x_pp, unscaled_test_y = test_data
    model_n_steps_out = model_cfg['n_steps_out']
    use_regimes, verbose = model_cfg['use_regimes'], model_cfg['verbose']
    assert len(test_x_pp) == len(unscaled_test_y)
    forecast_dfs, metrics, pred_times = [], [], []
    t0 = time.time()
    if parallel:
        executor = Parallel(n_jobs=cpu_count(), backend='multiprocessing')
        tasks = (
            delayed(model_pred)(model, model_cfg, model_func, model_n_steps_out, ss,
                                test_x_pp, training_cfg, unscaled_test_y, use_regimes, verbose, i)
            for i in tqdm(range(len(test_x_pp)))
        )
        result = executor(tasks)
        for df, metric, pred_t in result:
            forecast_dfs.append(df)
            metrics.append(metric)
            pred_times.append(pred_t)
    else:
        for i in range(len(test_x_pp)):
            print_progress_loop(i, len(test_x_pp), process_text='test bundles')
            df, metric, pred_t = model_pred(model, model_cfg, model_func, model_n_steps_out, ss,
                                            test_x_pp, training_cfg, unscaled_test_y, use_regimes, verbose, i)

            forecast_dfs.append(df)
            metrics.append(metric)
            pred_times.append(pred_t)
    print('Pred Time: {}s'.format(round(time.time() - t0, 4)))
    return forecast_dfs, metrics, pred_times


def model_pred(model, model_cfg, model_func, model_n_steps_out, ss, test_x_pp, training_cfg, unscaled_test_y,
               use_regimes, verbose, i):
    # model = tf.saved_model.load(model_path)
    test_pp, test_reg_prob_pp, unscaled_y = unpack_test_vars(i, test_x_pp, unscaled_test_y)
    assert test_pp.index.equals(unscaled_y.index)
    test_ix = test_pp.index
    test_pp, test_reg_prob_pp, unscaled_y = to_np([test_pp, test_reg_prob_pp, unscaled_y])
    assert len(test_pp) == len(unscaled_y) and \
           (True if test_reg_prob_pp is None else len(unscaled_y) == len(test_reg_prob_pp))
    scaled_pred_y, pred_t = walkforward_test_model(model, model_cfg, model_func, model_n_steps_out,
                                                   test_pp, test_reg_prob_pp, use_regimes, verbose - 1)
    forecast_reconst = reconstruct_pred(scaled_pred_y, model_n_steps_out, unscaled_y, ss=ss,
                                        preprocess=training_cfg.get('preprocess', True))
    metric = forecast_accuracy(forecast_reconst, unscaled_y)
    df = merge_forecast_df(unscaled_y, forecast_reconst, reg_prob=test_reg_prob_pp,
                           ix=test_ix, use_regimes=use_regimes)
    if training_cfg['append_train_to_test']:
        # first 'predictions' are only train data
        lookback = model_func['lookback'](model_cfg)
        # one train step included
        df = df.iloc[lookback - 1:, :]
    return df, metric, pred_t


def unpack_test_vars(i, test_x_pp, unscaled_test_y):
    test_pp, test_reg_prob_pp = test_x_pp[i][0], test_x_pp[i][1]
    unscaled_y = unscaled_test_y[i][0]
    return test_pp, test_reg_prob_pp, unscaled_y


def plot_forecast(df, model_cfg=None, n_states=0, metrics=None, features=None, use_regimes=False, size=(1980, 1080),
                  plot_title=True, label_scale=1, markers='lines', adjust_height=(False, 0.6), color_col=None,
                  save=False, file_path=None):
    name = model_cfg.get('name', 'model')
    if model_cfg is not None:
        model_str = {'n_steps_out': model_cfg[0][2]['n_steps_out']} if isinstance(model_cfg, list) else model_cfg
        model_title = name + ': ' + str(model_str)
    else:
        model_title = ''

    res_title = '<br>RES: ' + str(metrics) if metrics is not None else ''
    if features is None:
        features = list(df.columns)
    if n_states > 0 and use_regimes:
        for i in range(n_states):
            if 'regime ' + str(i) in features:
                features.remove('regime ' + str(i))
        for i in range(n_states):
            features.append('regime ' + str(i))

    rows = [0] * (len(features) - n_states) + [1 for _ in range(n_states)] if use_regimes else None
    plotly_time_series(df, features=features, rows=rows, size=size, label_scale=label_scale,
                       adjust_height=adjust_height, save=save, file_path=file_path,
                       color_col=color_col, title=model_title + res_title, markers=markers, plot_title=plot_title)


def reconstruct_forecast(model_n_steps_out, predictions, ss, test_y):
    forecast = prep_forecast(predictions)
    # forecast can be larger than test subset
    # forecast = forecast[:test_pp.shape[0]]
    # test_y = test_x[:, -1]
    forecast_reconst = reconstruct_pred(forecast, model_n_steps_out, test=test_y, ss=ss)
    return forecast_reconst, test_y


def walkforward_test_model(model, model_cfg, model_func, model_n_steps_out, test_pp, test_reg_prob_pp, use_regimes,
                           verbose):
    lookback = model_func['lookback'](model_cfg)
    x_test_pp_bundles = step_out_bundles(test_pp, model_n_steps_out, lookback)
    if use_regimes:
        reg_prob_test_bundles = step_out_bundles(test_reg_prob_pp, model_n_steps_out, lookback, all=True)
    else:
        reg_prob_test_bundles = [None for _ in range(len(test_pp))]
    # history starts with lookback data from test
    x_history = test_pp[:lookback, :-1]
    # predictions start with preprocessed lookback test data
    predictions = list(test_pp[:lookback, -1])
    # last hmm state available
    reg_prob = None if test_reg_prob_pp is None else test_reg_prob_pp[lookback, :]
    start_time = time.time()

    for i, (x_bundle, reg_prob_bundle) in enumerate(zip(x_test_pp_bundles, reg_prob_test_bundles)):
        print_progress(i, x_test_pp_bundles, verbose + 1)
        yhat = model_func['predict'](model, x_history, model_cfg, use_regimes, reg_prob)[:model_n_steps_out]
        [predictions.append(y) for y in yhat]
        # update next information available
        reg_prob = reg_prob_bundle[-1, :] if use_regimes else None
        x_history = np.vstack([x_history, reshape_bundle(x_bundle)])

    end_time = time.time()
    print_pred_time(start_time, x_test_pp_bundles, verbose)
    pred_t = round((end_time - start_time) / len(x_test_pp_bundles), 4)
    predictions = np.array(predictions)[:test_pp.shape[0]]
    return predictions, pred_t


def step_out_bundles(data, n_steps_out, lookback=0, y_col=-1, all=False):
    if data is not None:
        if all:
            bundles = [data[i:i + n_steps_out, :] for i in
                       range(lookback, data.shape[0], n_steps_out)]
        else:
            bundles = [data[i:i + n_steps_out, :y_col] for i in
                       range(lookback, data.shape[0], n_steps_out)]
        return bundles
    else:
        return None


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

def get_model_data_config(experiment_cfg, model_cfg, fixed_cfg):
    config = ExperimentConfig(experiment_cfg['formatter'], experiment_cfg)
    formatter = config.make_data_formatter()
    formatter.update_model_params(model_cfg)
    formatter.update_fixed_params(fixed_cfg)
    model_folder = os.path.join(config.model_folder, experiment_cfg['experiment_name'])

    return config, formatter, model_folder

def train_test_model(use_gpu,
                     architecture,
                     prefetch_data,
                     model_folder,
                     data_config,
                     data_formatter,
                     use_testing_mode=False,
                     predict_eval=True,
                     tb_callback=True,
                     use_best_params=False,
                     indicators_use_time_subset=False,
                     split_data=None):
    """Trains tft based on defined model params.
  Args:
      :param split_data:
      :param indicators_use_time_subset:
      :param architecture:
      :param use_best_params:
      :param tb_callback:
      :param predict_eval:
      :param use_gpu: Whether to run tensorflow with GPU operations
      :param use_testing_mode: Uses a smaller models and data sizes for testing purposes
      :param data_formatter: Dataset-specific data fromatter
      :param data_config: Data input file configurations
      :param model_folder: Folder path where models are serialized
      :param prefetch_data: Prefetch data for training
  """

    Model = get_model(architecture)

    print("Loading & splitting data...")
    if split_data is None:
        train, valid, test = data_formatter.split_data(data_config,
                                                       indicators_use_time_subset=indicators_use_time_subset)
    else:
        train, valid, test = split_data

    train_samples, valid_samples = data_formatter.get_num_samples_for_calibration()

    print('shape: {}'.format(train.shape))

    # Sets up default params
    fixed_params = data_formatter.get_experiment_params()
    params = data_formatter.get_default_model_params()
    params["model_folder"] = model_folder

    # Parameter overrides for testing only! Small sizes used to speed up script.
    if use_testing_mode:
        fixed_params["num_epochs"] = 1
        params["hidden_layer_size"] = 5
        train_samples, valid_samples = 100, 10

    # Sets up hyperparam manager
    print("*** Loading hyperparm manager ***")
    opt_manager = HyperparamOptManager({k: [params[k]] for k in params}, fixed_params, model_folder)

    # Training -- one iteration only
    print("*** Running calibration ***")
    print("Params Selected:")
    for k in params:
        print("{}: {}".format(k, params[k]))

    best_loss = np.Inf

    with tf.device('/device:GPU:0' if use_gpu else "/cpu:0"):

        params = opt_manager.get_next_parameters()
        # model = TemporalFusionTransformer(params, use_cudnn=use_gpu, tb_callback=tb_callback)
        model = Model(params, tb_callback=tb_callback)

        if not model.training_data_cached():
            model.cache_batched_data(train, "train", num_samples=train_samples)
            model.cache_batched_data(valid, "valid", num_samples=valid_samples)

        model.fit(prefetch_data)

        val_loss = model.evaluate()

        if val_loss < best_loss:
            opt_manager.update_score(params, val_loss, model)
            best_loss = val_loss

    print("Training completed @ {}".format(dte.datetime.now()))
    print("Validation loss = {}".format(val_loss))

    if predict_eval:
        if use_best_params:
            print("*** Running tests ***")
            params = opt_manager.get_best_params()
            model = Model(params, tb_callback=tb_callback)
            # model = TemporalFusionTransformer(params, use_cudnn=use_gpu)

            model.load(opt_manager.hyperparam_folder, use_keras_loadings=True)

            print("Computing best validation loss")
            val_loss = model.evaluate(valid)
            print("Best Validation loss = {}".format(val_loss))

        print("Training completed @ {}".format(dte.datetime.now()))
        print("Best validation loss = {}".format(val_loss))

        return predict_from_model(params, data_formatter, model, test)
    else:
        return val_loss


#
def predict_from_model(best_params, data_formatter, model, test):
    print("Computing test loss")
    output_map = model.predict(test, return_targets=True)
    unscaled_output_map = {}
    for k, df in output_map.items():
        unscaled_output_map[k] = data_formatter.format_predictions(df)

    losses = {}
    targets = unscaled_output_map['targets']
    for q in model.quantiles:
        key = 'p{}'.format(int(q * 100))
        losses[key + '_loss'] = utils.numpy_normalised_quantile_loss(
            extract_numerical_data(targets), extract_numerical_data(unscaled_output_map[key]), q)

    print("Params:")
    for k in best_params:
        print(k, " = ", best_params[k])
    print("\nNormalised Quantile Losses for Test Data: {}".format(
        [p_loss.mean() for k, p_loss in losses.items()]))

    results = {'quantiles': model.quantiles,
               'forecasts': unscaled_output_map,
               'losses': losses,
               'learning_rate': model.learning_rate,
               'fit_history': model.fit_history,
               'target': data_formatter.test_true_y.columns[0] if data_formatter.test_true_y is not None else None,
               'fixed_params': data_formatter.get_fixed_params(),
               'model_params': data_formatter.get_default_model_params()}

    return results


def load_predict_model(use_gpu,
                       architecture,
                       model_folder,
                       data_config,
                       data_formatter,
                       use_all_data=False,
                       last_layer_weights=None,
                       exclude_p50=True):
    """Trains tft based on defined model params.
  Args:
      :param use_all_data:
      :param use_gpu: Whether to run tensorflow with GPU operations
      :param use_testing_mode: Uses a smaller models and data sizes for testing purposes
      :param data_formatter: Dataset-specific data fromatter
      :param data_config: Data input file configurations
      :param model_folder: Folder path where models are serialized

  """
    Model = get_model(architecture)

    print("Loading & splitting data...")
    if use_all_data:
        test = data_formatter.process_data(data_config)
    else:
        train, valid, test = data_formatter.split_data(data_config)

    # Sets up default params
    fixed_params = data_formatter.get_experiment_params()
    params = data_formatter.get_default_model_params()
    params["model_folder"] = model_folder

    # Sets up hyperparam manager
    print("*** Loading hyperparm manager ***")
    opt_manager = HyperparamOptManager({k: [params[k]] for k in params}, fixed_params, model_folder)

    print("*** Running tests ***")
    params = opt_manager.get_next_parameters()

    with tf.device('/device:GPU:0' if use_gpu else "/cpu:0"):
        # model = TemporalFusionTransformer(params, use_cudnn=use_gpu)
        model = Model(params)
        model.load(opt_manager.hyperparam_folder, use_keras_loadings=True)

        # manually set last layer weights (used in multi-objective optimization)
        if last_layer_weights is not None:
            weights, last_layer = get_last_layer_weights(model)

            # get original p50 weights
            if exclude_p50:
                p50_w = weights[0][:, 1]
                p50_b = weights[1][1]
                weights_woP50 = [weights[0][:, [0, 2]], weights[1][[0, 2]]]

                # get conversion parameters from weights wo p50
                ind, w_params_woP50 = params_conversion_weights(weights_woP50)
                new_weights = reconstruct_weights(last_layer_weights, w_params_woP50)

                new_weights[0] = np.vstack([new_weights[0][:, 0],
                                            p50_w,
                                            new_weights[0][:, 1]]).T
                new_weights[1] = np.array([new_weights[1][0],
                                           p50_b,
                                           new_weights[1][1]])
            else:
                ind, w_params_wP50 = params_conversion_weights(weights)
                new_weights = reconstruct_weights(last_layer_weights, w_params_wP50)

            last_layer.set_weights(new_weights)

        return predict_from_model(params, data_formatter, model, test), test


def get_model(architecture):
    architecture_options = ['TFTModel', 'LSTMModel', 'DCNNModel']
    if architecture not in architecture_options:
        raise Exception('{} not a valid option. \nOptions: {}'.format(architecture, architecture_options))

    if architecture == 'TFTModel':
        model = TFTModel
    elif architecture == 'LSTMModel':
        model = LSTMModel
    elif architecture == 'DCNNModel':
        model = DCNNModel

    return model
