import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from timeseries.models.utils.forecast import merge_forecast_df
from timeseries.models.utils.metrics import forecast_accuracy
from timeseries.plotly.plot import plotly_time_series

from timeseries.models.market.utils.harness import reshape_bundle, prep_forecast
import tensorflow as tf
from algorithms.nnhmm.func import nnhmm_fit
from timeseries.data.lorenz.lorenz import regime_multivariate_lorenz
from timeseries.models.market.multivariate.architectures.cnnlstm import cnnlstm_func
from timeseries.models.market.utils.preprocessing import preprocess, reconstruct_pred
import numpy as np


def step_out_bundles(data, n_steps_out, lookback=0, y_col=-1, all=False):
    if all:
        bundles = [data[i:i + n_steps_out, :] for i in
                   range(lookback, data.shape[0], n_steps_out)]
    else:
        bundles = [data[i:i + n_steps_out, :y_col] for i in
                   range(lookback, data.shape[0], n_steps_out)]
    return bundles


if __name__ == '__main__':
    # %% DATA
    input_cfg = {"variate": "multi", "granularity": 5, "noise": True, 'preprocess': True,
                 'trend': False, 'detrend': 'ln_return', 'sigma': 0.5}

    use_regimes = True

    # if use_regimes:
    lorenz_df, train, test, t_train, t_test, hidden_states = regime_multivariate_lorenz(input_cfg)
    plotly_time_series(lorenz_df, features=['x', 'y', 'z'], rows=list(range(3)), markers='lines')
    train_x, train_reg_prob = train
    test_x, test_reg_prob = test
    # else:
    #     lorenz_df, train_x, test_x, t_train, t_test = lorenz_wrapper(input_cfg)
    #     train_reg_prob, test_reg_prob = None, None

    train_pp, test_pp, train_reg_prob_pp, test_reg_prob_pp, ss = preprocess(input_cfg, train_x, test_x,
                                                                            train_reg_prob, test_reg_prob)

    train_data = (t_train, train_x, train_pp, train_reg_prob, train_reg_prob_pp)
    test_data = (t_test, test_x, test_pp, test_reg_prob, test_reg_prob_pp)

    # %%
    model_cfg = {'name': 'CNNLSTM-REG', "n_steps_out": 6, "n_steps_in": 8, "n_seq": 2, "n_kernel": 3,
                 "n_filters": 64, "n_nodes": 128, "n_batch": 32, "n_epochs": 25}
    n_states = 2
    model_func = cnnlstm_func()

    # %% TRAIN


    model, train_t, train_loss = nnhmm_fit(train_pp, train_reg_prob_pp, model_cfg, n_states, model_func,
                                           verbose=1, use_regimes=use_regimes)

    model.summary()
    tf.keras.utils.plot_model(
        model, to_file='model.png', show_shapes=True, show_dtype=False,
        show_layer_names=True, rankdir='TB', expand_nested=False, dpi=96
    )
    # %% TEST
    model_n_steps_out = 6  # same as xy_args
    pred_steps = 6

    # %%
    lookback = model_func['lookback'](model_cfg)
    x_test_pp_bundles = step_out_bundles(test_pp, model_n_steps_out, lookback)
    reg_prob_test_bundles = step_out_bundles(test_reg_prob_pp, model_n_steps_out, lookback, all=True)

    # history starts with lookback data from test
    x_history = test_pp[:lookback, :-1]
    # predictions start with preprocessed lookback test data
    predictions = list(test_pp[:lookback, -1])
    # last hmm state available
    reg_prob = test_reg_prob_pp[lookback, :]

    for i, (x_bundle, reg_prob_bundle) in enumerate(zip(x_test_pp_bundles, reg_prob_test_bundles)):
        yhat = model_func['predict'](model, x_history, model_cfg, use_regimes, reg_prob)[:model_n_steps_out]
        [predictions.append(y) for y in yhat]
        # update next information available
        reg_prob = reg_prob_bundle[-1, :]
        x_history = np.vstack([x_history, reshape_bundle(x_bundle)])

    forecast = prep_forecast(predictions)
    # forecast can be larger than test subset
    forecast = forecast[:test_pp.shape[0]]

    # %%
    test_y = test_x[:, -1]
    forecast_reconst = reconstruct_pred(forecast, input_cfg, model_n_steps_out, test=test_y, ss=ss)
    metrics = forecast_accuracy(forecast_reconst, test_y)

    print(metrics)

    df = merge_forecast_df(test_y, forecast_reconst, t_test, test_reg_prob)
    plotly_time_series(df, rows=[0, 0] + [1 for _ in range(test_reg_prob.shape[1])])

