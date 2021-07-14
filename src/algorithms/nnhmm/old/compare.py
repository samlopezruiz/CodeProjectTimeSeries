import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from algorithms.nnhmm.func import nnhmm_fit
from timeseries.models.market.multivariate.architectures.cnnlstm import cnnlstm_func


import tensorflow as tf
import numpy as np
from timeseries.models.lorenz.functions.functions import get_bundles, ismv, reshape_bundle, \
    prep_forecast
from timeseries.models.utils.forecast import merge_forecast_df
from timeseries.models.utils.metrics import forecast_accuracy
from timeseries.plotly.plot import plotly_time_series
from timeseries.models.lorenz.functions.preprocessing import preprocess, reconstruct
from timeseries.data.lorenz.lorenz import lorenz_wrapper
from timeseries.models.lorenz.multivariate.multistep.cnnlstm.func import cnnlstm_get_multi_step_mv_funcs
# %%

if __name__ == '__main__':
    # %% GENERAL INPUTS
    in_cfg = {'steps': 6, 'save_results': False, 'verbose': 1, 'plot_title': True, 'plot_hist': False,
              'image_folder': 'images', 'results_folder': 'results'}

    # MODEL AND TIME SERIES INPUTS
    name = "CNN-LSTM"
    input_cfg = {"variate": "multi", "granularity": 5, "noise": False, 'preprocess': True,
                 'trend': False, 'detrend': 'ln_return'}
    model_cfg = {"n_steps_out": 6, "n_steps_in": 8, "n_seq": 2, "n_kernel": 3,
                 "n_filters": 64, "n_nodes": 128, "n_batch": 32, "n_epochs": 25} #, 'n_ensembles': 3
    func_cfg = cnnlstm_get_multi_step_mv_funcs()

    # %% DATA
    lorenz_df, train, test, t_train, t_test = lorenz_wrapper(input_cfg)
    train_pp, test_pp, ss = preprocess(input_cfg, train, test)
    data_in = (train_pp, test_pp, train, test, t_train, t_test)
    train_y, test_y = train[:, -1], test[:, -1]

    # forecast, _, _, _, _ = walk_forward_step_forecast(train_pp, test_pp, model_cfg, func_cfg[0], func_cfg[1],
    #                                                   verbose=1, steps=in_cfg['steps'])
    #%%
    steps = 6
    model_forecast, model_fit = func_cfg[0], func_cfg[1]
    is_mv = ismv(train)
    predictions = list()
    history, test_bundles, y_test = get_bundles(is_mv, steps, test_pp, train_pp)

    model_func = cnnlstm_func()
    model, train_t, train_loss = nnhmm_fit(train_pp, None, model_cfg, 3, model_func,
                                           verbose=1, use_regimes=False)
    # model, train_t, train_loss = model_fit(train_pp, model_cfg, verbose=1)

    model.summary()
    tf.keras.utils.plot_model(
        model, to_file='model_orig.png', show_shapes=True, show_dtype=False,
        show_layer_names=True, rankdir='TB', expand_nested=False, dpi=96
    )

    for i, bundle in enumerate(test_bundles):
        bundle = reshape_bundle(bundle, is_mv)
        yhat = model_forecast(model, steps=steps, history=history, cfg=model_cfg)
        [predictions.append(y) for y in yhat]
        history = np.vstack([history, bundle]) if is_mv else np.hstack([history, bundle])

    predictions = prep_forecast(predictions)
    forecast = predictions

    #%%
    import pandas as pd
    df = pd.DataFrame()
    df['true'] = test_pp[:, -1]
    df['pred'] = forecast
    plotly_time_series(df)

    #%%
    forecast_reconst = reconstruct(forecast, train, test, input_cfg, in_cfg['steps'], ss=ss)
    metrics = forecast_accuracy(forecast_reconst, test_y)

    # df = multi_step_forecast_df(train_y, test_y, forecast_reconst, t_train, t_test, train_prev_steps=200)
    df = merge_forecast_df(test_y, forecast_reconst, t_test, None, None)
    plotly_time_series(df)
    print(metrics)