import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from algorithms.wavenet.func import dcnn_1st_layer, dcnn_layer
import math
from timeseries.experiments.utils.tf import np_to_tf
import numpy as np
from timeseries.experiments.market.multivariate.architectures.func import n_features_3, lookback_seq_steps, arg_in_out, \
    lookback_in_steps, n_features_2
from timeseries.experiments.market.utils.dataprep import feature_multi_step_xy_from_mv
from numpy import array
from tensorflow import keras
from tensorflow.keras import initializers


def dcnn_layers(input_, cfg):
    n_steps_in, n_steps_out, n_filters = cfg['n_steps_in'], cfg['n_steps_out'], cfg['n_filters']
    n_layers, reg, n_kernel = cfg['n_layers'], cfg['reg'], cfg['n_kernel']
    # model_id = cfg.get('model_id', 0)
    assert n_steps_in > 2 ** (n_layers - 1) * n_kernel
    stddev = math.sqrt(2 / (n_kernel * n_filters))
    # ARCHITECTURE
    x = dcnn_1st_layer(n_filters, n_kernel, 1, None, reg=reg)(input_)
    for layer in range(1, n_layers):
        x = dcnn_layer(n_filters, n_kernel, 2 ** layer, None, reg=reg)(x)
    out_conv = keras.layers.Conv1D(n_filters, 1,
                                   padding='same', use_bias=True,
                                   activation='relu')(x)
    preoutput = keras.layers.Flatten()(out_conv)
    preoutput = keras.layers.Dense(n_steps_out * 2, kernel_regularizer=reg,
                                   kernel_initializer=initializers.RandomNormal(stddev=stddev))(preoutput)
    output = keras.layers.Dense(n_steps_out, kernel_regularizer=reg,
                                kernel_initializer=initializers.RandomNormal(stddev=stddev))(preoutput)
    return output


def dcnn_input_shape(cfg, n_features):
    input_shape = (cfg['n_steps_in'], n_features)
    return input_shape


def dcnn_prep_data(X, cfg):
    n_steps_in = cfg['n_steps_in']
    n_features = X.shape[1]
    x_input = array(X[-n_steps_in:]).reshape(1, n_steps_in, n_features)
    return x_input


def dcnn_predict(model, history, cfg, use_regimes, reg_prob=None):
    # prepare data
    x_input = dcnn_prep_data(history, cfg)
    if use_regimes:
        model_input = [np_to_tf(x_input), np_to_tf(np.expand_dims(reg_prob, axis=0))]
    else:
        model_input = np_to_tf(x_input)
    # forecast
    if isinstance(model, list):
        yhat = [m.predict(model_input, verbose=0)[0] for m in model]
        yhat = np.array(yhat).mean(axis=0)
    else:
        if hasattr(model, 'predict'):
            yhat = model.predict(model_input, verbose=0)[0]
        else:
            yhat = model(model_input)[0]
    return yhat


def dcnn_func():
    return {
        'name': 'D-CNN',
        'xy_from_train': feature_multi_step_xy_from_mv,
        'xy_args': arg_in_out,
        'intermediate_layers': dcnn_layers,
        'input_shape': dcnn_input_shape,
        'n_features': n_features_2,
        'prep_data': dcnn_prep_data,
        'predict': dcnn_predict,
        'lookback': lookback_in_steps,
    }
