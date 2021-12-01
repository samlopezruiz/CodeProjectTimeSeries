# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from timeseries.experiments.utils.tf import np_to_tf
import numpy as np
from timeseries.experiments.market.multivariate.architectures.func import n_features_3, lookback_seq_steps
from timeseries.experiments.market.utils.dataprep import step_feature_multi_step_xy_from_mv
from tensorflow.keras.layers import LSTM, Dense, TimeDistributed, Conv1D, MaxPooling1D, Flatten
from numpy import array


def cnnlstm_xy_args(cfg, reg_prob):
    n_seq, n_steps, n_steps_out = cfg['n_seq'], cfg['n_steps_in'], cfg['n_steps_out']
    n_input = n_seq * n_steps
    return n_input, n_steps_out, n_seq, reg_prob


def cnnlstm_layers(input_, cfg):
    n_steps_out, n_filters = cfg['n_steps_out'], cfg['n_filters']
    n_kernel, n_nodes = cfg['n_kernel'], cfg['n_nodes']
    conv_1 = TimeDistributed(Conv1D(n_filters, n_kernel, activation='relu'))(input_)
    conv_2 = TimeDistributed(Conv1D(n_filters, n_kernel, activation='relu'))(conv_1)
    pool = TimeDistributed(MaxPooling1D())(conv_2)
    flat = TimeDistributed(Flatten())(pool)
    lstm = LSTM(n_nodes,
                # activation='relu',
                activation='tanh',
                recurrent_activation='sigmoid',
                recurrent_dropout=0,
                unroll=False,
                use_bias=True
                )(flat)
    dense = Dense(n_nodes, activation='relu')(lstm)
    out = Dense(n_steps_out)(dense)
    return out


def cnnlstm_input_shape(cfg, n_features):
    n_seq, n_steps_in = cfg['n_seq'], cfg['n_steps_in']
    input_shape = (n_seq, n_steps_in, n_features)
    return input_shape


def cnnlstm_prep_data(X, cfg):
    n_seq, n_steps_in = cfg['n_seq'], cfg['n_steps_in']
    n_features = X.shape[1]
    n_input = n_seq * n_steps_in
    x_input = array(X[-n_input:]).reshape((1, n_seq, n_steps_in, n_features))

    return x_input


def cnnlstm_func():
    return {
        'name': 'CNN-LSTM',
        'xy_from_train': step_feature_multi_step_xy_from_mv,
        'xy_args': cnnlstm_xy_args,
        'intermediate_layers': cnnlstm_layers,
        'input_shape': cnnlstm_input_shape,
        'n_features': n_features_3,
        'prep_data': cnnlstm_prep_data,
        'predict': cnnlstm_predict,
        'lookback': lookback_seq_steps,
    }


def cnnlstm_predict(model, history, cfg, use_regimes, reg_prob=None):
    # unpack config
    n_seq, n_steps = cfg['n_seq'], cfg['n_steps_in']
    n_features = history.shape[1]
    n_input = n_seq * n_steps
    # prepare data
    x_input = array(history[-n_input:]).reshape((1, n_seq, n_steps, n_features))
    if use_regimes:
        model_input = [np_to_tf(x_input), np_to_tf(np.expand_dims(reg_prob, axis=0))]
        # model_input = [x_input, reg_prob]
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
