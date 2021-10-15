from timeseries.experiments.market.multivariate.architectures.func import arg_in_out, \
    lookback_in_steps, n_features_2
from timeseries.experiments.market.utils.dataprep import feature_multi_step_xy_from_mv
from numpy import array
from tensorflow import keras


def simple_layers(input_, cfg):
    output = keras.layers.Dense(1)(input_)
    return output


def simple_input_shape(cfg, n_features):
    input_shape = (cfg['n_steps_in'], n_features)
    return input_shape


def simple_prep_data(X, cfg):
    n_steps_in = cfg['n_steps_in']
    n_features = X.shape[1]
    x_input = array(X[-n_steps_in:]).reshape(1, n_steps_in, n_features)
    return x_input


def simple_predict(model, history, cfg, use_regimes, reg_prob=None):
    # prepare data
    x_input = simple_prep_data(history, cfg)
    yhat = [x_input[-1][-1, -1]]
    return yhat


def simple_func():
    return {
        'name': 'D-CNN',
        'xy_from_train': feature_multi_step_xy_from_mv,
        'xy_args': arg_in_out,
        'intermediate_layers': simple_layers,
        'input_shape': simple_input_shape,
        'n_features': n_features_2,
        'prep_data': simple_prep_data,
        'predict': simple_predict,
        'lookback': lookback_in_steps,
    }
