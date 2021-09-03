import time

import numpy as np
from keras.layers import TimeDistributed
from matplotlib import pyplot as plt
from tensorflow import keras

from timeseries.plotly.plot import plot_history

import tensorflow as tf
from tensorflow.python.keras.saving import saving_utils
from tensorflow.python.keras.layers import deserialize, serialize


def unpack(model, training_config, weights):
    restored_model = deserialize(model)
    if training_config is not None:
        restored_model.compile(**saving_utils.compile_args_from_training_config(
            training_config))
    restored_model.set_weights(weights)
    return restored_model


def make_keras_picklable():
    def __reduce__(self):
        model_metadata = saving_utils.model_metadata(self)
        training_config = model_metadata.get("training_config", None)
        model = serialize(self)
        weights = self.get_weights()
        return unpack, (model, training_config, weights)

    cls = tf.keras.models.Model
    cls.__reduce__ = __reduce__


# Ensures that tf.keras models are pickable
make_keras_picklable()


def inter_layers(i, probs_, input_, cfg, intermediate_layers):
    weights_sliced = keras.layers.Lambda(lambda x: x[:, i])(probs_)
    out = intermediate_layers(input_, cfg)
    mul = keras.layers.Multiply()([weights_sliced, out])
    return mul


def build_nnhmm_model(cfg, n_states, n_features, model_funcs, use_regimes=False):
    intermediate_layers, input_fn = model_funcs['intermediate_layers'], model_funcs['input_shape']
    input_shape = input_fn(cfg, n_features)
    input_ = keras.layers.Input(shape=input_shape, name='input')
    # input_ = TimeDistributed(input_)
    if use_regimes:
        probs_ = keras.layers.Input(shape=n_states, name='regime_prob')
        mul = []
        for i in range(n_states):
            mul.append(inter_layers(i, probs_, input_, cfg, intermediate_layers))

        output = keras.layers.Add()(mul)
        model = keras.Model(inputs=[input_, probs_], outputs=[output])
    else:
        out = intermediate_layers(input_, cfg)
        model = keras.Model(inputs=[input_], outputs=[out])

    loss, optimizer = cfg.get('loss', 'mse'), cfg.get('optimizer', 'adam')
    model.compile(loss=loss, optimizer=optimizer)
    return model


def nnhmm_fit(train, cfg, n_states, model_func, model=None, plot_hist=False, verbose=0, use_regimes=False):
    # X, y, X_reg = model_func['xy_from_train'](train, *model_func['xy_args'](cfg, reg_prob))
    X, y, X_reg = train
    n_features = model_func['n_features'](X)
    start_time = time.time()
    if model is None:
        model = build_nnhmm_model(cfg, n_states, n_features, model_func, use_regimes=use_regimes)
    model_input = [X, X_reg] if use_regimes else X
    history = model.fit(model_input, y, epochs=cfg['n_epochs'], batch_size=cfg['n_batch'], verbose=verbose)

    train_time = round((time.time() - start_time), 2)
    if plot_hist:
        plot_history(history) #, title=model_func['name'] + str(cfg), plot_title=True)
    return model, train_time, history.history['loss'][-1]


def nnhmm_predict(model, history, model_funcs, cfg, reg_prob=None, use_regimes=False):
    x_input = model_funcs['prep_data'](history, cfg)
    x_input = [x_input, reg_prob] if use_regimes else x_input
    if isinstance(model, list):
        yhat = [m.predict(x_input, verbose=0)[0] for m in model]
        yhat = np.array(yhat).mean(axis=0)
    else:
        yhat = model.predict(x_input, verbose=0)[0]
    return yhat
