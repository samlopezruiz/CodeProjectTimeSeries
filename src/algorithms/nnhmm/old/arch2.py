import os

from algorithms.nnhmm.func import build_nnhmm_model

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# %%


def intermediate_layers(input_, cfg):
    n_steps_out = cfg['n_steps_out']
    hidden = keras.layers.Dense(30, activation="relu")(input_)
    output_ = keras.layers.Dense(n_steps_out, activation="relu")(hidden)
    return output_


def input_shape_fn(cfg, n_features):
    n_steps_in = cfg['n_steps_in']
    return n_steps_in


if __name__ == '__main__':
    # %% DATA
    housing = fetch_california_housing()
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        housing.data, housing.target.reshape(-1, 1), random_state=42)
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train_full, y_train_full, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_valid_scaled = scaler.transform(X_valid)
    X_test_scaled = scaler.transform(X_test)

    # %% ARCHITECTURE
    n_states = 3
    model_funcs = {'intermediate_layers': intermediate_layers, 'input_shape': input_shape_fn}
    model_fn = (intermediate_layers, input_shape_fn)
    cfg = {'n_steps_in': X_train.shape[1:], 'n_steps_out': 1}
    n_features = X_test_scaled.shape[1]
    model = build_nnhmm_model(cfg, n_states, n_features, model_funcs, use_regimes=True)

    model.summary()
    tf.keras.utils.plot_model(
        model, to_file='model.png', show_shapes=True, show_dtype=False,
        show_layer_names=True, rankdir='TB', expand_nested=False, dpi=96
    )

    # %%
    reg_prob_train = np.full((X_train_scaled.shape[0], n_states), 0.5)
    model_input_train = [X_train_scaled, reg_prob_train]
    history = model.fit(model_input_train, y_train, epochs=20, batch_size=32, verbose=1)
    print('\n----\nFinal Loss: {:.4f}'.format(history.history['loss'][-1]))

    # %%
    reg_prob_test = np.full((1, n_states), 0.5)
    model_input_test = [X_test_scaled[0, :].reshape(1, -1), reg_prob_test]
    y_hat = model.predict(model_input_test)[0]
