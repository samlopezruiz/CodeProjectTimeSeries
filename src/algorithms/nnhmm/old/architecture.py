import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
#%%

if __name__ == '__main__':
    #%% DATA
    housing = fetch_california_housing()
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        housing.data, housing.target.reshape(-1, 1), random_state=42)
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train_full, y_train_full, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_valid_scaled = scaler.transform(X_valid)
    X_test_scaled = scaler.transform(X_test)

    #%% ARCHITECTURE
    # weights = tf.constant([0.5, 0.5, 0.5, 0.5])
    input_shape = X_train.shape[1:]
    output_shape = 2
    n_states = 3

    probs_ = keras.layers.Input(shape=n_states, name='weights')
    input_ = keras.layers.Input(shape=input_shape, name='input')

    def intermediate_layers(input_, output_shape):
        hid = keras.layers.Dense(30, activation="relu")(input_)
        output_ = keras.layers.Dense(output_shape, activation="relu")(hid)
        return output_

    mul = []
    for i in range(n_states):
        # weights_sliced = keras.layers.Lambda(lambda x: x[:, i*output_shape:i*output_shape+output_shape])(weights_)
        weights_sliced = keras.layers.Lambda(lambda x: x[:, i])(probs_)
        out = intermediate_layers(input_, output_shape)
        mul.append(keras.layers.Lambda(lambda x: x * weights_sliced)(out))
        # mul.append(keras.layers.Multiply()([weights_sliced, out]))

    output = keras.layers.Add()(mul)
    model = keras.Model(inputs=[input_, probs_], outputs=[output])

    model.summary()
    tf.keras.utils.plot_model(
        model, to_file='model.png', show_shapes=True, show_dtype=False,
        show_layer_names=True, rankdir='TB', expand_nested=False, dpi=96
    )
    model.compile(loss='mse')

    #%%
    w = tf.constant(np.full((1, 1), 6.))
    o = tf.constant(np.full((1, n_states), 2.))
    keras.layers.Multiply()([w, o])

    #%%
    # hidden1a = keras.layers.Dense(30, activation="relu")(input_)
    # hidden1b = keras.layers.Dense(30, activation="relu")(input_)
    # hidden2a = tf.keras.layers.Lambda(lambda x: x * weights_[0])(hidden1a)
    # hidden2b = tf.keras.layers.Lambda(lambda x: x * weights_[1])(hidden1b)
    # intermediate = tf.keras.layers.Reshape((output_shape*n_states, 1), input_shape=(output_shape*n_states,))(weights_)
    # first_half = tf.keras.layers.Cropping1D(cropping=(0, 2))(intermediate)
    # first_half = tf.keras.layers.Reshape((2,), input_shape=(2, 1))(first_half)
    # second_half = tf.keras.layers.Cropping1D(cropping=(2, 0))(intermediate)
    # second_half = tf.keras.layers.Reshape((2,), input_shape=(2, 1))(second_half)

    # hidden2a = keras.layers.Dense(2)(hidden1a)
    # hidden2b = keras.layers.Dense(2)(hidden1b)
    # output1 = keras.layers.Multiply()([hidden2a, first_half])
    # output2 = keras.layers.Multiply()([hidden2b, second_half])

    # concat = keras.layers.Concatenate()([output1, output2])
    # output = keras.layers.Multiply()([concat, weights_])
    # output = keras.layers.Add()([output1, output2])

    #%%
    weights = np.full((X_train_scaled.shape[0], n_states), 0.5)
    history = model.fit([X_train_scaled, weights], y_train, epochs=20, batch_size=32, verbose=1)
    print('\n----\nFinal Loss: {:.4f}'.format(history.history['loss'][-1]))
