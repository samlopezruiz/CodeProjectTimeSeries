import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from timeseries.experiments.lorenz.functions.dataprep import feature_one_step_xy_from_uv, feature_multi_step_xy_from_uv
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from numpy import array


def lstm_multi_step_uv_fit(train, config):
    # unpack architectures
    n_input, n_output, n_nodes, n_epochs, n_batch = config
    # prepare data
    X, y = feature_multi_step_xy_from_uv(train, n_input, n_output)
    # define model
    model = Sequential()
    # model.add(LSTM(50, activation='relu', return_sequences=True, input_shape=(n_input, 1)))
    # model.add(LSTM(50, activation='relu'))
    model.add(LSTM(n_nodes, activation='relu', input_shape=(n_input, 1)))
    model.add(Dense(n_nodes, activation='relu'))
    model.add(Dense(n_output))
    model.compile(loss='mse', optimizer='adam')
    # fit
    model.fit(X, y, epochs=n_epochs, batch_size=n_batch, verbose=0)
    return model


# forecast with a pre-fit model
def lstm_multi_step_uv_predict(model, history, cfg, steps=1):
    # unpack architectures
    n_input, _, _, _, _ = cfg
    # prepare data
    x_input = array(history[-n_input:]).reshape((1, n_input, 1))
    # forecast
    yhat = model.predict(x_input, verbose=0)
    return yhat[0]
