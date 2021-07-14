import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from timeseries.models.lorenz.functions.dataprep import feature_one_step_xy_from_uv, feature_one_step_xy_from_mv, \
    split_mv_seq_one_step
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from numpy import array


def lstm_one_step_mv_fit(train, cfg):
    n_input, n_nodes, n_epochs, n_batch = cfg['n_steps_in'], cfg['n_nodes'], cfg['n_epochs'], cfg['n_batch']
    # prepare data
    X, y = split_mv_seq_one_step(train, n_input)
    n_features = X.shape[2]
    # define model
    model = Sequential()
    # model.add(LSTM(50, activation='relu', return_sequences=True, input_shape=(n_input, 1)))
    # model.add(LSTM(50, activation='relu'))
    model.add(LSTM(n_nodes, activation='relu', input_shape=(n_input, n_features)))
    model.add(Dense(n_nodes, activation='relu'))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    # fit
    model.fit(X, y, epochs=n_epochs, batch_size=n_batch, verbose=0)
    return model


# forecast with a pre-fit model
def lstm_one_step_mv_predict(model, history, cfg, steps=1):
    # unpack architectures
    n_input = cfg['n_steps_in']
    n_features = history.shape[1]
    # prepare data
    x_input = array(history[-n_input:]).reshape((1, n_input, n_features))
    # forecast
    yhat = model.predict(x_input, verbose=0)
    return yhat[0][0]
