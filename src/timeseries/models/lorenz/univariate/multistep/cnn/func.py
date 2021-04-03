import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from timeseries.models.lorenz.functions.dataprep import feature_one_step_xy_from_uv, feature_multi_step_xy_from_uv
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv1D, MaxPooling1D
from numpy import array


def cnn_multi_step_uv_fit(train, cfg):
    # unpack config
    n_steps_in, n_steps_out, n_filters = cfg['n_steps_in'], cfg['n_steps_out'], cfg['n_filters']
    n_kernel, n_epochs, n_batch = cfg['n_kernel'], cfg['n_epochs'], cfg['n_batch']
    # prepare data
    X, y = feature_multi_step_xy_from_uv(train, n_steps_in, n_steps_out)
    # define model
    model = Sequential()
    model.add(Conv1D(n_filters, n_kernel, activation='relu', input_shape=(n_steps_in, 1)))
    model.add(Conv1D(n_filters, n_kernel, activation='relu'))
    model.add(MaxPooling1D())
    model.add(Flatten())
    model.add(Dense(50, activation='relu'))
    model.add(Dense(n_steps_out))
    model.compile(loss='mse', optimizer='adam')
    # fit
    model.fit(X, y, epochs=n_epochs, batch_size=n_batch, verbose=0)
    return model


# forecast with a pre-fit model
def cnn_multi_step_uv_predict(model, history, cfg, steps=1):
    # unpack config
    n_input = cfg['n_steps_in']
    # prepare data
    x_input = array(history[-n_input:]).reshape(1, n_input, 1)
    # forecast
    yhat = model.predict(x_input, verbose=0)
    return yhat[0]
