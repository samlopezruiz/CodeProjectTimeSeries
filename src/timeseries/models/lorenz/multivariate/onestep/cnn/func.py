import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from timeseries.models.lorenz.functions.dataprep import feature_one_step_xy_from_mv
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv1D, MaxPooling1D
from numpy import array


def cnn_one_step_mv_fit(train, cfg):
    # unpack config
    n_steps_in, n_filters = cfg['n_steps_in'], cfg['n_filters']
    n_kernel, n_epochs, n_batch = cfg['n_kernel'], cfg['n_epochs'], cfg['n_batch']
    # prepare data
    X, y = feature_one_step_xy_from_mv(train, n_steps_in)
    n_features = X.shape[2]
    # define model
    model = Sequential()
    model.add(Conv1D(n_filters, n_kernel, activation='relu', input_shape=(n_steps_in, n_features)))
    model.add(Conv1D(n_filters, n_kernel, activation='relu'))
    model.add(MaxPooling1D())
    model.add(Flatten())
    model.add(Dense(50, activation='relu'))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    # fit
    model.fit(X, y, epochs=n_epochs, batch_size=n_batch, verbose=0)
    return model


# forecast with a pre-fit model
def cnn_one_step_mv_predict(model, history, cfg, steps=1):
    # unpack config
    n_steps = cfg['n_steps_in']
    n_features = history.shape[1]
    # prepare data
    x_input = array(history[-n_steps:]).reshape(1, n_steps, n_features)
    # forecast
    yhat = model.predict(x_input, verbose=0)
    return yhat[0][0]
