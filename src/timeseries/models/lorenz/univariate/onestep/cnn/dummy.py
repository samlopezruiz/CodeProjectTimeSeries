import os
from timeseries.models.lorenz.functions.dataprep import feature_one_step_xy_from_uv
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from numpy import array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv1D, MaxPooling1D


if __name__ == '__main__':
    # define input sequence
    raw_seq = [10, 20, 30, 40, 50, 60, 70, 80, 90]
    # choose a number of time steps
    n_steps = 3
    # split into samples
    X, y = feature_one_step_xy_from_uv(raw_seq, n_steps)

    n_features = 1
    # define model
    model = Sequential()
    model.add(Conv1D(64, 2, activation='relu', input_shape=(n_steps, n_features)))
    model.add(MaxPooling1D())
    model.add(Flatten())
    model.add(Dense(50, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    # fit model
    model.fit(X, y, epochs=1000, verbose=0)
    # demonstrate prediction
    x_input = array([70, 80, 90])
    x_input = x_input.reshape((1, n_steps, n_features))
    yhat = model.predict(x_input, verbose=0)
    print(yhat)