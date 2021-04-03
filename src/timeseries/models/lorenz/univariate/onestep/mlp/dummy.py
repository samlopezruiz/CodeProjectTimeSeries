import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from timeseries.models.lorenz.functions.dataprep import split_uv_seq_one_step
from numpy import array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


if __name__ == '__main__':
    raw_seq = [10, 20, 30, 40, 50, 60, 70, 80, 90]
    # choose a number of time steps
    n_steps = 3
    # split into samples
    X, y = split_uv_seq_one_step(raw_seq, n_steps)
    n_input = X.shape[0]
    # define model
    model = Sequential()
    model.add(Dense(100, activation='relu', input_dim=n_steps))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    # fit model
    model.fit(X, y, epochs=2000, verbose=0)
    # demonstrate prediction
    x_input = array([70, 80, 90])
    x_input = x_input.reshape((1, n_steps))
    yhat = model.predict(x_input, verbose=0)
    print(yhat)