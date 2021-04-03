import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from timeseries.models.lorenz.functions.dataprep import split_uv_seq_multi_step
from numpy import array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


if __name__ == '__main__':
    # define input sequence
    seq = [10, 20, 30, 40, 50, 60, 70, 80, 90]

    # choose a number of time steps
    n_steps_in, n_steps_out = 3, 2
    # convert into input/output
    X, y = split_uv_seq_multi_step(seq, n_steps_in, n_steps_out)
    # define model
    model = Sequential()
    model.add(Dense(100, activation='relu', input_dim=n_steps_in))
    model.add(Dense(n_steps_out))
    model.compile(optimizer='adam', loss='mse')
    # fit model
    model.fit(X, y, epochs=2000, verbose=0)
    # demonstrate prediction
    x_input = array([70, 80, 90])
    x_input = x_input.reshape((1, n_steps_in))
    yhat = model.predict(x_input, verbose=0)
    print(yhat)