import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from numpy import array
from tensorflow.keras import Sequential
from tensorflow.keras.layers import ConvLSTM2D, Dense, Flatten
from timeseries.experiments.lorenz.functions.dataprep import row_col_multi_step_xy_from_uv

if __name__ == '__main__':
    # define input sequence
    seq = [10, 20, 30, 40, 50, 60, 70, 80, 90]
    # choose a number of time steps
    n_steps_in, n_steps_out = 4, 2
    n_features = 1
    n_seq = 2
    n_steps = int(n_steps_in / n_seq)
    # split into samples
    X, y = row_col_multi_step_xy_from_uv(seq, n_steps_in, n_steps_out, n_features, n_seq)
    # define model
    model = Sequential()
    model.add(ConvLSTM2D(64, (1, 2), activation='relu', input_shape=(n_seq, 1, n_steps, n_features)))
    model.add(Flatten())
    model.add(Dense(n_steps_out))
    model.compile(optimizer='adam', loss='mse')
    # fit model
    model.fit(X, y, epochs=500, verbose=0)
    # demonstrate prediction
    x_input = array([60, 70, 80, 90])
    x_input = x_input.reshape((1, n_seq, 1, n_steps, n_features))
    yhat = model.predict(x_input, verbose=0)
    print(yhat)