import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from numpy import array
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense, TimeDistributed, Conv1D, MaxPooling1D, Flatten
from timeseries.experiments.lorenz.functions.dataprep import step_feature_one_step_xy_from_uv

if __name__ == '__main__':
    # define input sequence
    seq = [10, 20, 30, 40, 50, 60, 70, 80, 90]
    # choose a number of time steps
    n_steps = 4
    n_features = 1
    n_seq = 2
    n_steps_in = int(n_steps / n_seq)
    # split into samples
    X, y = step_feature_one_step_xy_from_uv(seq, n_steps, n_features, n_seq)
    # define model
    model = Sequential()
    model.add(TimeDistributed(Conv1D(64, 1, activation='relu'), input_shape=(None, n_steps_in, n_features)))
    model.add(TimeDistributed(MaxPooling1D()))
    model.add(TimeDistributed(Flatten()))
    model.add(LSTM(50, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    # fit model
    model.fit(X, y, epochs=500, verbose=0)
    # demonstrate prediction
    x_input = array([60, 70, 80, 90])
    x_input = x_input.reshape((1, n_seq, n_steps_in, n_features))
    yhat = model.predict(x_input, verbose=0)
    print(yhat)