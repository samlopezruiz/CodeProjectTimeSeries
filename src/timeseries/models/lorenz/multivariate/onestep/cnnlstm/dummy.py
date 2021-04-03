import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from numpy import array
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense, TimeDistributed, Conv1D, MaxPooling1D, Flatten
from timeseries.models.lorenz.functions.dataprep import step_feature_one_step_xy_from_mv

if __name__ == '__main__':
    in_seq1 = array([10, 20, 30, 40, 50, 60, 70, 80, 90])
    in_seq2 = array([15, 25, 35, 45, 55, 65, 75, 85, 95])
    out_seq = array([in_seq1[i] + in_seq2[i] for i in range(len(in_seq1))])
    seqs = [in_seq1, in_seq2, out_seq]

    # choose a number of time steps
    n_steps = 4
    n_seq = 2
    n_steps_in = int(n_steps / n_seq)
    # split into samples
    X, y = step_feature_one_step_xy_from_mv(seqs, n_steps, n_seq)
    n_features = X.shape[3]
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
    x_input = array([[60, 65], [70, 75], [80, 85], [90, 95]])
    x_input = x_input.reshape((1, n_seq, n_steps_in, n_features))
    yhat = model.predict(x_input, verbose=0)
    print(yhat)