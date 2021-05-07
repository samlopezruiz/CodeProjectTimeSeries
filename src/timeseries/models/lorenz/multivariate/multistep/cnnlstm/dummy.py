import os
from timeseries.models.lorenz.functions.dataprep import row_col_one_step_xy_from_mv, row_col_multi_step_xy_from_mv, \
    step_feature_multi_step_xy_from_mv, split_mv_seq_multi_step
import numpy as np
from numpy import array
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, TimeDistributed, Conv1D, MaxPooling1D, Flatten


if __name__ == '__main__':
    in_seq1 = array([10, 20, 30, 40, 50, 60, 70, 80, 90])
    in_seq2 = array([15, 25, 35, 45, 55, 65, 75, 85, 95])
    out_seq = array([in_seq1[i] + in_seq2[i] for i in range(len(in_seq1))])
    seqs = [in_seq1, in_seq2, out_seq]


    # split into samples
    dataset = np.vstack(seqs).transpose()
    n_steps_in, n_steps_out = 4, 2
    n_seq = 2
    n_steps = int(n_steps_in / n_seq)
    # split into samples
    X, y = split_mv_seq_multi_step(seqs, n_steps_in, n_steps_out)

    X, y = step_feature_multi_step_xy_from_mv(seqs, n_steps_in, n_steps_out, n_seq)
    n_features = X.shape[3]
    # define model
    model = Sequential()
    model.add(TimeDistributed(Conv1D(64, 1, activation='relu'), input_shape=(None, n_steps, n_features)))
    model.add(TimeDistributed(MaxPooling1D()))
    model.add(TimeDistributed(Flatten()))
    model.add(LSTM(50, activation='relu'))
    model.add(Dense(n_steps_out))
    model.compile(optimizer='adam', loss='mse')
    # fit model
    model.fit(X, y, epochs=500, verbose=0)
    # demonstrate prediction

    x_input = array([[60, 65], [70, 75], [80, 85], [90, 95]])
    x_input = x_input.reshape((1, n_seq, n_steps, n_features))
    yhat = model.predict(x_input, verbose=0)
    print(yhat)