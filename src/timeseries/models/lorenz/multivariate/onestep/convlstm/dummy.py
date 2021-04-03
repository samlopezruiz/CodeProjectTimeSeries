import os
from timeseries.models.lorenz.functions.dataprep import row_col_one_step_xy_from_mv
import numpy as np
from numpy import array
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, ConvLSTM2D


if __name__ == '__main__':
    in_seq1 = array([10, 20, 30, 40, 50, 60, 70, 80, 90])
    in_seq2 = array([15, 25, 35, 45, 55, 65, 75, 85, 95])
    out_seq = array([in_seq1[i] + in_seq2[i] for i in range(len(in_seq1))])
    seqs = [in_seq1, in_seq2, out_seq]

    # split into samples
    dataset = np.vstack(seqs).transpose()
    n_steps = 4
    n_seq = 2
    n_steps_in = int(n_steps / n_seq)
    # split into samples
    X, y = row_col_one_step_xy_from_mv(seqs, n_steps, n_seq)
    n_features = X.shape[3]
    # define model
    model = Sequential()
    model.add(ConvLSTM2D(64, (1, 2), activation='relu', input_shape=(n_seq, 1, n_steps_in, n_features)))
    model.add(Flatten())
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    # fit model
    model.fit(X, y, epochs=500, verbose=0)
    # demonstrate prediction

    x_input = array([[60, 65], [70, 75], [80, 85], [90, 95]])
    x_input = x_input.reshape((1, n_seq, 1, n_steps_in, n_features))
    yhat = model.predict(x_input, verbose=0)
    print(yhat)