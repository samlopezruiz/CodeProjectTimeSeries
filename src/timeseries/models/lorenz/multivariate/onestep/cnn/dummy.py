import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from timeseries.models.lorenz.functions.dataprep import feature_one_step_xy_from_mv
from numpy import array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv1D, MaxPooling1D


if __name__ == '__main__':
    in_seq1 = array([10, 20, 30, 40, 50, 60, 70, 80, 90])
    in_seq2 = array([15, 25, 35, 45, 55, 65, 75, 85, 95])
    out_seq = array([in_seq1[i] + in_seq2[i] for i in range(len(in_seq1))])
    seqs = [in_seq1, in_seq2, out_seq]

    # choose a number of time steps
    n_steps = 3
    # split into samples
    X, y = feature_one_step_xy_from_mv(seqs, n_steps)
    # define model
    n_features = X.shape[2]

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
    x_input = array([[80, 85], [90, 95], [100, 105]])
    x_input = x_input.reshape((1, n_steps, n_features))
    yhat = model.predict(x_input, verbose=0)
    print(yhat)