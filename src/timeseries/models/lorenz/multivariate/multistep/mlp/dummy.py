import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from timeseries.models.lorenz.functions.dataprep import multi_step_xy_from_mv
from numpy import array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


if __name__ == '__main__':
    # define input sequence
    in_seq1 = array([10, 20, 30, 40, 50, 60, 70, 80, 90])
    in_seq2 = array([15, 25, 35, 45, 55, 65, 75, 85, 95])
    out_seq = array([in_seq1[i]+in_seq2[i] for i in range(len(in_seq1))])

    seqs = [in_seq1, in_seq2, out_seq]

    # choose a number of time steps
    n_steps_in, n_steps_out = 3, 2
    # convert into input/output
    X, y = multi_step_xy_from_mv(seqs, n_steps_in, n_steps_out)
    n_input = X.shape[0]
    # define model
    model = Sequential()
    model.add(Dense(100, activation='relu', input_dim=n_input))
    model.add(Dense(n_steps_out))
    model.compile(optimizer='adam', loss='mse')
    # fit model
    model.fit(X, y, epochs=2000, verbose=0)
    # demonstrate prediction
    x_input = array([[70, 75], [80, 85], [90, 95]])
    x_input = x_input.reshape((1, n_input))
    yhat = model.predict(x_input, verbose=0)
    print(yhat)