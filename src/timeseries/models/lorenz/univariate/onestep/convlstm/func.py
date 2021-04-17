import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from timeseries.models.lorenz.functions.dataprep import row_col_one_step_xy_from_uv
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import ConvLSTM2D, Dense, Flatten
from numpy import array


def convlstm_one_step_uv_fit(train, cfg, plot_hist=False, verbose=0):
    # unpack config
    n_seq, n_steps, n_filters = cfg['n_seq'], cfg['n_steps_in'], cfg['n_filters']
    n_kernel, n_nodes, n_epochs, n_batch = cfg['n_kernel'], cfg['n_nodes'], cfg['n_epochs'], cfg['n_batch']
    n_input = n_seq * n_steps
    # prepare data
    X, y = row_col_one_step_xy_from_uv(train, n_input, 1, n_seq)
    # define model
    model = Sequential()
    model.add(ConvLSTM2D(n_filters, (1, n_kernel), activation='relu', input_shape=(n_seq, 1, n_steps, 1)))
    model.add(Flatten())
    model.add(Dense(n_nodes, activation='relu'))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    # fit
    model.fit(X, y, epochs=n_epochs, batch_size=n_batch, verbose=verbose)
    return model


# forecast with a pre-fit model
def convlstm_one_step_uv_predict(model, history, cfg, steps=1):
    # unpack config
    n_seq, n_steps = cfg['n_seq'], cfg['n_steps_in']
    n_input = n_seq * n_steps
    # prepare data
    x_input = array(history[-n_input:]).reshape((1, n_seq, 1, n_steps, 1))
    # forecast
    yhat = model.predict(x_input, verbose=0)
    return yhat[0][0]
