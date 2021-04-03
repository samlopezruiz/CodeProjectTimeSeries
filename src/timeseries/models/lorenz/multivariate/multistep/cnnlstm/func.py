import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense, TimeDistributed, Conv1D, MaxPooling1D, Flatten
from timeseries.models.lorenz.functions.dataprep import step_feature_one_step_xy_from_uv, \
    step_feature_one_step_xy_from_mv, step_feature_multi_step_xy_from_mv
from numpy import array


def cnnlstm_multi_step_mv_fit(train, cfg):
    # unpack config
    n_seq, n_steps, n_steps_out, n_filters = cfg['n_seq'], cfg['n_steps_in'], cfg['n_steps_out'], cfg['n_filters']
    n_kernel, n_nodes, n_epochs, n_batch = cfg['n_kernel'], cfg['n_nodes'], cfg['n_epochs'], cfg['n_batch']
    n_input = n_seq * n_steps

    # prepare data
    X, y = step_feature_multi_step_xy_from_mv(train, n_input, n_steps_out, n_seq)
    n_features = X.shape[3]
    # define model
    model = Sequential()
    model.add(TimeDistributed(Conv1D(n_filters, n_kernel, activation='relu', input_shape=(None, n_steps, n_features))))
    model.add(TimeDistributed(Conv1D(n_filters, n_kernel, activation='relu')))
    model.add(TimeDistributed(MaxPooling1D()))
    model.add(TimeDistributed(Flatten()))
    model.add(LSTM(n_nodes, activation='relu'))
    model.add(Dense(n_nodes, activation='relu'))
    model.add(Dense(n_steps_out))
    model.compile(loss='mse', optimizer='adam')
    # fit
    model.fit(X, y, epochs=n_epochs, batch_size=n_batch, verbose=0)
    return model


# forecast with a pre-fit model
def cnnlstm_multi_step_mv_predict(model, history, cfg, steps=1):
    # unpack config
    n_seq, n_steps = cfg['n_seq'], cfg['n_steps_in']
    n_features = history.shape[1]
    n_input = n_seq * n_steps
    # prepare data
    x_input = array(history[-n_input:]).reshape((1, n_seq, n_steps, n_features))
    # forecast
    yhat = model.predict(x_input, verbose=0)
    return yhat[0]
