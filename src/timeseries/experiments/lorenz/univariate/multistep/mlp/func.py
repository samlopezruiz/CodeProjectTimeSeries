import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from timeseries.experiments.lorenz.functions.dataprep import split_uv_seq_multi_step
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from numpy import array


def mlp_multi_step_uv_fit(train, cfg):
    # unpack architectures
    n_steps_in, n_steps_out = cfg['n_steps_in'], cfg['n_steps_out']
    n_nodes, n_epochs, n_batch = cfg['n_nodes'], cfg['n_epochs'], cfg['n_batch']
    # prepare data
    X, y = split_uv_seq_multi_step(train, n_steps_in, n_steps_out)
    # define model
    model = Sequential()
    model.add(Dense(n_nodes, activation='relu', input_dim=n_steps_in))
    model.add(Dense(n_steps_out))
    model.compile(loss='mse', optimizer='adam')
    # fit
    model.fit(X, y, epochs=n_epochs, batch_size=n_batch, verbose=0)
    return model


# forecast with a pre-fit model
def mlp_multi_step_uv_predict(model, history, cfg, steps=1):
    # unpack architectures
    n_input = cfg['n_steps_in']
    # prepare data
    x_input = array(history[-n_input:]).reshape(1, n_input)
    # forecast
    yhat = model.predict(x_input, verbose=0)
    return yhat[0]


