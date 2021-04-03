import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from timeseries.models.lorenz.functions.dataprep import split_uv_seq_one_step
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from numpy import array


def mlp_one_step_uv_fit(train, cfg):
    # unpack config
    n_steps_in, n_nodes, n_epochs, n_batch = (cfg['n_steps_in'], cfg['n_nodes'],
                                              cfg['n_epochs'], cfg['n_batch'])
    # prepare data
    X, y = split_uv_seq_one_step(train, n_steps_in)
    # define model
    model = Sequential()
    model.add(Dense(n_nodes, activation='relu', input_dim=n_steps_in))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    # fit
    model.fit(X, y, epochs=n_epochs, batch_size=n_batch, verbose=0)
    return model


# forecast with a pre-fit model
def mlp_one_step_uv_predict(model, history, cfg, steps=1):
    # unpack config
    n_input = cfg['n_steps_in']
    # prepare data
    x_input = array(history[-n_input:]).reshape(1, n_input)
    # forecast
    yhat = model.predict(x_input, verbose=0)
    return yhat[0][0]


def mlp_configs():
    # define scope of configs
    n_input = [12]
    n_nodes = [50, 100]
    n_epochs = [100]
    n_batch = [1, 150]
    # create configs
    configs = list()
    for i in n_input:
        for j in n_nodes:
            for k in n_epochs:
                for l in n_batch:
                    cfg = (i, j, k, l)
                    configs.append(cfg)
    print('Total configs: %d' % len(configs))
    return configs
